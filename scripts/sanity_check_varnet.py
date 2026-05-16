# Autor: Massanori
# Data: 14/05/2026
# Descrição: Sanity check da VarNet brain 4x oficial antes da pré-computação
#            em escala. Recebe: caminho do checkpoint, caminho dos .h5 brain,
#            lista de 5 volume IDs (default: 2 AXFLAIR + 2 AXT1 + 1 AXT1POST
#            do val.txt), device (auto-detecta CUDA). Retorna: relatório
#            JSON com SSIM por volume + estatísticas, mais print legível.
#            Roda inferência slice-a-slice, agrupa por volume, empilha o
#            volume 3D, aplica center crop para o shape do target, computa
#            SSIM via fastmri.evaluate.ssim. Status do gate: PASSED se SSIM
#            médio >= 0.94, WARNING entre 0.85-0.94, FAILED abaixo. Exit
#            code reflete status (0/1/2). Cache opcional dos recons em
#            <output>/cache/{stem}.npz para evitar refazer inferência se o
#            cálculo posterior falhar — passe --use-cache para ler/escrever.
#            Use:
#            python scripts/sanity_check_varnet.py
#            ou no Kaggle: python scripts/sanity_check_varnet.py --device cuda


"""Sanity check da VarNet brain 4x - gate antes da pre-computacao em escala.

Valida que o pipeline (checkpoint + dataset + transform + mascara) produz
SSIM compativel com o reportado no paper (Sriram et al. 2020 reporta ~0.966
no brain 4x leaderboard) em 5 volumes do val.txt cobrindo as 3 modalidades
(AXFLAIR, AXT1, AXT1POST).

Estrategia:
    1. Carrega checkpoint via src.models.load_pretrained_varnet (eval, no-grad)
    2. Constroi dataset via src.data.kspace_dataset.build_brain_kspace_dataset
       no Modo B (volume_ids explicitos)
    3. Itera slice-a-slice com batch_size=1, num_workers=0 (simples e
       reproduzivel)
    4. Para cada slice: forward VarNet, center_crop para shape do target,
       acumula em dict[stem] -> list[(slice_num, recon)]
    5. (Opcional) salva cache .npz por volume em <output>/cache/{stem}.npz
       para evitar refazer inferencia se o passo 6 falhar
    6. Apos todos os slices: ordena por slice_num, empilha em volume 3D,
       computa SSIM via fastmri.evaluate.ssim contra o target 3D
    7. Reporta media/std/min/max e classifica status do gate
    8. Salva JSON em <output>/sanity_report.json
    9. Exit code: 0 (PASSED), 1 (FAILED), 2 (WARNING)

Refs:
    Sriram, A. et al. (2020). End-to-End Variational Networks for
        Accelerated MRI Reconstruction. MICCAI.
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from fastmri.data import transforms as T
from fastmri.evaluate import ssim as ssim_fn

# Adiciona a raiz do repo ao sys.path para importar src.* quando o script
# e chamado diretamente (python scripts/sanity_check_varnet.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.data.kspace_dataset import build_brain_kspace_dataset  # noqa: E402
from src.models import load_pretrained_varnet  # noqa: E402


# 5 volumes do val.txt cobrindo as 3 modalidades brain.
DEFAULT_VOLUMES = (
    'file_brain_AXFLAIR_200_6002460',
    'file_brain_AXFLAIR_201_6002878',
    'file_brain_AXT1_201_6002804',
    'file_brain_AXT1_202_2020098',
    'file_brain_AXT1POST_200_6002033',
)

# Path relativo a raiz do repo, resolvido na main.
DEFAULT_CHECKPOINT_RELPATH = 'checkpoints/brain_leaderboard_state_dict.pt'

# Sriram et al. 2020 reporta ~0.966 no brain 4x leaderboard. Os thresholds
# abaixo dao margem generosa para diferencas de subset de teste e detalhes
# de pre-processamento.
PASS_THRESHOLD = 0.94
WARN_THRESHOLD = 0.85

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Sanity check da VarNet brain 4x em 5 volumes do val.',
    )
    parser.add_argument(
        '--checkpoint',
        type=Path,
        default=None,
        help=f'Path do .pt. Default: <repo>/{DEFAULT_CHECKPOINT_RELPATH}',
    )
    parser.add_argument(
        '--data-path',
        type=Path,
        default=None,
        help='Diretorio com os .h5 brain. Default: config.anotados_dir().',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help='Diretorio para sanity_report.json. Default: '
             '<config.recons_dir()>/sanity/',
    )
    parser.add_argument(
        '--volumes',
        nargs='+',
        default=list(DEFAULT_VOLUMES),
        metavar='VOLUME_ID',
        help='IDs dos volumes (sem extensao .h5). Default: 5 volumes '
             'cobrindo as 3 modalidades brain.',
    )
    parser.add_argument(
        '--device',
        default=None,
        help='cpu, cuda, ou auto-detect (default).',
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=0,
        help='Workers do DataLoader. Default 0 (single-process, mais '
             'reproduzivel e suficiente para 5 volumes).',
    )
    parser.add_argument(
        '--use-cache',
        action='store_true',
        help='Se passado, salva recons/targets em <output>/cache/{stem}.npz '
             'apos inferencia e tenta ler de la antes de rodar VarNet. Util '
             'para iterar o calculo de SSIM sem refazer inferencia.',
    )
    return parser.parse_args()


def resolve_device(arg_device: Optional[str]) -> torch.device:
    """Auto-detecta CUDA se nada foi passado."""
    if arg_device:
        return torch.device(arg_device)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _to_scalar(x):
    """Converte tensor 0-dim ou lista de 1 elemento para python escalar."""
    if hasattr(x, 'item'):
        return x.item()
    if isinstance(x, (list, tuple)):
        return _to_scalar(x[0])
    return x


def _extract_fname(batch_fname) -> str:
    """fname vem como lista de strings apos default_collate (batch_size=1)."""
    if isinstance(batch_fname, (list, tuple)):
        return str(batch_fname[0])
    return str(batch_fname)


def _cache_path(cache_dir: Path, stem: str) -> Path:
    return cache_dir / f'{stem}.npz'


def _load_cached_volumes(cache_dir: Path, volume_stems):
    """Tenta carregar recons/targets/max_val do cache. Retorna (recons,
    targets, max_values) parcialmente preenchidos: stems que faltam no
    cache nao aparecem nos dicts.
    """
    recons = defaultdict(list)
    targets = defaultdict(list)
    max_values = defaultdict(list)

    if not cache_dir.is_dir():
        return recons, targets, max_values

    hits = 0
    for stem in volume_stems:
        p = _cache_path(cache_dir, stem)
        if not p.is_file():
            continue
        try:
            data = np.load(p)
            recon_vol = data['recon_vol']
            target_vol = data['target_vol']
            max_val = float(data['max_val'])
        except Exception as e:
            logger.warning(f'Cache corrompido em {p}: {e} - ignorando.')
            continue

        for slice_num in range(recon_vol.shape[0]):
            recons[stem].append((slice_num, recon_vol[slice_num]))
            targets[stem].append((slice_num, target_vol[slice_num]))
        max_values[stem].append(max_val)
        hits += 1

    if hits:
        logger.info(f'Cache: {hits}/{len(volume_stems)} volumes carregados.')
    return recons, targets, max_values


def _save_cache(cache_dir: Path, recons, targets, max_values):
    """Salva cada volume como {cache_dir}/{stem}.npz."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    for stem in recons:
        sorted_recons = sorted(recons[stem], key=lambda x: x[0])
        sorted_targets = sorted(targets[stem], key=lambda x: x[0])
        recon_vol = np.stack([r for _, r in sorted_recons])
        target_vol = np.stack([t for _, t in sorted_targets])
        max_val = float(max(max_values[stem]))

        out = _cache_path(cache_dir, stem)
        np.savez_compressed(
            out,
            recon_vol=recon_vol.astype(np.float32),
            target_vol=target_vol.astype(np.float32),
            max_val=np.float32(max_val),
        )
    logger.info(f'Cache salvo em {cache_dir} ({len(recons)} volumes).')


def run_inference(
    model: torch.nn.Module,
    dataset,
    device: torch.device,
    num_workers: int,
):
    """Itera o dataset slice-a-slice, agrupando por volume."""
    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=num_workers,
        shuffle=False,
    )

    recons = defaultdict(list)
    targets = defaultdict(list)
    max_values = defaultdict(list)

    with torch.no_grad():
        for batch in tqdm(loader, desc='Inferencia VarNet'):
            masked_kspace = batch.masked_kspace.to(device)
            mask = batch.mask.to(device)

            fname = _extract_fname(batch.fname)
            stem = Path(fname).stem
            slice_num = int(_to_scalar(batch.slice_num))
            max_val = float(_to_scalar(batch.max_value))

            crop_size = (
                int(_to_scalar(batch.crop_size[0])),
                int(_to_scalar(batch.crop_size[1])),
            )

            output = model(masked_kspace, mask)
            output = T.center_crop(output, crop_size)
            output_np = output.cpu().squeeze(0).numpy()

            target_t = batch.target
            target_np = (
                target_t.squeeze(0).numpy()
                if torch.is_tensor(target_t)
                else np.asarray(target_t)
            )

            recons[stem].append((slice_num, output_np))
            targets[stem].append((slice_num, target_np))
            max_values[stem].append(max_val)

    return recons, targets, max_values


def compute_ssim_per_volume(recons, targets, max_values):
    """Empilha slices por volume e calcula SSIM no volume 3D.

    Robusto a fastmri.evaluate.ssim retornar tanto escalar (versoes antigas
    da lib) quanto array (versoes novas) — em NumPy 1.25+, float(np.array([x]))
    levanta TypeError, dai a passagem por np.asarray().mean().
    """
    results = {}
    for stem in recons:
        sorted_recons = sorted(recons[stem], key=lambda x: x[0])
        sorted_targets = sorted(targets[stem], key=lambda x: x[0])

        recon_vol = np.stack([r for _, r in sorted_recons])
        target_vol = np.stack([t for _, t in sorted_targets])
        max_val = float(max(max_values[stem]))

        if recon_vol.shape != target_vol.shape:
            raise RuntimeError(
                f'Shape mismatch em {stem}: recon={recon_vol.shape}, '
                f'target={target_vol.shape}'
            )

        ssim_raw = ssim_fn(target_vol, recon_vol, maxval=max_val)
        ssim_arr = np.asarray(ssim_raw)
        ssim_value = float(ssim_arr.mean())
        logger.debug(
            f'{stem}: ssim_raw type={type(ssim_raw).__name__}, '
            f'shape={ssim_arr.shape}, mean={ssim_value:.4f}'
        )

        results[stem] = {
            'ssim': ssim_value,
            'n_slices': int(recon_vol.shape[0]),
            'shape': list(recon_vol.shape),
            'max_value': max_val,
        }
    return results


def summarize(results: dict) -> dict:
    """Estatisticas agregadas + classificacao do gate."""
    values = [r['ssim'] for r in results.values()]
    if not values:
        return {
            'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0,
            'status': 'FAILED', 'n_volumes': 0,
        }

    mean = float(np.mean(values))
    if mean >= PASS_THRESHOLD:
        status = 'PASSED'
    elif mean >= WARN_THRESHOLD:
        status = 'WARNING'
    else:
        status = 'FAILED'

    return {
        'mean': mean,
        'std': float(np.std(values)),
        'min': float(min(values)),
        'max': float(max(values)),
        'n_volumes': len(values),
        'status': status,
    }


def print_report(results: dict, summary: dict) -> None:
    sep = '=' * 72
    print(f'\n{sep}')
    print('SANITY CHECK - VarNet brain 4x')
    print(sep)

    print(f"\nVolumes processados: {summary['n_volumes']}")
    print('\nSSIM por volume:')
    for stem in sorted(results):
        r = results[stem]
        print(f"  {stem:50s} {r['ssim']:.4f}  ({r['n_slices']} slices, "
              f"shape {tuple(r['shape'])})")

    print('\nEstatisticas:')
    print(f"  Media: {summary['mean']:.4f}")
    print(f"  Desv:  {summary['std']:.4f}")
    print(f"  Min:   {summary['min']:.4f}")
    print(f"  Max:   {summary['max']:.4f}")
    print(f"  Esperado: ~0.966 (Sriram et al. 2020, leaderboard)")

    print(f"\n>>> STATUS: {summary['status']} <<<")
    if summary['status'] == 'PASSED':
        print('Pipeline validado. Pode seguir para o passo 5 '
              '(precompute_reconstructions em escala).')
    elif summary['status'] == 'WARNING':
        print(f'SSIM medio entre {WARN_THRESHOLD} e {PASS_THRESHOLD}. '
              'Investigar antes de seguir: transform, mascara, normalizacao.')
    else:
        print(f'SSIM medio abaixo de {WARN_THRESHOLD}. Algo esta seriamente '
              'errado. Suspeitos: prefix do state_dict, classe de mascara, '
              'ordem dos slices, crop incorreto.')
    print(sep)


def main(args):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
    )

    repo_root = Path(__file__).resolve().parent.parent
    checkpoint = args.checkpoint or (repo_root / DEFAULT_CHECKPOINT_RELPATH)
    data_path = args.data_path or config.anotados_dir()
    output_dir = args.output or (config.recons_dir() / 'sanity')
    cache_dir = output_dir / 'cache'
    device = resolve_device(args.device)

    print('Sanity check - configuracao:')
    print(f'  Checkpoint: {checkpoint}')
    print(f'  Data path:  {data_path}')
    print(f'  Output:     {output_dir}')
    print(f'  Cache:      {"ON em " + str(cache_dir) if args.use_cache else "OFF"}')
    print(f'  Device:     {device}')
    print(f'  Volumes:    {len(args.volumes)} selecionados')

    # 1. Tentar cache primeiro (se habilitado)
    cached_recons, cached_targets, cached_max_values = (
        _load_cached_volumes(cache_dir, args.volumes)
        if args.use_cache
        else (defaultdict(list), defaultdict(list), defaultdict(list))
    )
    volumes_missing = [
        v for v in args.volumes if v not in cached_recons
    ]

    # 2. Inferencia (apenas dos que faltam)
    if volumes_missing:
        print(f'\n[1/4] Carregando checkpoint...')
        model, sha256 = load_pretrained_varnet(checkpoint, device=device)
        print(f'      SHA-256: {sha256[:16]}...')

        print(f'\n[2/4] Construindo dataset ({len(volumes_missing)} volumes faltando)...')
        dataset = build_brain_kspace_dataset(
            data_path=data_path,
            volume_ids=volumes_missing,
        )
        print(f'      {len(dataset)} slices encontrados')

        print(f'\n[3/4] Rodando inferencia (slice-a-slice)...')
        t0 = time.time()
        new_recons, new_targets, new_max_values = run_inference(
            model, dataset, device, args.num_workers,
        )
        dt = time.time() - t0
        print(f'      Inferencia concluida em {dt:.1f}s')

        # Merge com cache
        for stem in new_recons:
            cached_recons[stem] = new_recons[stem]
            cached_targets[stem] = new_targets[stem]
            cached_max_values[stem] = new_max_values[stem]

        # Salvar cache (se habilitado)
        if args.use_cache:
            _save_cache(cache_dir, new_recons, new_targets, new_max_values)
    else:
        print('\n[1-3/4] Todos os volumes em cache, pulando inferencia.')
        sha256 = 'cached'  # nao recalcula sem carregar o modelo
        dt = 0.0

    # 4. SSIM
    print('\n[4/4] Computando SSIM por volume...')
    results = compute_ssim_per_volume(
        cached_recons, cached_targets, cached_max_values,
    )
    summary = summarize(results)

    print_report(results, summary)

    # Salvar JSON
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'checkpoint': str(checkpoint),
        'checkpoint_sha256': sha256,
        'device': str(device),
        'inference_seconds': dt,
        'volumes_requested': list(args.volumes),
        'volumes_processed': sorted(results.keys()),
        'pass_threshold': PASS_THRESHOLD,
        'warn_threshold': WARN_THRESHOLD,
        'per_volume': results,
        'summary': summary,
    }

    out_path = output_dir / 'sanity_report.json'
    out_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'\nReport: {out_path}')

    if summary['status'] == 'FAILED':
        sys.exit(1)
    elif summary['status'] == 'WARNING':
        sys.exit(2)


if __name__ == '__main__':
    main(parse_args())
