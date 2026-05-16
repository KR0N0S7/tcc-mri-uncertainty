# Autor: Massanori
# Data: 16/05/2026
# Descrição: Pipeline batch de pré-computação das reconstruções E2E-VarNet
#            para todos os volumes dos splits estratificados do S3 (213
#            train + 46 val + 46 cal + 47 test = 352 volumes). Recebe:
#            split a processar (train/val/cal/test/all), checkpoint, data
#            path, output dir, device. Retorna: nada — para cada volume
#            salva data/recons/{split}/{stem}.npz contendo {recon, target,
#            error_map, max_val, volume_id, split, acceleration,
#            center_fraction, varnet_sha256}. Idempotente e resumível:
#            pula volumes com .npz existente, escreve atomicamente via
#            rename de .npz.tmp. Modo smoke test via --max-volumes N para
#            validar setup antes de rodar 352 volumes. Após cada split
#            adiciona entrada em data/recons/precompute_manifest.json
#            para rastreabilidade de runs. Usa:
#            python scripts/precompute_reconstructions.py --split val --device cuda
#            ou no Kaggle: ver notebooks/kaggle_precompute.ipynb


"""Pre-computacao em batch das reconstrucoes E2E-VarNet brain 4x.

Estrategia:
    1. Carrega checkpoint UMA vez (sha256 entra como metadata em todos os .npz)
    2. Para cada split solicitado:
       a. Carrega volume_ids do splits/{split}.txt
       b. Filtra os ja salvos (resumibilidade: skip se {stem}.npz existe)
       c. Aplica --max-volumes se passado (smoke test)
       d. Constroi UM SliceDataset filtrado para os volumes restantes
       e. Itera slices via DataLoader, acumulando por volume
       f. Na transicao de volume (fname mudou), salva o anterior e libera RAM
       g. No fim do split, salva o ultimo volume
    3. Salva manifest agregado para rastreabilidade

Atomicidade por volume:
    Cada .npz e escrito em <stem>.npz.tmp e renomeado. Crash no meio de um
    volume nao corrompe o disco. Volumes anteriores ficam intactos.

Reprodutibilidade da mascara:
    VarNetDataTransform com use_seed=True deriva seed de fname.name (so
    arquivo, sem path). Mesma mascara para o mesmo volume em qualquer
    maquina, qualquer ordem de processamento.

Schema do .npz:
    recon:           (S, H, W) float32 - reconstrucao VarNet
    target:          (S, H, W) float32 - RSS da k-space fully-sampled
    error_map:       (S, H, W) float32 - |target - recon|
    max_val:         scalar float32 - max(target_vol), p/ normalizacao S5
    volume_id:       string - stem do .h5
    split:           string - 'train' | 'val' | 'cal' | 'test'
    acceleration:    int32 - 4
    center_fraction: float32 - 0.08
    varnet_sha256:   string - hash do checkpoint usado

Para carregar:
    >>> data = np.load('file_brain_AXFLAIR_200_6002460.npz')
    >>> recon = data['recon']                 # (S, H, W) array
    >>> volume_id = str(data['volume_id'])    # converte de 0-d array p/ str
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from fastmri.data import transforms as T

# Adiciona a raiz do repo ao sys.path para importar src.* quando o script
# e chamado diretamente.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.data.kspace_dataset import (  # noqa: E402
    DEFAULT_ACCELERATION,
    DEFAULT_CENTER_FRACTION,
    VALID_SPLIT_NAMES,
    build_brain_kspace_dataset,
    load_split,
)
from src.models import load_pretrained_varnet  # noqa: E402


DEFAULT_CHECKPOINT_RELPATH = 'checkpoints/brain_leaderboard_state_dict.pt'
MANIFEST_FILENAME = 'precompute_manifest.json'

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description='Pre-computacao em batch das reconstrucoes VarNet brain 4x.',
    )
    parser.add_argument(
        '--split',
        choices=list(VALID_SPLIT_NAMES) + ['all'],
        default='all',
        help='Split(s) a processar. Default: all.',
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
        help='Diretorio para os .npz. Default: config.recons_dir().',
    )
    parser.add_argument(
        '--splits-dir',
        type=Path,
        default=None,
        help='Diretorio dos {train,val,cal,test}.txt. '
             'Default: config.splits_dir().',
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
        help='Workers do DataLoader. CPU local: 0. Kaggle T4: 2-4 ajuda.',
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Re-processa volumes que ja tem .npz salvo (default: pula).',
    )
    parser.add_argument(
        '--max-volumes',
        type=int,
        default=None,
        help='Smoke test: processa apenas os N primeiros volumes por split.',
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def resolve_device(arg: Optional[str]) -> torch.device:
    if arg:
        return torch.device(arg)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _to_scalar(x):
    if hasattr(x, 'item'):
        return x.item()
    if isinstance(x, (list, tuple)):
        return _to_scalar(x[0])
    return x


def _extract_fname(batch_fname) -> str:
    if isinstance(batch_fname, (list, tuple)):
        return str(batch_fname[0])
    return str(batch_fname)


def save_volume(
    split_out_dir: Path,
    stem: str,
    slice_records,
    max_vals,
    sha256: str,
    split: str,
) -> Path:
    """Empilha slices, calcula error_map, salva .npz atomicamente.

    slice_records: list of (slice_num, recon_2d, target_2d)
    """
    sorted_records = sorted(slice_records, key=lambda x: x[0])
    recon_vol = np.stack([r for _, r, _ in sorted_records]).astype(np.float32)
    target_vol = np.stack([t for _, _, t in sorted_records]).astype(np.float32)
    error_map = np.abs(target_vol - recon_vol).astype(np.float32)
    max_val = float(max(max_vals))

    out_path = split_out_dir / f'{stem}.npz'
    # tmp_path PRECISA terminar em .npz porque np.savez_compressed adiciona
    # ".npz" automaticamente se a extensao nao for essa. Usamos ".tmp.npz"
    # para diferenciar do arquivo final mas manter a extensao esperada.
    tmp_path = split_out_dir / f'{stem}.tmp.npz'

    np.savez_compressed(
        tmp_path,
        recon=recon_vol,
        target=target_vol,
        error_map=error_map,
        max_val=np.float32(max_val),
        volume_id=np.array(stem),
        split=np.array(split),
        acceleration=np.int32(DEFAULT_ACCELERATION),
        center_fraction=np.float32(DEFAULT_CENTER_FRACTION),
        varnet_sha256=np.array(sha256),
    )

    # Verifica integridade antes do rename (catch partial writes).
    try:
        d = np.load(tmp_path)
        if d['recon'].shape != recon_vol.shape:
            raise ValueError(
                f'Shape mismatch apos save: {d["recon"].shape} != {recon_vol.shape}'
            )
        d.close()
    except Exception as e:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f'Verificacao de {tmp_path} falhou: {e}')

    tmp_path.replace(out_path)
    return out_path


# ---------------------------------------------------------------------------
# Processamento por split
# ---------------------------------------------------------------------------


def process_split(
    model: torch.nn.Module,
    sha256: str,
    split: str,
    data_path: Path,
    splits_dir: Path,
    output_dir: Path,
    device: torch.device,
    num_workers: int,
    overwrite: bool,
    max_volumes: Optional[int],
) -> dict:
    """Processa todos os volumes de 1 split (com resumibilidade)."""
    split_out_dir = output_dir / split
    split_out_dir.mkdir(parents=True, exist_ok=True)

    all_volumes = sorted(load_split(split, splits_dir))

    # Resumibilidade
    if overwrite:
        todo = all_volumes
        n_already_done = 0
    else:
        todo = [v for v in all_volumes if not (split_out_dir / f'{v}.npz').exists()]
        n_already_done = len(all_volumes) - len(todo)

    # Smoke test
    if max_volumes is not None:
        todo = todo[:max_volumes]

    print(f'\n=== Split: {split} ===')
    print(f'  Total no split:       {len(all_volumes)}')
    print(f'  Ja processados:       {n_already_done}')
    print(f'  A processar:          {len(todo)}')

    if not todo:
        print(f'  Nada a fazer.')
        return {
            'split': split,
            'processed': 0,
            'skipped': n_already_done,
            'errors': [],
            'seconds': 0.0,
        }

    # Constroi dataset filtrado pelos volumes restantes.
    dataset = build_brain_kspace_dataset(
        data_path=data_path,
        volume_ids=todo,
    )
    print(f'  Slices a processar:   {len(dataset)}')

    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=num_workers,
        shuffle=False,
    )

    # Acumulacao por volume (libera RAM em cada transicao)
    current_stem: Optional[str] = None
    current_records = []   # [(slice_num, recon_2d, target_2d), ...]
    current_max_vals = []

    processed = 0
    errors = []
    t0 = time.time()
    pbar = tqdm(total=len(dataset), desc=f'{split}', unit='slice')

    with torch.no_grad():
        for batch in loader:
            fname = _extract_fname(batch.fname)
            stem = Path(fname).stem

            # Transicao: salva o volume anterior e libera RAM
            if current_stem is not None and stem != current_stem:
                try:
                    save_volume(
                        split_out_dir, current_stem,
                        current_records, current_max_vals,
                        sha256, split,
                    )
                    processed += 1
                    pbar.set_postfix({'done': processed, 'last': current_stem[-12:]})
                except Exception as e:
                    logger.error(f'Falha ao salvar {current_stem}: {e}')
                    errors.append({'volume': current_stem, 'error': str(e)})
                current_records = []
                current_max_vals = []

            current_stem = stem

            # Inferencia de 1 slice
            masked_kspace = batch.masked_kspace.to(device)
            mask = batch.mask.to(device)
            slice_num = int(_to_scalar(batch.slice_num))
            max_val = float(_to_scalar(batch.max_value))
            crop_size = (
                int(_to_scalar(batch.crop_size[0])),
                int(_to_scalar(batch.crop_size[1])),
            )

            output = model(masked_kspace, mask)
            output = T.center_crop(output, crop_size)
            recon_2d = output.cpu().squeeze(0).numpy()

            target_t = batch.target
            target_2d = (
                target_t.squeeze(0).numpy()
                if torch.is_tensor(target_t)
                else np.asarray(target_t)
            )

            current_records.append((slice_num, recon_2d, target_2d))
            current_max_vals.append(max_val)
            pbar.update(1)

    pbar.close()

    # Salva o ultimo volume (nao houve transicao para dispara-lo)
    if current_stem is not None and current_records:
        try:
            save_volume(
                split_out_dir, current_stem,
                current_records, current_max_vals,
                sha256, split,
            )
            processed += 1
        except Exception as e:
            logger.error(f'Falha ao salvar {current_stem}: {e}')
            errors.append({'volume': current_stem, 'error': str(e)})

    dt = time.time() - t0
    print(f'  {split}: {processed} volumes em {dt:.1f}s, {len(errors)} erros')
    return {
        'split': split,
        'processed': processed,
        'skipped': n_already_done,
        'errors': errors,
        'seconds': dt,
    }


# ---------------------------------------------------------------------------
# Manifest agregado
# ---------------------------------------------------------------------------


def append_manifest(output_dir: Path, entry: dict) -> Path:
    """Anexa entrada ao precompute_manifest.json (lista cumulativa de runs)."""
    manifest_path = output_dir / MANIFEST_FILENAME
    if manifest_path.exists():
        try:
            content = json.loads(manifest_path.read_text(encoding='utf-8'))
            history = content if isinstance(content, list) else [content]
        except Exception:
            history = []
    else:
        history = []
    history.append(entry)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(history, indent=2), encoding='utf-8')
    return manifest_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
    )

    repo_root = Path(__file__).resolve().parent.parent
    checkpoint = args.checkpoint or (repo_root / DEFAULT_CHECKPOINT_RELPATH)
    data_path = args.data_path or config.anotados_dir()
    output_dir = args.output or config.recons_dir()
    splits_dir = args.splits_dir or config.splits_dir()
    device = resolve_device(args.device)

    splits = list(VALID_SPLIT_NAMES) if args.split == 'all' else [args.split]

    print('Pre-computacao de reconstrucoes E2E-VarNet brain 4x')
    print(f'  Checkpoint:  {checkpoint}')
    print(f'  Data:        {data_path}')
    print(f'  Output:      {output_dir}')
    print(f'  Splits dir:  {splits_dir}')
    print(f'  Device:      {device}')
    print(f'  Workers:     {args.num_workers}')
    print(f'  Splits:      {splits}')
    print(f'  Overwrite:   {args.overwrite}')
    if args.max_volumes:
        print(f'  Smoke test:  max {args.max_volumes} volumes por split')

    print('\nCarregando checkpoint...')
    model, sha256 = load_pretrained_varnet(checkpoint, device=device)
    print(f'  SHA-256: {sha256[:16]}...')

    t_start = time.time()
    reports = []
    for split in splits:
        report = process_split(
            model=model,
            sha256=sha256,
            split=split,
            data_path=data_path,
            splits_dir=splits_dir,
            output_dir=output_dir,
            device=device,
            num_workers=args.num_workers,
            overwrite=args.overwrite,
            max_volumes=args.max_volumes,
        )
        reports.append(report)

    dt = time.time() - t_start

    manifest_entry = {
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'checkpoint': str(checkpoint),
        'checkpoint_sha256': sha256,
        'device': str(device),
        'splits_requested': splits,
        'overwrite': args.overwrite,
        'max_volumes': args.max_volumes,
        'total_seconds': dt,
        'reports': reports,
    }
    manifest_path = append_manifest(output_dir, manifest_entry)

    total_processed = sum(r['processed'] for r in reports)
    total_errors = sum(len(r['errors']) for r in reports)
    total_skipped = sum(r['skipped'] for r in reports)

    sep = '=' * 72
    print(f'\n{sep}')
    print(f'CONCLUIDO em {dt:.1f}s')
    print(sep)
    print(f'Total processado:  {total_processed}')
    print(f'Total pulado:      {total_skipped} (ja existiam)')
    print(f'Total erros:       {total_errors}')
    print(f'Manifest:          {manifest_path}')

    if total_errors > 0:
        print(f'\nVolumes com erro:')
        for r in reports:
            for e in r['errors']:
                print(f"  {r['split']}/{e['volume']}: {e['error']}")
        sys.exit(1)


if __name__ == '__main__':
    main(parse_args())
