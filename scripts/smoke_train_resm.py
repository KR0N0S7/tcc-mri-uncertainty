# Autor: Massanori
# Data: 17/05/2026
# Descrição: Smoke test do Grupo A (ResM) para validar o pipeline de treino
#            do S5.1b antes de gastar GPU em 210000 iters. Recebe via CLI:
#            --recons-dir (default: src.config.recons_dir() / 'train'),
#            --device (auto-detect), --n-iters (default 500). Roda ~500 iters
#            de AdamW no Grupo A sobre os volumes em recons_dir/train e
#            valida em recons_dir/val. Saida: checkpoints/smoke_resm/ com
#            metrics.csv, last.pt e tb/. Criterio de sucesso: loss media
#            nas ultimas 50 iters < loss media nas primeiras 50 iters
#            (multiplicada por 0.9 para tolerar ruido). Roda em ~10 min
#            no Kaggle T4 ou ~60 min em CPU. NAO substitui o treino
#            completo — e gate de sanidade.


"""Smoke test do Grupo A (ResM) para validar o pipeline de treino.

Gate de sanidade obrigatorio antes do treino completo: se loss nao
diminui em 500 iters, ha bug — e melhor descobrir agora do que
apos 12h de GPU desperdicadas.

Roda com:
    # Local (CPU ou GPU local)
    python scripts/smoke_train_resm.py

    # Local com override
    python scripts/smoke_train_resm.py --device cuda --n-iters 1000

    # Kaggle (dataset montado em /kaggle/input/...)
    python scripts/smoke_train_resm.py \\
        --recons-dir /kaggle/input/tcc-mri-recons-varnet-brain-4x \\
        --device cuda
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Adiciona raiz do projeto ao sys.path para permitir execucao via
# `python scripts/smoke_train_resm.py` sem instalar como pacote.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402
from src.data import ReconsSliceDataset  # noqa: E402
from src.losses import resm_loss  # noqa: E402
from src.models import ResidualMagnitudeModule  # noqa: E402
from src.training import train  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
)
logger = logging.getLogger('smoke_train_resm')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Smoke test do Grupo A (ResM) — 500 iters de sanidade.'
    )
    parser.add_argument(
        '--recons-dir', type=Path, default=None,
        help='Diretorio raiz com subdirs train/ e val/ contendo .npz do S4. '
             'Default: src.config.recons_dir().',
    )
    parser.add_argument(
        '--device', default='auto',
        choices=['auto', 'cpu', 'cuda'],
        help='Device. auto = cuda se disponivel, senao cpu.',
    )
    parser.add_argument(
        '--n-iters', type=int, default=500,
        help='Numero de iteracoes do smoke (default 500).',
    )
    parser.add_argument(
        '--run-dir', type=Path, default=ROOT / 'checkpoints' / 'smoke_resm',
        help='Diretorio de saida.',
    )
    parser.add_argument(
        '--keep-run', action='store_true',
        help='Se passado, NAO apaga run-dir antes de comecar (permite resume).',
    )
    parser.add_argument(
        '--config', type=Path, default=ROOT / 'configs' / 'training_base.json',
        help='Path do training_base.json.',
    )
    return parser.parse_args()


def resolve_device(choice: str) -> str:
    if choice == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    if choice == 'cuda' and not torch.cuda.is_available():
        logger.warning('CUDA solicitado mas nao disponivel; caindo para CPU.')
        return 'cpu'
    return choice


def main() -> int:
    args = parse_args()

    # 1. Resolver paths
    recons_root = (args.recons_dir or cfg.recons_dir()).expanduser().resolve()
    train_dir = recons_root / 'train'
    val_dir = recons_root / 'val'
    for d in (train_dir, val_dir):
        if not d.is_dir():
            logger.error(f'Diretorio ausente: {d}')
            logger.error('Verifique TCC_RECONS_DIR ou --recons-dir.')
            return 2

    # 2. Limpar run_dir (a menos que --keep-run)
    if args.run_dir.exists() and not args.keep_run:
        logger.info(f'Removendo {args.run_dir} (use --keep-run para preservar)')
        shutil.rmtree(args.run_dir)
    args.run_dir.mkdir(parents=True, exist_ok=True)

    # 3. Resolver device
    device = resolve_device(args.device)
    logger.info(f'Device: {device}')

    # 4. Carregar config
    config = json.loads(args.config.read_text(encoding='utf-8'))
    logger.info(f'Config carregado: {args.config.name}')

    # 5. Datasets e DataLoaders
    # NAO precisamos de masks_dir para o Grupo A (interface unificada D3
    # retorna mascara de zeros, que e ignorada por resm_loss).
    train_ds = ReconsSliceDataset(train_dir, masks_dir=None)
    val_ds = ReconsSliceDataset(val_dir, masks_dir=None)

    train_loader = DataLoader(
        train_ds,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=0,  # 0 para Windows e debug; aumentar em Kaggle
        pin_memory=(device == 'cuda'),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=(device == 'cuda'),
    )
    logger.info(
        f'Dataset: train={len(train_ds)} slices, val={len(val_ds)} slices'
    )

    # 6. Modulo (Grupo A)
    module = ResidualMagnitudeModule(
        chans=config['model']['chans'],
        num_pool_layers=config['model']['num_pool_layers'],
    )
    n_params = sum(p.numel() for p in module.parameters())
    logger.info(f'ResidualMagnitudeModule: {n_params:,} parametros')

    # 7. Frequencias para smoke (override do config)
    n_iters = args.n_iters
    log_every = max(10, n_iters // 50)
    val_every = max(50, n_iters // 5)
    ckpt_every = max(100, n_iters // 2)

    # 8. Rodar
    result = train(
        module=module,
        loss_fn=resm_loss,
        train_loader=train_loader,
        val_loader=val_loader,
        total_iters=n_iters,
        config=config,
        run_dir=args.run_dir,
        device=device,
        seed=config['training']['seed'],
        val_every=val_every,
        log_every=log_every,
        checkpoint_every=ckpt_every,
        resume=args.keep_run,
        loss_kwargs=None,  # resm_loss nao precisa de kwargs
        config_snapshot={
            'group': 'A',
            'run': 'smoke_resm',
            'recons_root': str(recons_root),
            'n_iters': n_iters,
            'training_base': config,
        },
    )

    # 9. Criterio de sanidade: loss diminuiu?
    csv_path = args.run_dir / 'metrics.csv'
    if not csv_path.is_file():
        logger.error('metrics.csv ausente — train() falhou silenciosamente?')
        return 3

    # Le manualmente para nao depended de pandas
    train_losses = []
    with csv_path.open(encoding='utf-8') as f:
        header = f.readline().strip().split(',')
        idx_tl = header.index('train_loss')
        for line in f:
            parts = line.strip().split(',')
            if len(parts) > idx_tl and parts[idx_tl]:
                train_losses.append(float(parts[idx_tl]))

    if len(train_losses) < 4:
        logger.error(
            f'Apenas {len(train_losses)} pontos de loss — log_every alto demais?'
        )
        return 3

    k = max(2, len(train_losses) // 5)
    initial_avg = sum(train_losses[:k]) / k
    final_avg = sum(train_losses[-k:]) / k
    ratio = final_avg / initial_avg if initial_avg > 0 else float('inf')

    logger.info('=' * 60)
    logger.info(f'Resultado do smoke test:')
    logger.info(f'  Loss inicial (primeiros {k}): {initial_avg:.6f}')
    logger.info(f'  Loss final   (ultimos  {k}): {final_avg:.6f}')
    logger.info(f'  Razao final/inicial:         {ratio:.3f}')
    logger.info(f'  Best val loss:               {result["best_val_loss"]:.6f}')
    logger.info(f'  Elapsed:                     {result["elapsed_seconds"]:.1f} s')
    logger.info('=' * 60)

    # Criterio: razao < 0.9 (loss caiu pelo menos 10%). Margem para ruido.
    if ratio < 0.9:
        logger.info('SMOKE TEST PASSOU: loss diminuiu como esperado.')
        return 0
    else:
        logger.error(
            f'SMOKE TEST FALHOU: razao final/inicial = {ratio:.3f} >= 0.9. '
            f'Investigue antes de gastar GPU no treino completo.'
        )
        return 1


if __name__ == '__main__':
    sys.exit(main())
