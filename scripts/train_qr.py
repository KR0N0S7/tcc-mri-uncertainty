# Autor: Massanori
# Data: 19/05/2026
# Descrição: Treino completo do Grupo B (QR) — 210000 iters. Replicacao do
#            paper base (Giannakopoulos et al., 2026, secao III) com
#            QuantileRegressionModule + qr_loss + alpha=0.10. Mesmos
#            hiperparametros de treino do Grupo A (controle experimental,
#            D4): AdamW, lr=3e-4, warmup=7500 iters, batch=1, seed=42.
#            A unica diferenca arquitetural entre os scripts e o modulo
#            e a loss; o loop de treino e identico (interface unificada D3).


"""Treino completo do Grupo B (QR) sobre as reconstrucoes do S4.

Replicacao de Giannakopoulos et al. (2026, secao III.D). Diferenca em
relacao ao Grupo A:
    - Modulo: QuantileRegressionModule (2 U-Nets, ~15.5M params, ~2x ResM)
    - Loss: qr_loss (pinball nos quantis alpha/2 e 1-alpha/2)
    - loss_kwargs={'alpha': 0.10}: cobertura nominal 90%

Mesmos hiperparametros de treino do Grupo A. Mesma seed (42) para isolar
o efeito da loss/modelo conforme D4.

Roda com:
    python scripts/train_qr.py --device cuda \\
        --recons-dir /kaggle/input/datasets/<user>/tcc-mri-recons-varnet-brain-4x \\
        --run-dir /kaggle/working/runs/group_b_full
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402
from src.data import ReconsSliceDataset  # noqa: E402
from src.losses import DEFAULT_ALPHA, qr_loss  # noqa: E402
from src.models import QuantileRegressionModule  # noqa: E402
from src.training import train  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
)
logger = logging.getLogger('train_qr')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Treino completo do Grupo B (QR) — 210k iters.'
    )
    parser.add_argument('--recons-dir', type=Path, default=None)
    parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--total-iters', type=int, default=None,
                        help='Default: config.scheduler.total_steps_full (210000).')
    parser.add_argument('--run-dir', type=Path,
                        default=ROOT / 'checkpoints' / 'group_b_full')
    parser.add_argument('--config', type=Path,
                        default=ROOT / 'configs' / 'training_base.json')
    parser.add_argument('--no-resume', action='store_true')
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--val-every', type=int, default=1000)
    parser.add_argument('--log-every', type=int, default=100)
    parser.add_argument('--checkpoint-every', type=int, default=10000)
    parser.add_argument('--alpha', type=float, default=DEFAULT_ALPHA)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    recons_root = (args.recons_dir or cfg.recons_dir()).expanduser().resolve()
    train_dir = recons_root / 'train'
    val_dir = recons_root / 'val'
    for d in (train_dir, val_dir):
        if not d.is_dir():
            logger.error(f'Diretorio ausente: {d}')
            return 2
    logger.info(f'recons_root = {recons_root}')
    logger.info(f'  train: {len(list(train_dir.glob("*.npz")))} .npz')
    logger.info(f'  val:   {len(list(val_dir.glob("*.npz")))} .npz')

    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.error('CUDA solicitado mas nao disponivel.')
        return 3
    if args.device == 'cuda':
        logger.info(f'GPU: {torch.cuda.get_device_name(0)}')
        logger.info(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

    config = json.loads(args.config.read_text(encoding='utf-8'))
    total_iters = args.total_iters or config['scheduler']['total_steps_full']
    logger.info(f'Config: {args.config.name}, alpha={args.alpha}, total_iters={total_iters}')

    train_ds = ReconsSliceDataset(train_dir, masks_dir=None)
    val_ds = ReconsSliceDataset(val_dir, masks_dir=None)
    train_loader = DataLoader(
        train_ds, batch_size=config['training']['batch_size'],
        shuffle=True, num_workers=args.num_workers,
        pin_memory=(args.device == 'cuda'),
    )
    val_loader = DataLoader(
        val_ds, batch_size=config['training']['batch_size'],
        shuffle=False, num_workers=args.num_workers,
        pin_memory=(args.device == 'cuda'),
    )
    logger.info(f'Dataset: {len(train_ds)} train, {len(val_ds)} val slices')

    module = QuantileRegressionModule(
        chans=config['model']['chans'],
        num_pool_layers=config['model']['num_pool_layers'],
    )
    n_params = sum(p.numel() for p in module.parameters())
    logger.info(f'QuantileRegressionModule: {n_params:,} parametros')

    result = train(
        module=module,
        loss_fn=qr_loss,
        train_loader=train_loader,
        val_loader=val_loader,
        total_iters=total_iters,
        config=config,
        run_dir=args.run_dir,
        device=args.device,
        seed=config['training']['seed'],
        val_every=args.val_every,
        log_every=args.log_every,
        checkpoint_every=args.checkpoint_every,
        resume=not args.no_resume,
        loss_kwargs={'alpha': args.alpha},
        config_snapshot={
            'group': 'B',
            'run': 'full',
            'recons_root': str(recons_root),
            'total_iters': total_iters,
            'alpha': args.alpha,
            'training_base': config,
        },
    )

    logger.info('=' * 60)
    logger.info('TREINO CONCLUIDO')
    for k, v in result.items():
        logger.info(f'  {k}: {v}')
    logger.info('=' * 60)
    logger.info(f'Checkpoints em: {args.run_dir.resolve()}')
    logger.info('  Proximo passo: rode scripts/eval_qr_pearson.py para sanity check')
    return 0


if __name__ == '__main__':
    sys.exit(main())
