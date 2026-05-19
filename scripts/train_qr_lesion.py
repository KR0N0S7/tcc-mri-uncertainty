# Autor: Massanori
# Data: 19/05/2026
# Descrição: Treino completo do Grupo C (QR-Lesion) — 210000 iters. Contribuicao
#            original do TCC: pinball loss ponderada por lambda nas regioes
#            de lesao (Yeung et al., 2022, generalizado para CQR). Mesmos
#            hiperparametros de treino do Grupo A e B (controle experimental
#            D4): AdamW, lr=3e-4, warmup=7500 iters, batch=1, seed=42.
#            A unica variavel independente entre B e C e a loss aplicada;
#            a arquitetura (QuantileRegressionLesionModule) e literalmente
#            a mesma classe do Grupo B (alias). lambda_lesion=5.0 e a
#            configuracao MVP defensavel. Falha cedo se --masks-dir ausente.


"""Treino completo do Grupo C (QR-Lesion) — contribuicao original do TCC.

Replica os hiperparametros do Grupo B (Giannakopoulos et al., 2026, secao III.D)
com UMA diferenca:
    - Loss: qr_loss → qr_lesion_loss com lambda_lesion=5.0
    - Mesma arquitetura (alias de QuantileRegressionModule)
    - Mesma seed (42), mesmos hiperparametros

A hipotese e que a ponderacao por lesao melhora a calibracao da
incerteza dentro das regioes de interesse clinico, sem prejudicar
cobertura global. Validacao: S5.7 (Coverage_lesion, IoU_uncertainty,
ULAS), S5.8 (analise estatistica vs A e B).

Roda com:
    python scripts/train_qr_lesion.py --device cuda \\
        --recons-dir /kaggle/input/datasets/<user>/tcc-mri-recons-varnet-brain-4x \\
        --masks-dir /kaggle/input/datasets/<user>/tcc-mri-lesion-masks \\
        --run-dir /kaggle/working/runs/group_c_full

Refs:
    Giannakopoulos, C. et al. (2026). arXiv:2601.13236.
    Yeung, M. et al. (2022). Comput. Med. Imaging Graph. 95:102026.
    Isensee, F. et al. (2021). Nat. Methods 18(2):203-211. (nnU-Net)
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
from src.losses import DEFAULT_ALPHA, qr_lesion_loss  # noqa: E402
from src.models import QuantileRegressionLesionModule  # noqa: E402
from src.training import train  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
)
logger = logging.getLogger('train_qr_lesion')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Treino completo do Grupo C (QR-Lesion) — 210k iters.'
    )
    parser.add_argument('--recons-dir', type=Path, default=None)
    parser.add_argument(
        '--masks-dir', type=Path, required=True,
        help='OBRIGATORIO. Diretorio flat com <volume_id>.pt das mascaras (S3).',
    )
    parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--total-iters', type=int, default=None)
    parser.add_argument('--run-dir', type=Path,
                        default=ROOT / 'checkpoints' / 'group_c_full')
    parser.add_argument('--config', type=Path,
                        default=ROOT / 'configs' / 'training_base.json')
    parser.add_argument('--no-resume', action='store_true')
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--val-every', type=int, default=1000)
    parser.add_argument('--log-every', type=int, default=100)
    parser.add_argument('--checkpoint-every', type=int, default=10000)
    parser.add_argument('--alpha', type=float, default=DEFAULT_ALPHA)
    parser.add_argument(
        '--lambda-lesion', type=float, default=5.0,
        help='Fator de ponderacao de regioes de lesao. MVP defensavel = 5.0.',
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    recons_root = (args.recons_dir or cfg.recons_dir()).expanduser().resolve()
    masks_root = args.masks_dir.expanduser().resolve()
    train_dir = recons_root / 'train'
    val_dir = recons_root / 'val'
    for d in (train_dir, val_dir, masks_root):
        if not d.is_dir():
            logger.error(f'Diretorio ausente: {d}')
            return 2

    n_masks = len(list(masks_root.glob('*.pt')))
    if n_masks < 100:
        logger.error(
            f'masks_dir ({masks_root}) tem {n_masks} arquivos .pt; '
            f'esperado ~352. Falha defensiva para evitar C colapsar em B.'
        )
        return 2
    logger.info(f'masks_dir = {masks_root} ({n_masks} .pt files)')

    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.error('CUDA solicitado mas nao disponivel.')
        return 3
    if args.device == 'cuda':
        logger.info(f'GPU: {torch.cuda.get_device_name(0)}')
        logger.info(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

    config = json.loads(args.config.read_text(encoding='utf-8'))
    total_iters = args.total_iters or config['scheduler']['total_steps_full']
    logger.info(
        f'Config: {args.config.name}, alpha={args.alpha}, '
        f'lambda_lesion={args.lambda_lesion}, total_iters={total_iters}'
    )

    train_ds = ReconsSliceDataset(train_dir, masks_dir=masks_root)
    val_ds = ReconsSliceDataset(val_dir, masks_dir=masks_root)
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

    module = QuantileRegressionLesionModule(
        chans=config['model']['chans'],
        num_pool_layers=config['model']['num_pool_layers'],
    )
    n_params = sum(p.numel() for p in module.parameters())
    logger.info(f'QuantileRegressionLesionModule: {n_params:,} parametros')

    result = train(
        module=module,
        loss_fn=qr_lesion_loss,
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
        loss_kwargs={'alpha': args.alpha, 'lambda_lesion': args.lambda_lesion},
        config_snapshot={
            'group': 'C',
            'run': 'full',
            'recons_root': str(recons_root),
            'masks_root': str(masks_root),
            'total_iters': total_iters,
            'alpha': args.alpha,
            'lambda_lesion': args.lambda_lesion,
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
