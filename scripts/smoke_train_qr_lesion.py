# Autor: Massanori
# Data: 19/05/2026
# Descrição: Smoke test do Grupo C (QR-Lesion) para validar o pipeline antes
#            do treino completo. Recebe: --recons-dir (com train/val/...),
#            --masks-dir (com <volume_id>.pt flat), --device, --n-iters,
#            --lambda-lesion (default 5.0), --alpha (default 0.10).
#            Falha cedo se --masks-dir nao for fornecido: sem mascaras,
#            qr_lesion_loss colapsa em qr_loss (provado pelo teste
#            test_qr_lesion_loss_iguala_qr_loss_quando_mask_zerada).
#            Criterio: razao_final/inicial < 0.9. Roda em ~3-10 min no T4.


"""Smoke test do Grupo C (QR-Lesion) sobre as reconstrucoes do S4 + mascaras do S3.

Replica a estrutura do smoke_train_qr.py mas usa qr_lesion_loss com
lambda_lesion=5.0 e mascaras de lesao do S3. A arquitetura
(QuantileRegressionLesionModule) e identica a do Grupo B (alias) —
so muda a loss e os dados ponderados. Controle experimental D4.

Gate de sanidade antes do treino completo. Falha defensivamente se
--masks-dir nao for fornecido (sem mascaras, C colapsa em B silenciosamente).

Roda com:
    python scripts/smoke_train_qr_lesion.py --device cuda \\
        --recons-dir /kaggle/input/datasets/<user>/tcc-mri-recons-varnet-brain-4x \\
        --masks-dir /kaggle/input/datasets/<user>/tcc-mri-lesion-masks \\
        --run-dir /kaggle/working/runs/group_c_smoke
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
logger = logging.getLogger('smoke_train_qr_lesion')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Smoke test do Grupo C (QR-Lesion) — 500 iters.'
    )
    parser.add_argument('--recons-dir', type=Path, default=None)
    parser.add_argument(
        '--masks-dir', type=Path, required=True,
        help='OBRIGATORIO. Diretorio flat com <volume_id>.pt das mascaras (S3). '
             'Sem isso, qr_lesion_loss colapsa em qr_loss e C degrada em B.',
    )
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--n-iters', type=int, default=500)
    parser.add_argument('--run-dir', type=Path,
                        default=ROOT / 'checkpoints' / 'smoke_qr_lesion')
    parser.add_argument('--keep-run', action='store_true')
    parser.add_argument('--config', type=Path,
                        default=ROOT / 'configs' / 'training_base.json')
    parser.add_argument('--alpha', type=float, default=DEFAULT_ALPHA)
    parser.add_argument(
        '--lambda-lesion', type=float, default=5.0,
        help='Fator de ponderacao de regioes de lesao. MVP usa 5.0.',
    )
    return parser.parse_args()


def resolve_device(choice: str) -> str:
    if choice == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    if choice == 'cuda' and not torch.cuda.is_available():
        logger.warning('CUDA solicitado mas nao disponivel; CPU.')
        return 'cpu'
    return choice


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
            f'masks_dir ({masks_root}) tem apenas {n_masks} arquivos .pt '
            f'(esperado ~352). Suba o dataset de mascaras do S3 como Kaggle '
            f'dataset com slug "tcc-mri-lesion-masks".'
        )
        return 2
    logger.info(f'masks_dir = {masks_root} ({n_masks} .pt files)')

    if args.run_dir.exists() and not args.keep_run:
        logger.info(f'Removendo {args.run_dir} (use --keep-run para preservar)')
        shutil.rmtree(args.run_dir)
    args.run_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    logger.info(f'Device: {device}')

    config = json.loads(args.config.read_text(encoding='utf-8'))
    logger.info(
        f'Config carregado: {args.config.name}, alpha={args.alpha}, '
        f'lambda_lesion={args.lambda_lesion}'
    )

    train_ds = ReconsSliceDataset(train_dir, masks_dir=masks_root)
    val_ds = ReconsSliceDataset(val_dir, masks_dir=masks_root)
    train_loader = DataLoader(
        train_ds, batch_size=config['training']['batch_size'],
        shuffle=True, num_workers=0,
        pin_memory=(device == 'cuda'),
    )
    val_loader = DataLoader(
        val_ds, batch_size=config['training']['batch_size'],
        shuffle=False, num_workers=0,
        pin_memory=(device == 'cuda'),
    )
    logger.info(f'Dataset: train={len(train_ds)} slices, val={len(val_ds)} slices')

    module = QuantileRegressionLesionModule(
        chans=config['model']['chans'],
        num_pool_layers=config['model']['num_pool_layers'],
    )
    n_params = sum(p.numel() for p in module.parameters())
    logger.info(
        f'QuantileRegressionLesionModule: {n_params:,} parametros '
        f'(arquitetura identica ao Grupo B, alias)'
    )

    n_iters = args.n_iters
    log_every = max(10, n_iters // 50)
    val_every = max(50, n_iters // 5)
    ckpt_every = max(100, n_iters // 2)

    result = train(
        module=module,
        loss_fn=qr_lesion_loss,
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
        loss_kwargs={'alpha': args.alpha, 'lambda_lesion': args.lambda_lesion},
        config_snapshot={
            'group': 'C',
            'run': 'smoke_qr_lesion',
            'recons_root': str(recons_root),
            'masks_root': str(masks_root),
            'n_iters': n_iters,
            'alpha': args.alpha,
            'lambda_lesion': args.lambda_lesion,
            'training_base': config,
        },
    )

    csv_path = args.run_dir / 'metrics.csv'
    if not csv_path.is_file():
        logger.error('metrics.csv ausente.')
        return 3

    train_losses = []
    with csv_path.open(encoding='utf-8') as f:
        header = f.readline().strip().split(',')
        idx_tl = header.index('train_loss')
        for line in f:
            parts = line.strip().split(',')
            if len(parts) > idx_tl and parts[idx_tl]:
                train_losses.append(float(parts[idx_tl]))

    if len(train_losses) < 4:
        logger.error(f'Apenas {len(train_losses)} pontos de loss.')
        return 3

    k = max(2, len(train_losses) // 5)
    initial_avg = sum(train_losses[:k]) / k
    final_avg = sum(train_losses[-k:]) / k
    ratio = final_avg / initial_avg if initial_avg > 0 else float('inf')

    logger.info('=' * 60)
    logger.info(f'Resultado do smoke test (Grupo C, lambda={args.lambda_lesion}):')
    logger.info(f'  Loss inicial (primeiros {k}): {initial_avg:.6f}')
    logger.info(f'  Loss final   (ultimos  {k}): {final_avg:.6f}')
    logger.info(f'  Razao final/inicial:         {ratio:.3f}')
    logger.info(f'  Best val loss:               {result["best_val_loss"]:.6f}')
    logger.info(f'  Elapsed:                     {result["elapsed_seconds"]:.1f} s')
    logger.info('=' * 60)

    if ratio < 0.9:
        logger.info('SMOKE TEST PASSOU: loss diminuiu como esperado.')
        return 0
    logger.error(
        f'SMOKE TEST FALHOU: razao = {ratio:.3f} >= 0.9. '
        f'Investigue antes do treino completo.'
    )
    return 1


if __name__ == '__main__':
    sys.exit(main())
