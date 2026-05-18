# Autor: Massanori
# Data: 17/05/2026
# Descrição: Treino completo do Grupo A (ResM) — 210000 iters, hiperparametros
#            replicando Giannakopoulos et al. (2026). Recebe: --recons-dir
#            (default cfg.recons_dir()), --device (cuda recomendado),
#            --run-dir (checkpoints/group_a_full), --total-iters (default vem
#            do config.scheduler.total_steps_full). Saidas: run_dir/last.pt,
#            run_dir/best.pt, run_dir/metrics.csv, run_dir/tb/. Resume
#            automatico em run_dir/last.pt se existir (--no-resume desativa).
#            Diferente do smoke, NAO aplica criterio razao_final/inicial —
#            esse e o treino real, criterio de sucesso e val_loss
#            comparavel ao reportado no paper.


"""Treino completo do Grupo A (ResM) sobre as reconstrucoes do S4.

Hiperparametros default do configs/training_base.json:
    optimizer: AdamW, lr=3e-4, weight_decay=1e-4, betas=(0.9, 0.999), eps=1e-8
    scheduler: warmup linear de 7500 iters, depois constante
    batch: 1, seed: 42

Roda com:
    # Kaggle (acesso direto ao dataset montado)
    python scripts/train_resm.py --device cuda \\
        --recons-dir /kaggle/input/tcc-mri-recons-varnet-brain-4x \\
        --run-dir /kaggle/working/runs/group_a_full

    # Local (com TCC_RECONS_DIR no .env)
    python scripts/train_resm.py --device cuda
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Adiciona raiz do projeto ao sys.path
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
logger = logging.getLogger('train_resm')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Treino completo do Grupo A (ResM) — 210k iters.'
    )
    parser.add_argument(
        '--recons-dir', type=Path, default=None,
        help='Diretorio raiz com subdirs train/ e val/ contendo .npz do S4. '
             'Default: src.config.recons_dir() (TCC_RECONS_DIR).',
    )
    parser.add_argument(
        '--device', default='cuda', choices=['cpu', 'cuda'],
        help='Device. cuda recomendado (12h em T4); cpu apenas para debug.',
    )
    parser.add_argument(
        '--total-iters', type=int, default=None,
        help='Total de iters. Default: config.scheduler.total_steps_full (210000).',
    )
    parser.add_argument(
        '--run-dir', type=Path,
        default=ROOT / 'checkpoints' / 'group_a_full',
        help='Diretorio de saida. Resume automatico se last.pt existir.',
    )
    parser.add_argument(
        '--config', type=Path,
        default=ROOT / 'configs' / 'training_base.json',
        help='Path do training_base.json.',
    )
    parser.add_argument(
        '--no-resume', action='store_true',
        help='Desativa resume automatico. Use para comecar do zero.',
    )
    parser.add_argument(
        '--num-workers', type=int, default=2,
        help='Workers do DataLoader (default 2; em Windows use 0).',
    )
    parser.add_argument(
        '--val-every', type=int, default=1000,
        help='Frequencia da validacao em iters.',
    )
    parser.add_argument(
        '--log-every', type=int, default=100,
        help='Frequencia do log no CSV/TB em iters.',
    )
    parser.add_argument(
        '--checkpoint-every', type=int, default=10000,
        help='Frequencia do save de last.pt em iters.',
    )
    return parser.parse_args()


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
    n_train_npz = len(list(train_dir.glob('*.npz')))
    n_val_npz = len(list(val_dir.glob('*.npz')))
    logger.info(f'recons_root = {recons_root}')
    logger.info(f'  train: {n_train_npz} arquivos .npz')
    logger.info(f'  val:   {n_val_npz} arquivos .npz')

    # 2. Validar device
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.error(
            'CUDA solicitado mas nao disponivel. '
            'Em Kaggle: Settings -> Accelerator -> T4 x1.'
        )
        return 3
    if args.device == 'cuda':
        logger.info(f'GPU: {torch.cuda.get_device_name(0)}')
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f'VRAM: {vram:.1f} GB')

    # 3. Carregar config
    config = json.loads(args.config.read_text(encoding='utf-8'))
    total_iters = args.total_iters or config['scheduler']['total_steps_full']
    logger.info(f'Config: {args.config.name}')
    logger.info(f'Total iters: {total_iters}')

    # 4. Datasets e DataLoaders
    train_ds = ReconsSliceDataset(train_dir, masks_dir=None)
    val_ds = ReconsSliceDataset(val_dir, masks_dir=None)
    train_loader = DataLoader(
        train_ds,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(args.device == 'cuda'),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(args.device == 'cuda'),
    )
    logger.info(
        f'Dataset: {len(train_ds)} train slices, {len(val_ds)} val slices'
    )

    # 5. Modulo (Grupo A)
    module = ResidualMagnitudeModule(
        chans=config['model']['chans'],
        num_pool_layers=config['model']['num_pool_layers'],
    )
    n_params = sum(p.numel() for p in module.parameters())
    logger.info(f'ResidualMagnitudeModule: {n_params:,} parametros')

    # 6. Rodar
    result = train(
        module=module,
        loss_fn=resm_loss,
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
        loss_kwargs=None,
        config_snapshot={
            'group': 'A',
            'run': 'full',
            'recons_root': str(recons_root),
            'total_iters': total_iters,
            'training_base': config,
        },
    )

    logger.info('=' * 60)
    logger.info('TREINO CONCLUIDO')
    for k, v in result.items():
        logger.info(f'  {k}: {v}')
    logger.info('=' * 60)
    logger.info(f'Checkpoints em: {args.run_dir.resolve()}')
    logger.info('  last.pt, best.pt, metrics.csv, tb/, config_snapshot.json')
    return 0


if __name__ == '__main__':
    sys.exit(main())
