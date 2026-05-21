# Autor: Massanori
# Data: 19/05/2026
# Descrição: Calibracao conforme de um modulo treinado dos Grupos A/B/C sobre
#            o split cal. Recebe via CLI: --group {A, B, C}, --checkpoint
#            (best.pt do grupo), --recons-dir (com subdir cal/), opcionalmente
#            --masks-dir (para C usar Dataset com mascaras — nao afeta a
#            calibracao por si, mas mantem consistencia). Saida: arquivo JSON
#            com q_hat, metadata, e snapshot do checkpoint usado, para
#            auditoria. O JSON e o que vira input do compute_metrics.py
#            do S5.8 e do notebook da 4a entrega.


"""Calibracao conforme de um checkpoint treinado.

Dado um best.pt de um dos 3 grupos do S5, computa q_hat usando o split
cal e salva como JSON para uso downstream (S5.8 metrics, S5.9 docs).

O mesmo script atende os 3 grupos via flag --group:
    A -> ResidualMagnitudeModule + calibrate_resm (scaled CP)
    B -> QuantileRegressionModule + calibrate_qr (CQR)
    C -> QuantileRegressionLesionModule + calibrate_qr (CQR, mesmo metodo do B)

Roda com:
    # Grupo A
    python scripts/calibrate.py --group A \\
        --checkpoint /kaggle/input/.../tcc-mri-resm-checkpoints/best.pt \\
        --recons-dir /kaggle/input/.../tcc-mri-recons-varnet-brain-4x \\
        --output /kaggle/working/q_hat_A.json

    # Grupo B
    python scripts/calibrate.py --group B \\
        --checkpoint /kaggle/input/.../tcc-mri-qr-checkpoints/best.pt \\
        --recons-dir /kaggle/input/.../tcc-mri-recons-varnet-brain-4x \\
        --output /kaggle/working/q_hat_B.json

    # Grupo C (igual a B; masks_dir opcional, nao afeta calibracao)
    python scripts/calibrate.py --group C \\
        --checkpoint /kaggle/input/.../tcc-mri-qr-lesion-checkpoints/best.pt \\
        --recons-dir /kaggle/input/.../tcc-mri-recons-varnet-brain-4x \\
        --output /kaggle/working/q_hat_C.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import math
import numpy as np
import src.calibration.conformal as conformal_mod
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402
from src.calibration import calibrate_qr, calibrate_resm  # noqa: E402
from src.data import ReconsSliceDataset  # noqa: E402
from src.losses import DEFAULT_ALPHA  # noqa: E402
from src.models import (  # noqa: E402
    QuantileRegressionModule,
    ResidualMagnitudeModule,
)
from src.training import load_checkpoint  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
)
logger = logging.getLogger('calibrate')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Calibracao conforme de um modulo treinado dos Grupos A/B/C.'
    )
    parser.add_argument(
        '--group', required=True, choices=['A', 'B', 'C'],
        help='Grupo do TCC (define modulo + metodo de calibracao).',
    )
    parser.add_argument(
        '--checkpoint', type=Path, required=True,
        help='best.pt do treino. Para Grupo C: usa o mesmo best.pt do C.',
    )
    parser.add_argument(
        '--recons-dir', type=Path, default=None,
        help='Diretorio raiz com subdir cal/. Default: cfg.recons_dir().',
    )
    parser.add_argument(
        '--masks-dir', type=Path, default=None,
        help='Opcional. Para C: dir flat com <volume_id>.pt. Nao afeta '
             'calibracao em si, apenas consistencia com o treino.',
    )
    parser.add_argument(
        '--output', type=Path, required=True,
        help='Path do JSON de saida com q_hat + metadata.',
    )
    parser.add_argument(
        '--alpha', type=float, default=DEFAULT_ALPHA,
        help=f'Nivel de miscoverage. Default {DEFAULT_ALPHA}.',
    )
    parser.add_argument(
        '--device', default='auto', choices=['auto', 'cpu', 'cuda'],
    )
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--chans', type=int, default=32)
    parser.add_argument('--num-pool-layers', type=int, default=4)
    return parser.parse_args()


def compute_sha256(path: Path) -> str:
    """SHA-256 do checkpoint, para auditoria."""
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()

def resolve_device(device_arg: str) -> str:
    if device_arg == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    if device_arg == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA solicitado mas nao disponivel.')
    return device_arg


def conformal_quantile_numpy_fallback(scores_flat: torch.Tensor, alpha: float) -> float:
    s = scores_flat.detach().reshape(-1).to(torch.float32).cpu().numpy()
    n = int(s.size)
    if n == 0:
        raise ValueError('scores_flat vazio na calibracao.')
    k = int(math.ceil((n + 1) * (1.0 - alpha)))
    k = min(max(k, 1), n)
    idx = k - 1
    return float(np.partition(s, idx)[idx])


def install_cpu_quantile_hotfix() -> None:
    original_fn = conformal_mod.conformal_quantile

    def wrapped(scores_flat: torch.Tensor, alpha: float):
        try:
            return original_fn(scores_flat, alpha)
        except RuntimeError as e:
            if 'quantile() input tensor is too large' in str(e).lower():
                logger.warning(
                    'torch.quantile falhou por tensor grande; usando fallback numpy.partition.'
                )
                return conformal_quantile_numpy_fallback(scores_flat, alpha)
            raise

    conformal_mod.conformal_quantile = wrapped

def main() -> int:
    args = parse_args()

    try:
        device = resolve_device(args.device)
    except RuntimeError as e:
        logger.error(str(e))
        return 3

    if device == 'cpu':
        install_cpu_quantile_hotfix()

    logger.info(f'Device: {device}')

    recons_root = (args.recons_dir or cfg.recons_dir()).expanduser().resolve()
    cal_dir = recons_root / 'cal'
    if not cal_dir.is_dir():
        logger.error(f'Split cal ausente em: {cal_dir}')
        return 2

    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.error('CUDA solicitado mas nao disponivel.')
        return 3
    logger.info(f'Device: {args.device}')
    logger.info(f'recons_root: {recons_root}')
    logger.info(f'checkpoint: {args.checkpoint}')

    # 1. Hash do checkpoint para auditoria
    ckpt_sha = compute_sha256(args.checkpoint)
    logger.info(f'checkpoint SHA-256: {ckpt_sha}')

    # 2. Instanciar modulo correto e selecionar metodo de calibracao
    if args.group == 'A':
        module = ResidualMagnitudeModule(
            chans=args.chans, num_pool_layers=args.num_pool_layers,
        )
        calibrate_fn = calibrate_resm
        method_label = 'ScaledCP'
        masks_dir_used = None  # ResM ignora mascara
    else:
        # B e C compartilham arquitetura (QuantileRegressionLesionModule e alias)
        module = QuantileRegressionModule(
            chans=args.chans, num_pool_layers=args.num_pool_layers,
        )
        calibrate_fn = calibrate_qr
        method_label = 'CQR'
        # Para C, opcional carregar mascaras (so para consistencia com treino;
        # calibrate_qr ignora a mask key do batch)
        masks_dir_used = args.masks_dir

    n_params = sum(p.numel() for p in module.parameters())
    logger.info(f'Modulo Grupo {args.group}: {n_params:,} parametros')

    # 3. Carregar checkpoint
    load_checkpoint(args.checkpoint, module, device=device)
    module = module.to(device)
    logger.info(f'Checkpoint carregado.')

    # 4. Cal loader
    cal_ds = ReconsSliceDataset(cal_dir, masks_dir=masks_dir_used)
    cal_loader = DataLoader(
        cal_ds, batch_size=1, shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == 'cuda'),
    )
    logger.info(f'Cal dataset: {len(cal_ds)} slices')

    # 5. Calibrar
    result = calibrate_fn(
        module, cal_loader, alpha=args.alpha, device=device,
    )

    # 6. Montar JSON de saida com metadata para auditoria
    payload = {
        'group': args.group,
        'method': method_label,
        'q_hat': result['q_hat'],
        'alpha': result['alpha'],
        'n_pixels': result['n_pixels'],
        'n_batches': result['n_batches'],
        'mean_score': result['mean_score'],
        'checkpoint_path': str(args.checkpoint),
        'checkpoint_sha256': ckpt_sha,
        'recons_root': str(recons_root),
        'cal_dir': str(cal_dir),
        'cal_n_slices': len(cal_ds),
        'masks_dir': str(masks_dir_used) if masks_dir_used else None,
        'chans': args.chans,
        'num_pool_layers': args.num_pool_layers,
        'created_at': datetime.now(timezone.utc).isoformat(),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2), encoding='utf-8',
    )

    logger.info('=' * 60)
    logger.info(f'CALIBRACAO CONCLUIDA (Grupo {args.group})')
    logger.info(f'  q_hat:        {result["q_hat"]:.6f}')
    logger.info(f'  alpha:        {result["alpha"]}')
    logger.info(f'  n_pixels:     {result["n_pixels"]:,}')
    logger.info(f'  mean_score:   {result["mean_score"]:.6f}')
    logger.info(f'  Output JSON:  {args.output}')
    logger.info('=' * 60)

    # Heads-up sobre a esperanca para B (paper reporta lambda_cal ~ 1.54)
    if args.group == 'B':
        logger.info(
            f'Heads-up: Giannakopoulos et al. (2026) reporta lambda_cal ~ 1.54 '
            f'para brain 4x. Sua replicacao deu q_hat={result["q_hat"]:.4f}. '
            f'Se Pearson do Pearson check estiver > paper, q_hat pode estar '
            f'mais baixo (intervalos brutos ja proximos da escala correta).'
        )

    return 0


if __name__ == '__main__':
    sys.exit(main())
