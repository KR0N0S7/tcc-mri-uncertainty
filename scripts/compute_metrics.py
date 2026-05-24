# Autor: Massanori
# Data: 21/05/2026
# Descricao: Pipeline S5.8 — computa metricas por slice para um grupo
#            (A, B ou C) sobre o split test, salvando CSV detalhado +
#            JSON sumario. Recebe via CLI: --group, --checkpoint, --qhat
#            (JSON do S5.7), --recons-dir (com test/), --masks-dir,
#            --output (CSV). Para cada slice computa:
#              - Coverage global e em lesoes (sob exchangeability,
#                Romano et al., 2019, Teorema 1)
#              - Mean interval width global e em lesoes
#              - IoU(top-X%) global e em lesoes (X default 0.05)
#              - ULAS em lesoes + null baseline com z-score
#            Os CSVs por grupo alimentam o S5.9 (analise estatistica
#            Friedman+Nemenyi, BCa bootstrap, Clopper-Pearson).


"""Computa metricas do S5.8 para um grupo (A/B/C) sobre o split test.

Roda com:
    python scripts/compute_metrics.py --group A \\
        --checkpoint /kaggle/input/.../best.pt \\
        --qhat /kaggle/input/.../q_hat_A.json \\
        --recons-dir /kaggle/input/.../tcc-mri-recons-varnet-brain-4x \\
        --masks-dir  /kaggle/input/.../tcc-mri-lesion-masks \\
        --output /kaggle/working/metrics_A.csv

Duracao tipica: ~5 min/grupo em CPU (~750 slices test, ULAS x10 perms),
~2 min em GPU.

Output CSV: uma linha por slice com colunas:
    volume_id, slice_idx, sequence, group, method, q_hat,
    n_pixels_total, n_pixels_lesion,
    coverage_global, coverage_lesion,
    mean_width_global, mean_width_lesion,
    iou_topk_global, iou_topk_lesion,
    ulas_lesion, ulas_null_mean, ulas_z_score,
    mean_uncertainty, mean_error.

Output JSON sumario (paralelo ao CSV): estatisticas agregadas
(mean, std, median, n_valid) por metrica.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402
from src.calibration import apply_cqr_interval, apply_resm_interval  # noqa: E402
from src.data import ReconsSliceDataset  # noqa: E402
from src.losses import DEFAULT_ALPHA  # noqa: E402
from src.metrics.iou import iou_topk  # noqa: E402
from src.metrics.ulas import ulas_with_null  # noqa: E402
from src.models import QuantileRegressionModule, ResidualMagnitudeModule  # noqa: E402
from src.training import load_checkpoint  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
)
logger = logging.getLogger('compute_metrics')

CSV_FIELDS = [
    'volume_id', 'slice_idx', 'sequence', 'group', 'method', 'q_hat',
    'n_pixels_total', 'n_pixels_lesion',
    'coverage_global', 'coverage_lesion',
    'mean_width_global', 'mean_width_lesion',
    'iou_topk_global', 'iou_topk_lesion',
    'ulas_lesion', 'ulas_null_mean', 'ulas_z_score',
    'mean_uncertainty', 'mean_error',
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Computa metricas S5.8 para um grupo sobre split test.'
    )
    parser.add_argument('--group', required=True, choices=['A', 'B', 'C'])
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--qhat', type=Path, required=True,
                        help='JSON do S5.7 (q_hat_X.json).')
    parser.add_argument('--recons-dir', type=Path, default=None,
                        help='Dir com split test/ (.npz por volume).')
    parser.add_argument('--masks-dir', type=Path, required=True,
                        help='Dir com mascaras .pt (necessario para '
                             'Coverage_lesion, IoU_lesion, ULAS).')
    parser.add_argument('--output', type=Path, required=True,
                        help='CSV de saida.')
    parser.add_argument('--top-pct', type=float, default=0.05,
                        help='X em top-X%% para IoU (default 0.05).')
    parser.add_argument('--n-perms-ulas', type=int, default=10,
                        help='Permutacoes para null baseline do ULAS.')
    parser.add_argument('--alpha', type=float, default=DEFAULT_ALPHA)
    parser.add_argument('--device', default='auto',
                        choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--chans', type=int, default=32)
    parser.add_argument('--num-pool-layers', type=int, default=4)
    parser.add_argument('--log-every', type=int, default=50,
                        help='Imprime progresso a cada N slices.')
    return parser.parse_args()


def compute_sha256(path: Path) -> str:
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


def _safe_mean(t: torch.Tensor, mask: Optional[torch.Tensor]) -> float:
    """Media de t restrita a mask. Retorna NaN se mask vazia."""
    if mask is None:
        return float(t.mean().item())
    if mask.sum() == 0:
        return float('nan')
    return float(t[mask].mean().item())


def _compute_interval(
    group: str,
    forward_out,
    recon: torch.Tensor,
    q_hat: float,
):
    """Aplica intervalo calibrado conforme o grupo.

    Returns (lower_cal, upper_cal, uncertainty_for_logging).
    """
    if group == 'A':
        u = forward_out  # ResM retorna u(x)
        lower_cal, upper_cal = apply_resm_interval(recon, u, q_hat)
        return lower_cal, upper_cal, u
    # B ou C: QR retorna dict com lower/upper
    lower, upper = forward_out['lower'], forward_out['upper']
    lower_cal, upper_cal = apply_cqr_interval(lower, upper, q_hat)
    # 'Uncertainty' simbolica para logging = half-width pre-calibracao
    u = (upper - lower) / 2.0
    return lower_cal, upper_cal, u


def _aggregate_summary(rows: list) -> dict:
    """Estatisticas agregadas (mean, std, median, n_valid) por metrica."""
    metric_keys = [
        'coverage_global', 'coverage_lesion',
        'mean_width_global', 'mean_width_lesion',
        'iou_topk_global', 'iou_topk_lesion',
        'ulas_lesion', 'ulas_null_mean', 'ulas_z_score',
        'mean_uncertainty', 'mean_error',
    ]
    summary = {}
    for k in metric_keys:
        values = [r[k] for r in rows if not math.isnan(r[k])]
        if not values:
            summary[k] = {'mean': float('nan'), 'std': float('nan'),
                          'median': float('nan'), 'n_valid': 0}
            continue
        t = torch.tensor(values)
        summary[k] = {
            'mean': float(t.mean().item()),
            'std': float(t.std(unbiased=True).item()) if len(values) > 1 else 0.0,
            'median': float(t.median().item()),
            'n_valid': len(values),
        }
    return summary


def main() -> int:
    args = parse_args()

    # Carrega q_hat do JSON do S5.7
    if not args.qhat.is_file():
        logger.error(f'q_hat JSON nao encontrado: {args.qhat}')
        return 2
    qhat_payload = json.loads(args.qhat.read_text())
    q_hat = float(qhat_payload['q_hat'])
    method = qhat_payload['method']
    logger.info(f'q_hat = {q_hat:.6f} (metodo: {method}, '
                f'alpha = {qhat_payload["alpha"]})')

    if not args.checkpoint.is_file():
        logger.error(f'Checkpoint nao encontrado: {args.checkpoint}')
        return 2

    ckpt_sha = compute_sha256(args.checkpoint)
    logger.info(f'checkpoint SHA-256: {ckpt_sha[:16]}...')

    try:
        device = resolve_device(args.device)
    except RuntimeError as e:
        logger.error(str(e))
        return 3
    logger.info(f'Device: {device}')

    recons_root = (args.recons_dir or cfg.recons_dir()).expanduser().resolve()
    test_dir = recons_root / 'test'
    if not test_dir.is_dir():
        logger.error(f'Split test ausente em: {test_dir}')
        return 2

    masks_dir = args.masks_dir.expanduser().resolve()
    if not masks_dir.is_dir():
        logger.error(f'masks_dir ausente: {masks_dir}')
        return 2

    # Instancia modelo do grupo
    if args.group == 'A':
        module = ResidualMagnitudeModule(
            chans=args.chans, num_pool_layers=args.num_pool_layers
        )
    else:
        module = QuantileRegressionModule(
            chans=args.chans, num_pool_layers=args.num_pool_layers
        )

    n_params = sum(p.numel() for p in module.parameters())
    logger.info(f'Modulo Grupo {args.group}: {n_params:,} parametros')
    load_checkpoint(args.checkpoint, module, device=device)
    module = module.to(device).eval()
    logger.info('Checkpoint carregado.')

    # Dataset test (com mascaras!)
    test_ds = ReconsSliceDataset(test_dir, masks_dir=masks_dir)
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == 'cuda'),
    )
    logger.info(f'Test dataset: {len(test_ds)} slices')

    rows = []
    n_lesion_slices = 0

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            recon = batch['recon'].to(device, non_blocking=True)
            target = batch['target'].to(device, non_blocking=True)
            lesion_mask = batch['lesion_mask'].to(device, non_blocking=True)

            forward_out = module(recon)
            lower_cal, upper_cal, u = _compute_interval(
                args.group, forward_out, recon, q_hat,
            )

            # Move para CPU para metricas (Sobel + IoU + permutacoes)
            target_c = target.detach().cpu()
            lower_c = lower_cal.detach().cpu()
            upper_c = upper_cal.detach().cpu()
            u_c = u.detach().cpu()
            mask_c = lesion_mask.detach().cpu()

            # Erro absoluto (ground truth)
            error_c = (target_c - recon.detach().cpu()).abs()

            # Coverage (booleano: target esta dentro do intervalo)
            inside = (target_c >= lower_c) & (target_c <= upper_c)
            mask_bool = mask_c > 0.5

            cov_global = inside.float().mean().item()
            cov_lesion = _safe_mean(inside.float(), mask_bool)

            # Interval width
            width = upper_c - lower_c
            w_global = width.mean().item()
            w_lesion = _safe_mean(width, mask_bool)

            # IoU(top-X%): u vs error. Aceita (1,1,H,W); achata internamente.
            iou_g = iou_topk(u_c, error_c, top_pct=args.top_pct)
            if mask_bool.any():
                iou_l = iou_topk(u_c, error_c, top_pct=args.top_pct,
                                 restrict_mask=mask_c)
            else:
                iou_l = float('nan')

            # ULAS em lesoes + null baseline
            n_lesion = int(mask_bool.sum().item())
            if n_lesion > 0:
                # ulas espera (H, W) ou (B,1,H,W); o batch ja vem como (1,1,H,W)
                ulas_result = ulas_with_null(
                    u_c, error_c, mask_c,
                    n_permutations=args.n_perms_ulas,
                    seed=42 + i,  # seed varia por slice para nao correlacionar
                )
                ulas_val = ulas_result['ulas']
                ulas_null_mean = ulas_result['null_mean']
                ulas_z = ulas_result['z_score']
                n_lesion_slices += 1
            else:
                ulas_val = float('nan')
                ulas_null_mean = float('nan')
                ulas_z = float('nan')

            row = {
                'volume_id': batch['volume_id'][0],
                'slice_idx': int(batch['slice_idx'][0].item()) if torch.is_tensor(batch['slice_idx']) else int(batch['slice_idx'][0]),
                'sequence': batch['sequence'][0],
                'group': args.group,
                'method': method,
                'q_hat': q_hat,
                'n_pixels_total': int(target_c.numel()),
                'n_pixels_lesion': n_lesion,
                'coverage_global': cov_global,
                'coverage_lesion': cov_lesion,
                'mean_width_global': w_global,
                'mean_width_lesion': w_lesion,
                'iou_topk_global': iou_g,
                'iou_topk_lesion': iou_l,
                'ulas_lesion': ulas_val,
                'ulas_null_mean': ulas_null_mean,
                'ulas_z_score': ulas_z,
                'mean_uncertainty': float(u_c.mean().item()),
                'mean_error': float(error_c.mean().item()),
            }
            rows.append(row)

            if (i + 1) % args.log_every == 0:
                logger.info(
                    f'  [{i + 1}/{len(test_ds)}] vol={row["volume_id"][:24]}... '
                    f'slice={row["slice_idx"]}, '
                    f'cov_g={cov_global:.3f}, '
                    f'cov_l={cov_lesion:.3f}, '
                    f'ulas={ulas_val:.3f}'
                )

    # Salva CSV
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f'CSV salvo: {args.output} ({len(rows)} linhas)')

    # Sumario
    summary = _aggregate_summary(rows)
    summary_payload = {
        'group': args.group,
        'method': method,
        'q_hat': q_hat,
        'alpha': qhat_payload['alpha'],
        'checkpoint_path': str(args.checkpoint),
        'checkpoint_sha256': ckpt_sha,
        'qhat_path': str(args.qhat),
        'qhat_sha256_from_json': qhat_payload.get('checkpoint_sha256', None),
        'recons_root': str(recons_root),
        'masks_dir': str(masks_dir),
        'top_pct': args.top_pct,
        'n_perms_ulas': args.n_perms_ulas,
        'n_slices_total': len(rows),
        'n_slices_with_lesion': n_lesion_slices,
        'metrics_summary': summary,
        'created_at': datetime.now(timezone.utc).isoformat(),
    }
    summary_path = args.output.with_suffix('.summary.json')
    summary_path.write_text(json.dumps(summary_payload, indent=2),
                            encoding='utf-8')
    logger.info(f'Sumario salvo: {summary_path}')

    # Print headline
    logger.info('=' * 60)
    logger.info(f'METRICAS S5.8 CONCLUIDAS (Grupo {args.group})')
    logger.info(f'  Slices totais:     {len(rows)}')
    logger.info(f'  Slices com lesao:  {n_lesion_slices}')
    for k in ['coverage_global', 'coverage_lesion',
              'mean_width_global', 'mean_width_lesion',
              'iou_topk_global', 'iou_topk_lesion',
              'ulas_lesion', 'ulas_z_score']:
        s = summary[k]
        logger.info(f'  {k:<22} mean={s["mean"]:.4f}, '
                    f'std={s["std"]:.4f}, n={s["n_valid"]}')
    logger.info('=' * 60)

    return 0


if __name__ == '__main__':
    sys.exit(main())
