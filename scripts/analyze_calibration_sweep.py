# Autor: Massanori
# Data: 02/06/2026
# Descrição: Varredura de calibracao (calibration sweep), destino:
#            scripts/analyze_calibration_sweep.py. Para um (grupo, calibrador),
#            calibra sobre o split cal em uma GRADE de niveis nominais (1-alpha)
#            e avalia, no split test, cobertura empirica (global e em lesao) e
#            largura media (global e em lesao). Uma unica passada cal + uma
#            passada test alimentam tres analises de uma vez:
#              - Item 1: comparar CQR-Norm vs CQR-aditivo vs ScaledCP em
#                        Coverage_lesion no nivel 0.90 (isola mecanismo).
#              - Item 2: fronteira de eficiencia (cobertura vs largura) —
#                        comparacao justa "cobertura a largura igualada".
#              - Item 4: curva de confiabilidade (nominal vs empirico).
#            Recebe via CLI: --group {A,B,C}, --calibrator {scaled,cqr,cqr_norm},
#            --checkpoint, --recons-dir (com cal/ e test/), --masks-dir, --output.
#            Gera: CSV tidy (uma linha por nivel) + JSON sumario com SHA-256 do
#            checkpoint para auditoria. Rode 1x por (grupo, calibrador) e
#            concatene os CSVs no plot_calibration_extras.py.
#            Fundamentos: Romano et al. (2019); Lei et al. (2018); Angelopoulos
#            & Bates (2023, cobertura vs nivel nominal); a fronteira de
#            eficiencia segue o principio sharpness-subject-to-calibration.

"""Varredura de nivel nominal: cobertura e largura por calibrador.

Por que uma unica passada serve para 3 itens
--------------------------------------------
Para todos os calibradores deste projeto a cobertura empirica em um threshold
q e' exatamente a fracao de pixels com score <= q (ver equivalencias abaixo),
e a largura do intervalo e' AFIM em q. Logo basta:

  1. (pass cal)  acumular os scores do calibrador -> q(nivel) para cada nivel.
  2. (pass test) acumular, por nivel, contagem de pixels cobertos (score<=q),
                 e somas de (base_width, scale) — independentes de q.

Formulas por calibrador (score s, largura W(q) = base_width + 2*q*scale):
  - ScaledCP (Grupo A):  s = |y-x|/u ; base_width = 0 ; scale = u
       intervalo [x - q*u, x + q*u] ; W = 2*q*u
  - CQR aditivo (B/C):   s = max(lower-y, y-upper) ; base_width = upper-lower ; scale = 1
       intervalo [lower - q, upper + q] ; W = (upper-lower) + 2q
  - CQR normalizado:     s = max(lower-y, y-upper)/w ; base_width = w ; scale = w   (w=upper-lower)
       intervalo [lower - q*w, upper + q*w] ; W = w*(1 + 2q)

cobertura(q) = mean(s <= q) ; W_media(q) = mean(base_width) + 2*q*mean(scale),
calculadas separadamente em global e em pixels de lesao.

Exemplos de uso (rode 5x e concatene):
    python scripts/analyze_calibration_sweep.py --group A --calibrator scaled \\
        --checkpoint .../tcc-mri-resm-checkpoints/best.pt \\
        --recons-dir .../tcc-mri-recons-varnet-brain-4x \\
        --masks-dir  .../tcc-mri-lesion-masks \\
        --output /kaggle/working/sweep_A_scaled.csv
    python scripts/analyze_calibration_sweep.py --group B --calibrator cqr      ... --output sweep_B_cqr.csv
    python scripts/analyze_calibration_sweep.py --group B --calibrator cqr_norm ... --output sweep_B_cqrnorm.csv
    python scripts/analyze_calibration_sweep.py --group C --calibrator cqr      ... --output sweep_C_cqr.csv
    python scripts/analyze_calibration_sweep.py --group C --calibrator cqr_norm ... --output sweep_C_cqrnorm.csv

Refs:
    Romano, Patterson & Candes (2019). Conformalized Quantile Regression. NeurIPS.
    Lei et al. (2018). Distribution-Free Predictive Inference for Regression. JASA.
    Angelopoulos & Bates (2023). Conformal Prediction: A Gentle Introduction. FnT ML.
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

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402
from src.calibration.conformal import cqr_score, scaled_cp_score  # noqa: E402
from src.calibration.adaptive_cqr import cqr_normalized_score, DEFAULT_EPS  # noqa: E402
from src.data import ReconsSliceDataset  # noqa: E402
from src.models import QuantileRegressionModule, ResidualMagnitudeModule  # noqa: E402
from src.training import load_checkpoint  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
)
logger = logging.getLogger('calibration_sweep')

CSV_FIELDS = [
    'group', 'calibrator', 'nominal_coverage', 'alpha', 'q_hat',
    'coverage_global', 'coverage_lesion',
    'width_global', 'width_lesion',
    'n_pixels_global', 'n_pixels_lesion',
]

# Grade de niveis nominais (1 - alpha). Inclui 0.90 (alvo do projeto).
DEFAULT_LEVELS = sorted(set(
    [round(x, 2) for x in np.arange(0.50, 0.951, 0.05)] +
    [0.90, 0.925, 0.95, 0.96, 0.97, 0.98, 0.99]
))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Varredura de nivel nominal por calibrador.')
    p.add_argument('--group', required=True, choices=['A', 'B', 'C'])
    p.add_argument('--calibrator', required=True,
                   choices=['scaled', 'cqr', 'cqr_norm'],
                   help='scaled=ResM(A); cqr=CQR aditivo(B/C); '
                        'cqr_norm=CQR localmente adaptativo(B/C).')
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--recons-dir', type=Path, default=None,
                   help='Dir com subdirs cal/ e test/ (.npz por volume).')
    p.add_argument('--masks-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True, help='CSV de saida.')
    p.add_argument('--eps', type=float, default=DEFAULT_EPS,
                   help='Piso da largura para cqr_norm (DEVE casar com apply).')
    p.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    p.add_argument('--num-workers', type=int, default=2)
    p.add_argument('--chans', type=int, default=32)
    p.add_argument('--num-pool-layers', type=int, default=4)
    p.add_argument('--log-every', type=int, default=100)
    return p.parse_args()


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


def _validate_calibrator(group: str, calibrator: str) -> None:
    if group == 'A' and calibrator != 'scaled':
        raise ValueError("Grupo A (ResM) so admite --calibrator scaled.")
    if group in ('B', 'C') and calibrator == 'scaled':
        raise ValueError("Grupos B/C nao admitem --calibrator scaled "
                         "(use cqr ou cqr_norm).")


def _score_and_widths(calibrator, forward_out, recon, target, eps):
    """Retorna (score, base_width, scale) por pixel para o calibrador.

    score: tensor de nonconformity (cobertura(q) = mean(score <= q)).
    base_width, scale: largura W(q) = base_width + 2*q*scale.
    """
    if calibrator == 'scaled':
        u = forward_out  # ResM -> u(x)
        score = scaled_cp_score(u, recon, target)
        base_width = torch.zeros_like(u)
        scale = u
        return score, base_width, scale
    lower, upper = forward_out['lower'], forward_out['upper']
    if calibrator == 'cqr':
        score = cqr_score(lower, upper, target)
        base_width = (upper - lower)
        scale = torch.ones_like(lower)
        return score, base_width, scale
    # cqr_norm
    w = (upper - lower).clamp_min(eps)
    score = cqr_normalized_score(lower, upper, target, eps=eps)
    base_width = w
    scale = w
    return score, base_width, scale


def _empirical_quantile_corrected(scores_np: np.ndarray, level: float) -> float:
    """Quantile (level)(n+1)/n com correcao finite-sample (Romano et al. 2019).

    Usa np.quantile method='higher' (k-esimo menor), consistente com o
    conformal_quantile do projeto (selecao do ceil((n+1)*level)-esimo).
    """
    n = scores_np.size
    q_level = min(level * (n + 1) / n, 1.0)
    return float(np.quantile(scores_np, q_level, method='higher'))


@torch.no_grad()
def _forward(module, calibrator, recon):
    return module(recon)


def main() -> int:
    args = parse_args()
    try:
        _validate_calibrator(args.group, args.calibrator)
    except ValueError as e:
        logger.error(str(e))
        return 4

    if not args.checkpoint.is_file():
        logger.error(f'Checkpoint nao encontrado: {args.checkpoint}')
        return 2

    try:
        device = resolve_device(args.device)
    except RuntimeError as e:
        logger.error(str(e))
        return 3

    recons_root = (args.recons_dir or cfg.recons_dir()).expanduser().resolve()
    cal_dir, test_dir = recons_root / 'cal', recons_root / 'test'
    for d in (cal_dir, test_dir):
        if not d.is_dir():
            logger.error(f'Split ausente em: {d}')
            return 2
    masks_dir = args.masks_dir.expanduser().resolve()
    if not masks_dir.is_dir():
        logger.error(f'masks_dir ausente: {masks_dir}')
        return 2

    ckpt_sha = compute_sha256(args.checkpoint)
    logger.info(f'Device={device} | checkpoint SHA-256={ckpt_sha[:16]}...')

    if args.group == 'A':
        module = ResidualMagnitudeModule(chans=args.chans,
                                         num_pool_layers=args.num_pool_layers)
    else:
        module = QuantileRegressionModule(chans=args.chans,
                                          num_pool_layers=args.num_pool_layers)
    load_checkpoint(args.checkpoint, module, device=device)
    module = module.to(device).eval()

    levels = list(DEFAULT_LEVELS)
    logger.info(f'Niveis nominais: {levels}')

    # ----- PASS 1: cal -> pool de scores -> q por nivel -----
    cal_ds = ReconsSliceDataset(cal_dir, masks_dir=masks_dir)
    cal_loader = DataLoader(cal_ds, batch_size=1, shuffle=False,
                            num_workers=args.num_workers,
                            pin_memory=(device == 'cuda'))
    logger.info(f'[cal] {len(cal_ds)} slices')
    cal_scores = []
    with torch.no_grad():
        for batch in cal_loader:
            recon = batch['recon'].to(device, non_blocking=True)
            target = batch['target'].to(device, non_blocking=True)
            out = module(recon)
            s, _, _ = _score_and_widths(args.calibrator, out, recon, target, args.eps)
            cal_scores.append(s.cpu().flatten())
    cal_scores_np = torch.cat(cal_scores).to(torch.float64).numpy()
    del cal_scores
    q_by_level = {lvl: _empirical_quantile_corrected(cal_scores_np, lvl)
                  for lvl in levels}
    logger.info(f'[cal] n_pixels={cal_scores_np.size:,} | '
                f'q(0.90)={q_by_level.get(0.90, float("nan")):.6f}')

    # ----- PASS 2: test -> contagens cobertas por nivel + somas de largura -----
    q_arr = np.array([q_by_level[lvl] for lvl in levels], dtype=np.float64)
    covered_global = np.zeros(len(levels), dtype=np.float64)
    covered_lesion = np.zeros(len(levels), dtype=np.float64)
    n_global = 0
    n_lesion = 0
    bw_sum_g = sc_sum_g = 0.0
    bw_sum_l = sc_sum_l = 0.0

    test_ds = ReconsSliceDataset(test_dir, masks_dir=masks_dir)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                             num_workers=args.num_workers,
                             pin_memory=(device == 'cuda'))
    logger.info(f'[test] {len(test_ds)} slices')
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            recon = batch['recon'].to(device, non_blocking=True)
            target = batch['target'].to(device, non_blocking=True)
            mask = batch['lesion_mask'].to(device, non_blocking=True)
            out = module(recon)
            s, base_width, scale = _score_and_widths(
                args.calibrator, out, recon, target, args.eps)

            s_flat = s.detach().reshape(-1).to(torch.float64).cpu().numpy()
            m_flat = (mask.detach().reshape(-1) > 0.5).cpu().numpy()

            n_global += s_flat.size
            n_lesion += int(m_flat.sum())

            bw = base_width.detach().reshape(-1).to(torch.float64).cpu().numpy()
            sc = scale.detach().reshape(-1).to(torch.float64).cpu().numpy()
            bw_sum_g += float(bw.sum()); sc_sum_g += float(sc.sum())
            if m_flat.any():
                bw_sum_l += float(bw[m_flat].sum())
                sc_sum_l += float(sc[m_flat].sum())

            # cobertura por nivel: contagem de score <= q[l]
            # searchsorted no vetor ordenado de scores e' O(P log L); aqui
            # fazemos broadcast simples (L ~ 13 niveis), suficiente.
            le = s_flat[:, None] <= q_arr[None, :]  # (P, L)
            covered_global += le.sum(axis=0)
            if m_flat.any():
                covered_lesion += le[m_flat].sum(axis=0)

            if (i + 1) % args.log_every == 0:
                logger.info(f'  [test {i + 1}/{len(test_ds)}]')

    if n_global == 0:
        logger.error('Split test vazio.')
        return 2

    rows = []
    for j, lvl in enumerate(levels):
        q = q_by_level[lvl]
        cov_g = covered_global[j] / n_global
        cov_l = (covered_lesion[j] / n_lesion) if n_lesion > 0 else float('nan')
        w_g = bw_sum_g / n_global + 2.0 * q * (sc_sum_g / n_global)
        w_l = ((bw_sum_l / n_lesion + 2.0 * q * (sc_sum_l / n_lesion))
               if n_lesion > 0 else float('nan'))
        rows.append({
            'group': args.group,
            'calibrator': args.calibrator,
            'nominal_coverage': lvl,
            'alpha': round(1.0 - lvl, 4),
            'q_hat': q,
            'coverage_global': cov_g,
            'coverage_lesion': cov_l,
            'width_global': w_g,
            'width_lesion': w_l,
            'n_pixels_global': int(n_global),
            'n_pixels_lesion': int(n_lesion),
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f'CSV salvo: {args.output} ({len(rows)} niveis)')

    summary = {
        'group': args.group, 'calibrator': args.calibrator,
        'checkpoint_path': str(args.checkpoint), 'checkpoint_sha256': ckpt_sha,
        'recons_root': str(recons_root), 'masks_dir': str(masks_dir),
        'eps': args.eps, 'levels': levels,
        'cal_n_slices': len(cal_ds), 'cal_n_pixels': int(cal_scores_np.size),
        'test_n_slices': len(test_ds),
        'n_pixels_global': int(n_global), 'n_pixels_lesion': int(n_lesion),
        'created_at': datetime.now(timezone.utc).isoformat(),
    }
    args.output.with_suffix('.summary.json').write_text(
        json.dumps(summary, indent=2), encoding='utf-8')

    # Headline no nivel 0.90 (alvo do projeto)
    r90 = next((r for r in rows if abs(r['nominal_coverage'] - 0.90) < 1e-9), None)
    if r90:
        logger.info('=' * 60)
        logger.info(f'NIVEL 0.90 | Grupo {args.group} | {args.calibrator}')
        logger.info(f'  q_hat            = {r90["q_hat"]:.6f}')
        logger.info(f'  coverage_global  = {r90["coverage_global"]:.4f}')
        logger.info(f'  coverage_lesion  = {r90["coverage_lesion"]:.4f}')
        logger.info(f'  width_global     = {r90["width_global"]:.5f}')
        logger.info(f'  width_lesion     = {r90["width_lesion"]:.5f}')
        logger.info('=' * 60)
    return 0


if __name__ == '__main__':
    sys.exit(main())
