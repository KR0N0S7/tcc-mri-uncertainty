# Autor: Massanori
# Data: 04/06/2026
# Descrição: Calibração e avaliação Mondrian por sequência, destino:
#            scripts/analyze_mondrian_coverage.py. Para um (grupo, calibrador),
#            calibra um q por sequência (AXFLAIR/AXT1/AXT1POST) sobre o split
#            cal e avalia, no split test, a cobertura (global e em lesão) e a
#            largura por sequência, comparando o esquema MARGINAL (q único) com
#            o MONDRIAN (q por sequência). Endereça diretamente o achado do
#            S5-extras item 3: sub-cobertura concentrada em AXT1.
#            Recebe via CLI: --group {A,B,C}, --calibrator {scaled,cqr,cqr_norm},
#            --checkpoint, --recons-dir (com cal/ e test/), --masks-dir,
#            --alpha (default 0.10), --output. Gera: CSV tidy
#            [scheme, stratum, q_hat, coverage_global, coverage_lesion,
#            width_global, width_lesion, n_pixels_global, n_pixels_lesion,
#            used_fallback] + JSON sumário com SHA-256 do checkpoint.
#            Fundamentos: Vovk et al. (2005, Mondrian CP); Romano et al. (2019);
#            Barber et al. (2021, limites da cobertura condicional).

"""Calibração Mondrian por sequência: marginal vs condicional ao estrato.

Pergunta científica
--------------------
A calibração marginal entrega P(y in C(x)) >= 1 - alpha, mas o S5-extras
mostrou sub-cobertura sistemática em AXT1. A calibração Mondrian (um q por
sequência) deve, por construção, levar a cobertura *global por sequência* ao
alvo. A questão empírica é se ela também reduz o gap *em lesão* dentro de cada
sequência — em particular se recupera a cobertura de lesão em AXT1.

Procedimento (para um grupo/calibrador):
  1. (cal) acumula scores por sequência -> q_s (Mondrian) e q_marginal (pool).
  2. (test) por slice, conhece a sequência; avalia cobertura/largura sob o q
     marginal e sob o q_s da sequência, agregando por sequência e no total.

Saída: linhas (scheme in {marginal, mondrian}) x (stratum in {ALL, AXFLAIR,
AXT1, AXT1POST}). Comparar a coluna coverage_lesion entre os dois schemes por
sequência responde à pergunta.

Exemplo (uma linha):
    python scripts/analyze_mondrian_coverage.py --group A --calibrator scaled --checkpoint <A/best.pt> --recons-dir <recons> --masks-dir <masks> --output results/mondrian_A_scaled.csv

Refs:
    Vovk, Gammerman & Shafer (2005). Algorithmic Learning in a Random World.
    Romano, Patterson & Candes (2019). Conformalized Quantile Regression. NeurIPS.
    Barber et al. (2021). The limits of distribution-free conditional predictive inference.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402
from src.calibration.adaptive_cqr import DEFAULT_EPS  # noqa: E402
from src.calibration.mondrian import (  # noqa: E402
    score_and_widths, empirical_conformal_quantile, mondrian_quantiles,
)
from src.data import ReconsSliceDataset  # noqa: E402
from src.models import QuantileRegressionModule, ResidualMagnitudeModule  # noqa: E402
from src.training import load_checkpoint  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(name)s: %(message)s')
logger = logging.getLogger('mondrian_coverage')

CSV_FIELDS = [
    'scheme', 'stratum', 'group', 'calibrator', 'alpha', 'q_hat',
    'coverage_global', 'coverage_lesion', 'width_global', 'width_lesion',
    'n_pixels_global', 'n_pixels_lesion', 'used_fallback',
]
MIN_N_STRATUM = 50_000  # abaixo disso, estrato usa o q marginal (instável)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Calibração Mondrian por sequência.')
    p.add_argument('--group', required=True, choices=['A', 'B', 'C'])
    p.add_argument('--calibrator', required=True, choices=['scaled', 'cqr', 'cqr_norm'])
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--recons-dir', type=Path, default=None)
    p.add_argument('--masks-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--alpha', type=float, default=0.10)
    p.add_argument('--eps', type=float, default=DEFAULT_EPS)
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


def resolve_device(arg: str) -> str:
    if arg == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    if arg == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA solicitado mas indisponível.')
    return arg


def _validate(group, calibrator):
    if group == 'A' and calibrator != 'scaled':
        raise ValueError('Grupo A admite apenas --calibrator scaled.')
    if group in ('B', 'C') and calibrator == 'scaled':
        raise ValueError('Grupos B/C não admitem --calibrator scaled.')


def _seq_of(batch) -> str:
    s = batch['sequence']
    if isinstance(s, (list, tuple)):
        s = s[0]
    return str(s)


def main() -> int:
    args = parse_args()
    try:
        _validate(args.group, args.calibrator)
    except ValueError as e:
        logger.error(str(e)); return 4
    if not args.checkpoint.is_file():
        logger.error(f'Checkpoint ausente: {args.checkpoint}'); return 2
    try:
        device = resolve_device(args.device)
    except RuntimeError as e:
        logger.error(str(e)); return 3

    recons_root = (args.recons_dir or cfg.recons_dir()).expanduser().resolve()
    cal_dir, test_dir = recons_root / 'cal', recons_root / 'test'
    masks_dir = args.masks_dir.expanduser().resolve()
    for d in (cal_dir, test_dir, masks_dir):
        if not d.is_dir():
            logger.error(f'Diretório ausente: {d}'); return 2

    ckpt_sha = compute_sha256(args.checkpoint)
    logger.info(f'Device={device} | alpha={args.alpha} | ckpt={ckpt_sha[:16]}...')

    if args.group == 'A':
        module = ResidualMagnitudeModule(chans=args.chans, num_pool_layers=args.num_pool_layers)
    else:
        module = QuantileRegressionModule(chans=args.chans, num_pool_layers=args.num_pool_layers)
    load_checkpoint(args.checkpoint, module, device=device)
    module = module.to(device).eval()

    # ---- PASS 1 (cal): scores por sequência + pool marginal ----
    cal_ds = ReconsSliceDataset(cal_dir, masks_dir=masks_dir)
    cal_loader = DataLoader(cal_ds, batch_size=1, shuffle=False,
                            num_workers=args.num_workers, pin_memory=(device == 'cuda'))
    logger.info(f'[cal] {len(cal_ds)} slices')
    scores_by_seq = defaultdict(list)
    with torch.no_grad():
        for batch in cal_loader:
            recon = batch['recon'].to(device, non_blocking=True)
            target = batch['target'].to(device, non_blocking=True)
            out = module(recon)
            s, _, _ = score_and_widths(args.calibrator, out, recon, target, args.eps)
            scores_by_seq[_seq_of(batch)].append(s.detach().cpu().flatten().to(torch.float64).numpy())

    scores_by_seq = {k: np.concatenate(v) for k, v in scores_by_seq.items()}
    all_scores = np.concatenate(list(scores_by_seq.values()))
    q_marginal = empirical_conformal_quantile(all_scores, args.alpha)
    q_mondrian = mondrian_quantiles(scores_by_seq, args.alpha,
                                    min_n=MIN_N_STRATUM, fallback_q=q_marginal)
    logger.info(f'[cal] q_marginal={q_marginal:.6f} | '
                f'q_mondrian={ {k: round(v["q_hat"], 6) for k, v in q_mondrian.items()} }')

    # ---- PASS 2 (test): cobertura/largura por (scheme, sequência) ----
    # acumuladores: chave (scheme, stratum) -> dict de somas
    acc = defaultdict(lambda: {'cov_g': 0.0, 'cov_l': 0.0, 'n_g': 0, 'n_l': 0,
                               'bw_g': 0.0, 'bw_l': 0.0, 'sc_g': 0.0, 'sc_l': 0.0})

    def add(scheme, stratum, score, bw, sc, mask, q):
        a = acc[(scheme, stratum)]
        cov = score <= q
        a['cov_g'] += float(cov.sum()); a['n_g'] += score.size
        a['bw_g'] += float(bw.sum()); a['sc_g'] += float(sc.sum())
        if mask.any():
            a['cov_l'] += float(cov[mask].sum()); a['n_l'] += int(mask.sum())
            a['bw_l'] += float(bw[mask].sum()); a['sc_l'] += float(sc[mask].sum())

    test_ds = ReconsSliceDataset(test_dir, masks_dir=masks_dir)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                             num_workers=args.num_workers, pin_memory=(device == 'cuda'))
    logger.info(f'[test] {len(test_ds)} slices')
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            recon = batch['recon'].to(device, non_blocking=True)
            target = batch['target'].to(device, non_blocking=True)
            mask_t = batch['lesion_mask'].to(device, non_blocking=True)
            seq = _seq_of(batch)
            out = module(recon)
            s, bw, sc = score_and_widths(args.calibrator, out, recon, target, args.eps)
            s = s.detach().reshape(-1).to(torch.float64).cpu().numpy()
            bw = bw.detach().reshape(-1).to(torch.float64).cpu().numpy()
            sc = sc.detach().reshape(-1).to(torch.float64).cpu().numpy()
            m = (mask_t.detach().reshape(-1) > 0.5).cpu().numpy()

            q_seq = q_mondrian.get(seq, {'q_hat': q_marginal})['q_hat']
            # marginal: agrega no total (ALL) e por sequência
            add('marginal', 'ALL', s, bw, sc, m, q_marginal)
            add('marginal', seq, s, bw, sc, m, q_marginal)
            # mondrian: q por sequência
            add('mondrian', 'ALL', s, bw, sc, m, q_seq)
            add('mondrian', seq, s, bw, sc, m, q_seq)
            if (i + 1) % args.log_every == 0:
                logger.info(f'  [test {i + 1}/{len(test_ds)}]')

    # ---- montar linhas ----
    def q_for(scheme, stratum):
        if scheme == 'marginal':
            return q_marginal, False
        if stratum == 'ALL':
            return float('nan'), False  # mondrian ALL usa q variável por slice
        info = q_mondrian.get(stratum, {'q_hat': q_marginal, 'used_fallback': True})
        return info['q_hat'], info.get('used_fallback', False)

    rows = []
    for (scheme, stratum), a in sorted(acc.items()):
        q_hat, fb = q_for(scheme, stratum)
        cov_g = a['cov_g'] / a['n_g'] if a['n_g'] else float('nan')
        cov_l = a['cov_l'] / a['n_l'] if a['n_l'] else float('nan')
        # largura: precisa do q por pixel; para ALL-mondrian é variável, então
        # reportamos largura apenas onde q é único (marginal, ou mondrian por seq)
        if scheme == 'mondrian' and stratum == 'ALL':
            w_g = w_l = float('nan')
        else:
            w_g = (a['bw_g'] + 2.0 * q_hat * a['sc_g']) / a['n_g'] if a['n_g'] else float('nan')
            w_l = (a['bw_l'] + 2.0 * q_hat * a['sc_l']) / a['n_l'] if a['n_l'] else float('nan')
        rows.append({
            'scheme': scheme, 'stratum': stratum, 'group': args.group,
            'calibrator': args.calibrator, 'alpha': args.alpha, 'q_hat': q_hat,
            'coverage_global': cov_g, 'coverage_lesion': cov_l,
            'width_global': w_g, 'width_lesion': w_l,
            'n_pixels_global': a['n_g'], 'n_pixels_lesion': a['n_l'],
            'used_fallback': fb,
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader(); w.writerows(rows)
    logger.info(f'CSV salvo: {args.output} ({len(rows)} linhas)')

    args.output.with_suffix('.summary.json').write_text(json.dumps({
        'group': args.group, 'calibrator': args.calibrator, 'alpha': args.alpha,
        'checkpoint_path': str(args.checkpoint), 'checkpoint_sha256': ckpt_sha,
        'recons_root': str(recons_root), 'masks_dir': str(masks_dir),
        'q_marginal': q_marginal,
        'q_mondrian': {k: v['q_hat'] for k, v in q_mondrian.items()},
        'mondrian_n_pixels': {k: v['n_pixels'] for k, v in q_mondrian.items()},
        'cal_n_slices': len(cal_ds), 'test_n_slices': len(test_ds),
        'created_at': datetime.now(timezone.utc).isoformat(),
    }, indent=2), encoding='utf-8')

    # resumo legível: foco no gap em lesão por sequência, marginal vs mondrian
    logger.info('=' * 64)
    logger.info(f'COBERTURA EM LESÃO por sequência | grupo {args.group} / {args.calibrator}')
    seqs = sorted({st for (_, st) in acc if st != 'ALL'})
    for st in seqs:
        m = acc[('marginal', st)]; mo = acc[('mondrian', st)]
        cm = m['cov_l'] / m['n_l'] if m['n_l'] else float('nan')
        cmo = mo['cov_l'] / mo['n_l'] if mo['n_l'] else float('nan')
        logger.info(f'  {st:>10}: marginal={cm:.4f} -> mondrian={cmo:.4f} '
                    f'(q_seq={q_mondrian.get(st, {}).get("q_hat", float("nan")):.5f})')
    logger.info('=' * 64)
    return 0


if __name__ == '__main__':
    sys.exit(main())
