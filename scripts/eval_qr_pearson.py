# Autor: Massanori
# Data: 19/05/2026
# Descrição: Sanity check pos-treino do Grupo B (QR). Carrega o checkpoint
#            best.pt e computa a Pearson correlation entre uncertainty media
#            ((upper-lower)/2) e error medio (|recon-target|) por slice no
#            split escolhido (default: val). Compara com o valor reportado
#            por Giannakopoulos et al. (2026, Tabela 1) de ~0.91 para brain
#            4x. Recebe: --recons-dir, --checkpoint (best.pt), --split.
#            Saida: imprime r, p, n_slices e diagnostico de PASS/FAIL contra
#            threshold (default 0.85). NAO precisa de scipy: Pearson e
#            implementado inline (numpy) e p-value via Fisher z-transform.


"""Sanity check do Grupo B: Pearson(uncertainty, error) por slice.

Esperado: r ~ 0.91 para brain 4x (Giannakopoulos et al., 2026, Tabela 1).
Se r < threshold (default 0.85), o treino nao convergiu como esperado —
investigar antes de prosseguir para S5.4 (Grupo C) e S5.7 (calibracao).

A Pearson aqui usa o intervalo bruto (pre-conformal). Conformal calibration
(S5.7) ajusta a largura absoluta para atingir cobertura, mas nao altera
significativamente a correlacao com o erro (Romano et al., 2019, secao 3.3).

Refs:
    Giannakopoulos, C. et al. (2026). Pixelwise Uncertainty Quantification
        of Accelerated MRI Reconstruction. arXiv:2601.13236. (Tabela 1)
    Romano, Y.; Patterson, E.; Candes, E. (2019). Conformalized Quantile
        Regression. NeurIPS 32.
    Fisher, R.A. (1921). On the probable error of a coefficient of
        correlation deduced from a small sample. Metron 1:3-32.
"""
from __future__ import annotations

import argparse
import logging
import sys
from math import erf, sqrt
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import ReconsSliceDataset  # noqa: E402
from src.models import QuantileRegressionModule  # noqa: E402
from src.training import load_checkpoint  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
)
logger = logging.getLogger('eval_qr_pearson')


def pearson(x: np.ndarray, y: np.ndarray) -> tuple:
    """Pearson correlation com p-value via Fisher z-transform.

    Implementacao inline para evitar dependencia em scipy. Para n > 30,
    a aproximacao de Fisher (1921) e essencialmente equivalente ao
    scipy.stats.pearsonr.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(x)
    if n < 3 or len(y) != n:
        raise ValueError(f'Inputs invalidos: n={n}, len(y)={len(y)}')

    mx, my = x.mean(), y.mean()
    xm, ym = x - mx, y - my
    num = (xm * ym).sum()
    den = float(np.sqrt((xm ** 2).sum() * (ym ** 2).sum()))
    if den < 1e-12:
        return 0.0, 1.0
    r = num / den

    if n > 3:
        # Fisher z-transform: z = 0.5 * ln((1+r)/(1-r)) ~ N(0, 1/(n-3))
        # Sob H0: rho=0, z*sqrt(n-3) ~ N(0,1)
        z = 0.5 * np.log((1 + r) / max(1 - r, 1e-12)) * np.sqrt(n - 3)
        p = 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(z) / sqrt(2.0))))
    else:
        p = float('nan')
    return float(r), float(p)


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Sanity check Grupo B: Pearson(uncertainty, error) por slice.'
    )
    parser.add_argument('--recons-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True,
                        help='best.pt do Grupo B.')
    parser.add_argument('--split', default='val', choices=['val', 'cal', 'test'])
    parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--chans', type=int, default=32)
    parser.add_argument('--num-pool-layers', type=int, default=4)
    parser.add_argument('--threshold', type=float, default=0.85,
                        help='r >= threshold passa. Default 0.85 (paper reporta ~0.91).')
    parser.add_argument('--num-workers', type=int, default=2)
    args = parser.parse_args()

    split_dir = args.recons_dir / args.split
    if not split_dir.is_dir():
        logger.error(f'Diretorio ausente: {split_dir}')
        return 2

    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning('CUDA solicitado mas nao disponivel; usando CPU.')
        args.device = 'cpu'

    logger.info(f'Carregando dataset de {split_dir}...')
    ds = ReconsSliceDataset(split_dir, masks_dir=None)
    loader = DataLoader(
        ds, batch_size=1, shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(args.device == 'cuda'),
    )
    logger.info(f'  {len(ds)} slices')

    logger.info(f'Carregando modulo de {args.checkpoint}...')
    module = QuantileRegressionModule(
        chans=args.chans,
        num_pool_layers=args.num_pool_layers,
    )
    load_checkpoint(args.checkpoint, module, device=args.device)
    module = module.to(args.device).eval()

    logger.info('Computando uncertainty e error por slice...')
    uncertainties = []
    errors = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            recon = batch['recon'].to(args.device, non_blocking=True)
            target = batch['target'].to(args.device, non_blocking=True)
            pred = module(recon)

            # Half-width do intervalo predito = (upper - lower) / 2.
            # Representa a magnitude da incerteza atribuida ao slice.
            uncertainty = ((pred['upper'] - pred['lower']) / 2).mean().item()
            error = (recon - target).abs().mean().item()

            uncertainties.append(uncertainty)
            errors.append(error)

            if (i + 1) % 100 == 0:
                logger.info(f'  {i+1}/{len(ds)} slices')

    uncertainties = np.array(uncertainties)
    errors = np.array(errors)
    r, p = pearson(uncertainties, errors)

    logger.info('=' * 60)
    logger.info(f'PEARSON SANITY CHECK — split {args.split}')
    logger.info(f'  n slices:           {len(ds)}')
    logger.info(f'  mean uncertainty:   {uncertainties.mean():.6f}')
    logger.info(f'  mean error:         {errors.mean():.6f}')
    logger.info(f'  Pearson r:          {r:.4f}')
    logger.info(f'  Pearson p:          {p:.6e}')
    logger.info(f'  Esperado (paper):   ~0.91 (Giannakopoulos et al., 2026, Tabela 1)')
    logger.info(f'  Threshold:          {args.threshold}')
    logger.info('=' * 60)

    if r < args.threshold:
        logger.error(
            f'FALHOU: r = {r:.4f} < {args.threshold}. '
            f'Treino nao convergiu como esperado. Investigar antes do Grupo C.'
        )
        return 1
    logger.info('PASSOU: correlacao adequada com o erro de reconstrucao.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
