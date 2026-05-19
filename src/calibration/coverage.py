# Autor: Massanori
# Data: 19/05/2026
# Descrição: Calibracao end-to-end e avaliacao de cobertura para os 3 grupos
#            do S5. Recebe: modulo treinado, DataLoader, alpha, kind ('qr' ou
#            'resm'), device. Calibrate() roda o modulo no split cal,
#            agrega scores pixel-wise e retorna qhat. Evaluate() roda o modulo
#            no split test, aplica o qhat e retorna dict com cobertura global,
#            cobertura per-lesao, mean_width, e listas per-slice para uso
#            posterior em Friedman/Nemenyi (Demsar, 2006). coverage_stats() e
#            funcao pura que computa estatisticas por batch — util para tests.

"""Calibracao end-to-end e avaliacao de cobertura.

Pipeline tipico:
    qhat, scores = calibrate(module, cal_loader, alpha=0.10, kind='qr', ...)
    metrics = evaluate(module, test_loader, qhat, kind='qr', ...)

Memory: scores podem ter ~75M floats em pixel-wise sobre split cal
completo. Coletados em CPU em chunks (1 batch por vez) -> ~300 MB peak.
Passes 1x no cal split, 1x no test split.

Refs:
    Romano, Y.; Patterson, E.; Candes, E. (2019). NeurIPS 32.
    Angelopoulos, A.N. et al. (2022). Image-to-Image Regression with
        Distribution-Free Uncertainty Quantification. ICML.
    Demsar, J. (2006). Statistical Comparisons of Classifiers over
        Multiple Data Sets. JMLR 7:1-30.
"""
from __future__ import annotations

import logging
from typing import Callable, Dict, Tuple, Union

import torch
from torch.utils.data import DataLoader

from src.calibration.nonconformity import (
    apply_qhat_qr,
    apply_qhat_resm,
    compute_qhat,
    nonconformity_qr,
    nonconformity_resm,
)

logger = logging.getLogger(__name__)

ValidKind = str  # 'qr' or 'resm'


def coverage_stats(
    lower_cal: torch.Tensor,
    upper_cal: torch.Tensor,
    target: torch.Tensor,
    lesion_mask: torch.Tensor = None,
) -> dict:
    """Estatisticas de cobertura para um batch ja calibrado.

    Funcao pura: nao usa modulo nem device. Usada internamente por
    evaluate() e tambem util para testes unitarios com input controlado.

    Parameters
    ----------
    lower_cal, upper_cal : torch.Tensor
        Intervalo apos aplicacao de qhat. Mesmo shape de target.
    target : torch.Tensor
        Ground truth.
    lesion_mask : torch.Tensor or None
        Mascara binaria. Se None, retorna so estatisticas globais.

    Returns
    -------
    dict
        Chaves: n_total, n_covered, sum_width, n_lesion, n_lesion_covered.
        Todas escalares (int ou float). Util para acumular ao longo de
        multiplos batches.
    """
    if not (lower_cal.shape == upper_cal.shape == target.shape):
        raise ValueError(
            f'Shapes incompativeis: lower {tuple(lower_cal.shape)}, '
            f'upper {tuple(upper_cal.shape)}, target {tuple(target.shape)}'
        )

    covered = (target >= lower_cal) & (target <= upper_cal)
    width = upper_cal - lower_cal

    stats = {
        'n_total': int(covered.numel()),
        'n_covered': int(covered.sum().item()),
        'sum_width': float(width.sum().item()),
        'n_lesion': 0,
        'n_lesion_covered': 0,
    }

    if lesion_mask is not None:
        mask_b = lesion_mask.bool()
        if mask_b.shape != target.shape:
            raise ValueError(
                f'Shape de lesion_mask {tuple(mask_b.shape)} != target '
                f'{tuple(target.shape)}'
            )
        stats['n_lesion'] = int(mask_b.sum().item())
        stats['n_lesion_covered'] = int((covered & mask_b).sum().item())

    return stats


def calibrate(
    module: torch.nn.Module,
    cal_loader: DataLoader,
    alpha: float,
    kind: ValidKind,
    device: Union[str, torch.device] = 'cuda',
    log_every: int = 0,
) -> Tuple[float, torch.Tensor]:
    """Calcula qhat sobre o split cal.

    Roda module no cal_loader, agrega scores pixel-wise em CPU,
    computa o quantil empirico com correcao finita.

    Parameters
    ----------
    module : nn.Module
        Modulo treinado (ResM, QR, ou QR-Lesion).
    cal_loader : DataLoader
        Loader sobre o split cal.
    alpha : float
        Nivel de miscoverage em (0, 1). Cobertura nominal = 1 - alpha.
    kind : str
        'qr' para Grupos B/C, 'resm' para Grupo A.
    device : str or torch.device
        Device de inferencia.
    log_every : int, default 0
        Se > 0, loga progresso a cada N batches.

    Returns
    -------
    (float, torch.Tensor)
        (qhat, all_scores_1d). all_scores util para inspecao posterior
        (histograma, sensibilidade a alpha).
    """
    if kind not in ('qr', 'resm'):
        raise ValueError(f"kind deve ser 'qr' ou 'resm', recebido '{kind}'")

    module.eval()
    scores_list = []

    with torch.no_grad():
        for i, batch in enumerate(cal_loader):
            recon = batch['recon'].to(device, non_blocking=True)
            target = batch['target'].to(device, non_blocking=True)
            pred = module(recon)

            if kind == 'qr':
                scores = nonconformity_qr(pred['lower'], pred['upper'], target)
            else:  # resm
                scores = nonconformity_resm(pred, recon, target)

            scores_list.append(scores.flatten().cpu())

            if log_every and (i + 1) % log_every == 0:
                logger.info(f'  calibrate: batch {i+1} processado')

    all_scores = torch.cat(scores_list)
    qhat = compute_qhat(all_scores, alpha)
    logger.info(
        f'calibrate (kind={kind}, alpha={alpha}): '
        f'qhat={qhat:.6f} sobre n={all_scores.numel()} pixels'
    )
    return qhat, all_scores


def evaluate(
    module: torch.nn.Module,
    test_loader: DataLoader,
    qhat: float,
    kind: ValidKind,
    device: Union[str, torch.device] = 'cuda',
    log_every: int = 0,
) -> dict:
    """Avalia cobertura no split test apos aplicar qhat.

    Computa:
        - coverage_global: % pixels cobertos no test split
        - coverage_lesion: idem restrito as mascaras de lesao
        - mean_width: largura media do intervalo calibrado
        - per_slice_coverage / per_slice_width: listas para Friedman test

    Parameters
    ----------
    module : nn.Module
    test_loader : DataLoader
        Loader sobre o split test. ReconsSliceDataset deve ter masks_dir
        para que lesion_mask seja util.
    qhat : float
        Valor obtido via calibrate() no split cal.
    kind : str
        'qr' ou 'resm'.
    device : str or torch.device
    log_every : int, default 0

    Returns
    -------
    dict
        coverage_global, coverage_lesion, mean_width,
        per_slice_coverage (list), per_slice_width (list),
        n_total, n_covered, n_lesion, n_lesion_covered.
    """
    if kind not in ('qr', 'resm'):
        raise ValueError(f"kind deve ser 'qr' ou 'resm', recebido '{kind}'")

    module.eval()
    totals = {
        'n_total': 0,
        'n_covered': 0,
        'sum_width': 0.0,
        'n_lesion': 0,
        'n_lesion_covered': 0,
    }
    per_slice_coverage = []
    per_slice_width = []

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            recon = batch['recon'].to(device, non_blocking=True)
            target = batch['target'].to(device, non_blocking=True)
            lesion_mask = batch['lesion_mask'].to(device, non_blocking=True)
            pred = module(recon)

            if kind == 'qr':
                lower_cal, upper_cal = apply_qhat_qr(pred['lower'], pred['upper'], qhat)
            else:  # resm
                lower_cal, upper_cal = apply_qhat_resm(pred, recon, qhat)

            stats = coverage_stats(lower_cal, upper_cal, target, lesion_mask)
            for k in totals:
                totals[k] += stats[k]

            # Per-slice
            covered = (target >= lower_cal) & (target <= upper_cal)
            per_slice_coverage.append(covered.float().mean().item())
            per_slice_width.append((upper_cal - lower_cal).mean().item())

            if log_every and (i + 1) % log_every == 0:
                logger.info(f'  evaluate: batch {i+1} processado')

    coverage_lesion = (
        totals['n_lesion_covered'] / totals['n_lesion']
        if totals['n_lesion'] > 0
        else float('nan')
    )

    return {
        'qhat': qhat,
        'kind': kind,
        'coverage_global': totals['n_covered'] / totals['n_total'],
        'coverage_lesion': coverage_lesion,
        'mean_width': totals['sum_width'] / totals['n_total'],
        'per_slice_coverage': per_slice_coverage,
        'per_slice_width': per_slice_width,
        **totals,
    }
