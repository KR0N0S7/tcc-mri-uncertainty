# Autor: Massanori
# Data: 21/05/2026
# Descricao: IoU (Intersection over Union) entre regiao top-X% de uncertainty
#            e regiao top-X% de erro. Recebe: dois tensores de mesma shape
#            (uncertainty e error). Retorna: scalar [0, 1] (IoU) ou dict +
#            AUC para curva multi-threshold. Suporta restricao opcional a
#            uma mascara (e.g. lesao) para IoU local. Necessario para o
#            S5.8: o IoU_uncertainty mede se o modelo aponta para os pixels
#            certos como incertos, complementando o Pearson global (que
#            so mede correlacao de magnitudes).


"""IoU entre mapas de incerteza e erro.

Deixa entrar 2 tensores (uncertainty u e erro |y-x|), pega os top-X% pixels
de cada, e computa IoU = |A intersect B| / |A union B|.

Motivacao: o Pearson global mede so correlacao de magnitudes pixelwise.
IoU(top-X%) mede ALINHAMENTO ESPACIAL: o modelo aponta para os pixels
certos como incertos? Para o Grupo C (QR-Lesion), espera-se IoU mais alto
que para A/B dentro de lesoes — reflexo direto da loss ponderada.

Anchoring do threshold:
    O X em top-X% deve ser escolhido (a) por proporcao de lesao
    (e.g., se lesoes ocupam 3-5% dos pixels, usar X em torno disso),
    ou (b) reportado como curva IoU(X) com integral AUC.
    Recomendacao da banca: reportar AUC ao inves de um unico X.

Refs:
    Jaccard, P. (1912). The distribution of the flora in the alpine
        zone. The New Phytologist 11(2):37-50.
    Rezatofighi et al. (2019). Generalized Intersection over Union.
        CVPR 2019, 658-666.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch

logger = logging.getLogger(__name__)

DEFAULT_TOP_PCTS = (0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50)


def _top_k_mask(
    values: torch.Tensor,
    top_pct: float,
    restrict_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Retorna mascara bool dos top-k pixels onde k = top_pct * N (ou de
    pixels dentro de restrict_mask).

    Usa torch.topk (O(N log k)) em vez de sort (O(N log N)).
    """
    flat = values.flatten()
    if restrict_mask is not None:
        mask_flat = restrict_mask.flatten().bool()
        if not mask_flat.any():
            return torch.zeros_like(flat, dtype=torch.bool)
        # Top-k apenas entre os pixels da mascara
        candidates_idx = mask_flat.nonzero(as_tuple=True)[0]
        candidates_values = flat[candidates_idx]
        n_candidates = candidates_idx.numel()
        k = max(int(round(n_candidates * top_pct)), 1)
        k = min(k, n_candidates)
        _, top_idx_in_candidates = torch.topk(candidates_values, k, largest=True)
        top_global_idx = candidates_idx[top_idx_in_candidates]
        out = torch.zeros_like(flat, dtype=torch.bool)
        out[top_global_idx] = True
        return out
    # Sem restricao: top-k sobre todo o tensor
    n = flat.numel()
    k = max(int(round(n * top_pct)), 1)
    k = min(k, n)
    _, top_idx = torch.topk(flat, k, largest=True)
    out = torch.zeros_like(flat, dtype=torch.bool)
    out[top_idx] = True
    return out


def iou_topk(
    uncertainty: torch.Tensor,
    error: torch.Tensor,
    top_pct: float = 0.05,
    restrict_mask: Optional[torch.Tensor] = None,
) -> float:
    """IoU entre top-X% pixels de uncertainty e top-X% pixels de error.

    Parameters
    ----------
    uncertainty, error : torch.Tensor
        Mesmo shape, tipicamente (1, H, W) ou (H, W). Float.
    top_pct : float, default 0.05
        Fracao em (0, 1) dos pixels considerados como 'altos'.
    restrict_mask : torch.Tensor or None
        Se fornecido, IoU computado apenas entre os pixels da mascara
        (top-X% calculado sobre os candidatos da mascara apenas).
        Util para IoU_lesion.

    Returns
    -------
    float
        IoU em [0, 1]. Retorna 0.0 se a uniao for vazia (degenerado).
    """
    if uncertainty.shape != error.shape:
        raise ValueError(
            f'Shapes incompativeis: uncertainty {tuple(uncertainty.shape)}, '
            f'error {tuple(error.shape)}'
        )
    if not 0.0 < top_pct < 1.0:
        raise ValueError(
            f'top_pct deve estar em (0, 1), recebido {top_pct}'
        )
    if restrict_mask is not None and restrict_mask.shape != uncertainty.shape:
        raise ValueError(
            f'restrict_mask shape {tuple(restrict_mask.shape)} != '
            f'uncertainty shape {tuple(uncertainty.shape)}'
        )

    A = _top_k_mask(uncertainty, top_pct, restrict_mask)
    B = _top_k_mask(error, top_pct, restrict_mask)

    intersection = (A & B).sum().item()
    union = (A | B).sum().item()

    if union == 0:
        return 0.0
    return intersection / union


def iou_curve(
    uncertainty: torch.Tensor,
    error: torch.Tensor,
    top_pcts: Tuple[float, ...] = DEFAULT_TOP_PCTS,
    restrict_mask: Optional[torch.Tensor] = None,
) -> Tuple[dict, float]:
    """IoU(X) para varios X + integral AUC via trapezio.

    Parameters
    ----------
    uncertainty, error, restrict_mask : ver iou_topk.
    top_pcts : tuple de float
        Sequencia ordenada de X em (0, 1). Por padrao
        (1%, 2%, 5%, 10%, 20%, 30%, 50%) — cobre o range de interesse
        para regioes pequenas (lesoes) ate metade da imagem.

    Returns
    -------
    ious : dict {top_pct: iou}
        IoU em cada threshold.
    auc : float
        Integral trapezoidal do IoU ao longo do X.
        Normalizada por (X_max - X_min) para ser independente do range.
    """
    if not all(0.0 < x < 1.0 for x in top_pcts):
        raise ValueError(f'Todos top_pcts devem estar em (0, 1): {top_pcts}')

    xs = sorted(top_pcts)
    ious = {x: iou_topk(uncertainty, error, x, restrict_mask) for x in xs}

    # AUC trapezoidal, normalizada pelo range
    if len(xs) < 2:
        return ious, 0.0
    auc_raw = sum(
        (xs[i + 1] - xs[i]) * (ious[xs[i + 1]] + ious[xs[i]]) / 2.0
        for i in range(len(xs) - 1)
    )
    auc = auc_raw / (xs[-1] - xs[0])
    return ious, auc
