# Autor: Massanori
# Data: 17/05/2026
# Descrição: Implementacao da pinball loss (quantile loss) usada nos Grupos B
#            (QR) e C (QR-Lesion) do S5. Recebe: predicao q_hat (tensor),
#            ground truth y (tensor de mesmo shape), nivel alpha em (0, 1).
#            Retorna: tensor com loss por pixel (pinball_per_pixel) ou
#            escalar agregado (pinball_loss, reduction=mean/sum). Formulacao
#            equivalente de Koenker e Bassett (1978), Romano et al. (2019)
#            e Giannakopoulos et al. (2026, eq. 1). Protegida por testes
#            em tests/test_uncertainty_losses.py.


"""Pinball loss (quantile loss) por pixel e agregada.

Para um quantil-alvo alpha em (0, 1), a pinball loss e:

    L_alpha(q_hat, y) = max(alpha * (y - q_hat), (alpha - 1) * (y - q_hat))

Equivalentemente, no caso por caso:
    - se y > q_hat:   L = alpha * (y - q_hat)
    - caso contrario: L = (1 - alpha) * (q_hat - y)

Minimizar L_alpha sobre uma amostra de y faz q_hat convergir para o quantil
empirico de y no nivel alpha (Koenker & Bassett, 1978, Teorema 3.4). Aplicada
pixel a pixel sobre pares (target, q_pred), treina dois U-Nets para estimar
os quantis alpha/2 (lower) e 1-alpha/2 (upper) da distribuicao condicional
p(y|x), conforme Giannakopoulos et al. (2026, secao II.B.2).

Refs:
    Koenker, R.; Bassett, G. (1978). Regression Quantiles. Econometrica,
        46(1):33-50.
    Romano, Y.; Patterson, E.; Candes, E. (2019). Conformalized Quantile
        Regression. NeurIPS 32, 3543-3553.
    Giannakopoulos, C. et al. (2026). Pixelwise Uncertainty Quantification
        of Accelerated MRI Reconstruction. arXiv:2601.13236.
"""
from __future__ import annotations

import torch


def pinball_per_pixel(
    q_pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Pinball loss por pixel, sem reducao.

    Util quando a loss precisa ser ponderada espacialmente, como na
    hybrid loss do Grupo C que aplica peso lambda nas regioes de lesao.

    Parameters
    ----------
    q_pred : torch.Tensor
        Predicao do quantil. Qualquer shape.
    target : torch.Tensor
        Ground truth, mesmo shape de q_pred.
    alpha : float
        Nivel do quantil em (0, 1). Para intervalo de cobertura 1-alpha,
        usar alpha/2 (lower) e 1 - alpha/2 (upper).

    Returns
    -------
    torch.Tensor
        Tensor com o mesmo shape de q_pred, contendo a loss por pixel.

    Raises
    ------
    ValueError
        Se alpha esta fora de (0, 1) ou shapes incompativeis.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f'alpha deve estar em (0, 1), recebido {alpha}')
    if q_pred.shape != target.shape:
        raise ValueError(
            f'Shapes incompativeis: q_pred {tuple(q_pred.shape)} vs '
            f'target {tuple(target.shape)}'
        )

    diff = target - q_pred
    # Formulacao com torch.maximum evita branches no autograd e e
    # numericamente equivalente ao if/else; ambas geram o mesmo grafo
    # computacional, mas torch.maximum e mais legivel.
    return torch.maximum(alpha * diff, (alpha - 1.0) * diff)


def pinball_loss(
    q_pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float,
    reduction: str = 'mean',
) -> torch.Tensor:
    """Pinball loss agregada.

    Parameters
    ----------
    q_pred : torch.Tensor
        Predicao do quantil.
    target : torch.Tensor
        Ground truth.
    alpha : float
        Nivel do quantil em (0, 1).
    reduction : str, default 'mean'
        'mean', 'sum' ou 'none'. 'none' retorna o tensor por pixel.

    Returns
    -------
    torch.Tensor
        Loss escalar (mean/sum) ou tensor por pixel ('none').
    """
    per_pixel = pinball_per_pixel(q_pred, target, alpha)
    if reduction == 'mean':
        return per_pixel.mean()
    if reduction == 'sum':
        return per_pixel.sum()
    if reduction == 'none':
        return per_pixel
    raise ValueError(
        f"reduction deve ser 'mean', 'sum' ou 'none', recebido '{reduction}'"
    )
