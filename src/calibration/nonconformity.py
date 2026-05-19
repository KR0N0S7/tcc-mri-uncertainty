# Autor: Massanori
# Data: 19/05/2026
# Descrição: Scores de nao-conformidade e calibracao conforme para os Grupos
#            A/B/C do S5. Tres familias de funcoes:
#            (1) nonconformity_qr: score aditivo max(l-y, y-u) usado em B e C
#                (Romano et al., 2019, eq. 8). Positivo fora do intervalo,
#                negativo dentro.
#            (2) nonconformity_resm: score multiplicativo |x-y|/u usado em A
#                (Edupuganti et al., 2021). Sempre nao-negativo. Generaliza
#                o calibration scheme para predicoes pontuais ao inves de
#                intervalos.
#            (3) compute_qhat: quantil empirico com correcao finita de
#                Romano et al. (2019, eq. 9-10), (1-alpha)(n+1)/n.
#            Apply_qhat_{qr,resm} aplica o qhat para gerar o intervalo
#            calibrado [l-qhat, u+qhat] ou [x-qhat*u, x+qhat*u].

"""Scores de nao-conformidade e calibracao.

Para QR (Grupos B e C):
    score(x, y) = max(l(x) - y, y - u(x))
    intervalo calibrado = [l(x) - qhat, u(x) + qhat]

Para ResM (Grupo A):
    score(x, y) = |x - y| / u(x)
    intervalo calibrado = [x - qhat * u(x), x + qhat * u(x)]

Em ambos:
    qhat = quantile empirico de nivel (1-alpha)(n+1)/n sobre os scores
           do conjunto de calibracao.

Sob exchangeability dos splits cal/test, a cobertura marginal e garantida
ser >= 1-alpha (Romano et al., 2019, Theorem 1; Vovk et al., 2005).

Refs:
    Romano, Y.; Patterson, E.; Candes, E. (2019). Conformalized Quantile
        Regression. NeurIPS 32. (Eqs. 8-10, Theorem 1)
    Edupuganti, V. et al. (2021). Uncertainty Quantification in Deep MRI
        Reconstruction. IEEE TMI 40(1):239-250.
    Vovk, V.; Gammerman, A.; Shafer, G. (2005). Algorithmic Learning in a
        Random World. Springer.
"""
from __future__ import annotations

from typing import Tuple, Union

import numpy as np
import torch

NumericArray = Union[torch.Tensor, np.ndarray]


def nonconformity_qr(
    lower: torch.Tensor,
    upper: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Score aditivo CQR (Grupos B e C).

    score = max(lower - target, target - upper)

    Por pixel. Positivo quando target esta fora de [lower, upper], com
    magnitude = distancia ao bound mais proximo. Negativo quando dentro,
    com magnitude = -1 vez a distancia ao bound mais proximo.

    Parameters
    ----------
    lower, upper, target : torch.Tensor
        Mesmo shape (qualquer). Tipicamente (B, 1, H, W).

    Returns
    -------
    torch.Tensor
        Score por pixel, mesmo shape dos inputs.
    """
    if not (lower.shape == upper.shape == target.shape):
        raise ValueError(
            f'Shapes incompativeis: lower {tuple(lower.shape)}, '
            f'upper {tuple(upper.shape)}, target {tuple(target.shape)}'
        )
    return torch.maximum(lower - target, target - upper)


def nonconformity_resm(
    uncertainty: torch.Tensor,
    recon: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Score multiplicativo escalonado (Grupo A).

    score = |recon - target| / (uncertainty + eps)

    Por pixel. Sempre nao-negativo. Generaliza a calibracao para
    predicoes pontuais (u nao e um intervalo, e um escalar por pixel).
    O eps evita divisao por zero em pixels com u proximo de zero.

    Parameters
    ----------
    uncertainty : torch.Tensor
        Saida do modulo ResM, u(x). Esperado >= 0.
    recon, target : torch.Tensor
        Reconstrucao e ground truth, mesmo shape de uncertainty.
    eps : float, default 1e-8
        Estabilidade numerica. Pequeno suficiente para nao distorcer.

    Returns
    -------
    torch.Tensor
        Score por pixel, mesmo shape dos inputs.
    """
    if not (uncertainty.shape == recon.shape == target.shape):
        raise ValueError(
            f'Shapes incompativeis: u {tuple(uncertainty.shape)}, '
            f'recon {tuple(recon.shape)}, target {tuple(target.shape)}'
        )
    error = torch.abs(recon - target)
    return error / (uncertainty + eps)


def compute_qhat(
    scores: NumericArray,
    alpha: float,
) -> float:
    """Quantil conforme com correcao finita (Romano et al., 2019, eq. 9-10).

    qhat = empirical (1-alpha)(1 + 1/n) quantile of scores

    A correcao (n+1)/n garante cobertura marginal >= 1-alpha sob
    exchangeability (Theorem 1 do paper). Para n >= 1000, a correcao e
    negligible (~0.1% no nivel do quantil). Para n pequeno, e essencial.

    Parameters
    ----------
    scores : torch.Tensor or np.ndarray
        Scores de nao-conformidade. 1D ou flattenable.
    alpha : float
        Nivel de miscoverage em (0, 1). Cobertura nominal = 1 - alpha.

    Returns
    -------
    float
        Valor de qhat.

    Raises
    ------
    ValueError
        Se alpha fora de (0, 1) ou scores vazio.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f'alpha deve estar em (0, 1), recebido {alpha}')

    if isinstance(scores, torch.Tensor):
        scores_np = scores.detach().cpu().numpy()
    else:
        scores_np = np.asarray(scores)
    scores_np = scores_np.reshape(-1)

    n = scores_np.size
    if n == 0:
        raise ValueError('scores esta vazio')

    # Correcao finita: (1-alpha) * (n+1)/n, capada em 1.0 para n pequeno.
    quantile_level = min(1.0, (1.0 - alpha) * (n + 1) / n)
    return float(np.quantile(scores_np, quantile_level))


def apply_qhat_qr(
    lower: torch.Tensor,
    upper: torch.Tensor,
    qhat: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Aplica calibracao conforme aditiva (Grupos B/C).

    Retorna o intervalo calibrado:
        lower_cal = lower - qhat
        upper_cal = upper + qhat

    Para qhat > 0, o intervalo se alarga (compensa under-coverage do
    intervalo bruto). Para qhat < 0, se estreita (raro, ocorre quando o
    intervalo bruto e super conservador).

    Parameters
    ----------
    lower, upper : torch.Tensor
        Predicoes brutas do modulo QR.
    qhat : float
        Valor obtido de compute_qhat sobre o split cal.

    Returns
    -------
    (torch.Tensor, torch.Tensor)
        (lower_cal, upper_cal).
    """
    return lower - qhat, upper + qhat


def apply_qhat_resm(
    uncertainty: torch.Tensor,
    recon: torch.Tensor,
    qhat: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Aplica calibracao conforme multiplicativa (Grupo A).

    Retorna o intervalo calibrado centrado na reconstrucao:
        lower_cal = recon - qhat * uncertainty
        upper_cal = recon + qhat * uncertainty

    Parameters
    ----------
    uncertainty : torch.Tensor
        u(x) saida do modulo ResM.
    recon : torch.Tensor
        Reconstrucao VarNet.
    qhat : float
        Valor obtido de compute_qhat sobre o split cal.

    Returns
    -------
    (torch.Tensor, torch.Tensor)
        (lower_cal, upper_cal).
    """
    return recon - qhat * uncertainty, recon + qhat * uncertainty
