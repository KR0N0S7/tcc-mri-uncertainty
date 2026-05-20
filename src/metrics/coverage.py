# Autor: Massanori
# Data: 19/05/2026
# Descrição: Metricas de cobertura empirica para os intervalos calibrados
#            do S5.8. Recebe: tensors lower/upper/target/lesion_mask.
#            Retorna: empirical_coverage (P(y in [L, U]) com IC Clopper-Pearson),
#            mean_interval_width. As metricas operam pixelwise e podem
#            ser restritas a regioes de lesao (mascara binaria) para a
#            analise por regiao onde a contribuicao original do Grupo C atua.
#            Clopper-Pearson e implementado inline (sem scipy) via incomplete
#            beta function por continued fraction (Numerical Recipes §6.4).


"""Metricas de cobertura para intervalos de predicao calibrados.

Duas familias principais:
    - Coverage: fracao empirica de pixels com target dentro de [L, U].
      A garantia teorica da calibracao conforme e coverage >= 1 - alpha.
    - Interval width: mean(upper - lower). Trade-off com coverage —
      intervalos mais estreitos com mesma cobertura sao melhores.

Ambas podem ser computadas globalmente ou restritas as mascaras de lesao
para a analise por-regiao (S5.8) onde o Grupo C deve diferenciar.

Refs:
    Romano, Patterson & Candes (2019), Secao 3.3.
    Angelopoulos & Bates (2023), Secao 2.1.
    Clopper, C.J.; Pearson, E.S. (1934). The use of confidence or
        fiducial limits illustrated in the case of the binomial.
        Biometrika 26(4):404-413.
    Press et al. (2007). Numerical Recipes 3rd Ed. §6.4 (incomplete beta).
"""
from __future__ import annotations

import logging
import math
from typing import Optional, Tuple

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cobertura empirica
# ---------------------------------------------------------------------------

def empirical_coverage(
    lower: torch.Tensor,
    upper: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    confidence: float = 0.95,
) -> dict:
    """Cobertura empirica pixelwise: P(y em [L, U]).

    Parameters
    ----------
    lower, upper, target : torch.Tensor
        Mesmo shape, tipicamente (B, 1, H, W).
    mask : torch.Tensor or None
        Se fornecida, conta apenas pixels onde mask > 0. Shape compativel
        (broadcastable) com target.
    confidence : float, default 0.95
        Nivel do IC Clopper-Pearson para a proporcao.

    Returns
    -------
    dict
        - coverage : float, fracao de pixels cobertos
        - n_covered : int
        - n_total : int
        - ci_lower, ci_upper : Clopper-Pearson exato (binomial)
    """
    if not (lower.shape == upper.shape == target.shape):
        raise ValueError(
            f'Shape mismatch: lower {tuple(lower.shape)}, '
            f'upper {tuple(upper.shape)}, target {tuple(target.shape)}'
        )

    inside = (target >= lower) & (target <= upper)

    if mask is not None:
        mask_bool = mask > 0
        if mask_bool.shape != target.shape:
            mask_bool = mask_bool.expand_as(target)
        inside = inside & mask_bool
        n_total = int(mask_bool.sum().item())
    else:
        n_total = int(target.numel())

    if n_total == 0:
        return {
            'coverage': float('nan'),
            'n_covered': 0,
            'n_total': 0,
            'ci_lower': float('nan'),
            'ci_upper': float('nan'),
        }

    n_covered = int(inside.sum().item())
    coverage = n_covered / n_total
    ci_lower, ci_upper = clopper_pearson_interval(
        n_covered, n_total, confidence=confidence
    )

    return {
        'coverage': coverage,
        'n_covered': n_covered,
        'n_total': n_total,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
    }


# ---------------------------------------------------------------------------
# Interval width
# ---------------------------------------------------------------------------

def mean_interval_width(
    lower: torch.Tensor,
    upper: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> float:
    """Largura media do intervalo predito: mean(upper - lower).

    Trade-off com cobertura: em cobertura fixa, mais estreito e melhor.
    Apos calibracao, revela qual metodo (A/B/C) gera intervalos mais
    apertados para a mesma cobertura nominal.

    Parameters
    ----------
    lower, upper : torch.Tensor
        Mesmo shape. Apos calibracao, ja com q_hat aplicado.
    mask : torch.Tensor or None
        Se fornecida, computa apenas em pixels onde mask > 0.

    Returns
    -------
    float
        Largura media. NaN se mask zera todos os pixels.
    """
    if lower.shape != upper.shape:
        raise ValueError(
            f'Shape mismatch: lower {tuple(lower.shape)}, upper {tuple(upper.shape)}'
        )

    width = upper - lower

    if mask is not None:
        mask_bool = mask > 0
        if mask_bool.shape != width.shape:
            mask_bool = mask_bool.expand_as(width)
        n = int(mask_bool.sum().item())
        if n == 0:
            return float('nan')
        return width[mask_bool].mean().item()

    return width.mean().item()


# ---------------------------------------------------------------------------
# Clopper-Pearson exato (sem scipy)
# ---------------------------------------------------------------------------

def clopper_pearson_interval(
    k: int,
    n: int,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """IC exato Clopper-Pearson para uma proporcao binomial.

    Retorna (lower, upper) tal que P(theta_true em [lower, upper]) >= confidence.

    Implementado via incomplete beta regularizada (sem scipy):
        I(p; a, b) = beta_inc_reg(a, b, p)
    e sua inversa por bisection.

    Parameters
    ----------
    k : int
        Numero de sucessos (pixels cobertos).
    n : int
        Total de tentativas (pixels avaliados).
    confidence : float, default 0.95
        Nivel de confianca do IC.

    Returns
    -------
    tuple
        (lower, upper) com proporcao em [0, 1].

    Ref: Clopper & Pearson (1934).
    """
    if k < 0 or n <= 0 or k > n:
        raise ValueError(f'Invalid (k, n) = ({k}, {n})')
    if not 0 < confidence < 1:
        raise ValueError(f'confidence deve estar em (0, 1), recebido {confidence}')

    alpha = 1 - confidence

    # Limite inferior: F(k-1; n, p_low) = 1 - alpha/2
    # Equivalente: p_low = beta_inv(alpha/2; k, n - k + 1)
    if k == 0:
        lower = 0.0
    else:
        lower = _beta_quantile(alpha / 2, k, n - k + 1)

    if k == n:
        upper = 1.0
    else:
        upper = _beta_quantile(1 - alpha / 2, k + 1, n - k)

    return lower, upper


def _beta_quantile(
    p: float,
    a: float,
    b: float,
    tol: float = 1e-9,
    max_iter: int = 200,
) -> float:
    """Inversa da incomplete beta regularizada via bisection.

    Encontra q tal que I_q(a, b) = p, com I em [0, 1]. Convergencia
    monotonica garantida pela monotonia de I_q em q.
    """
    if not 0 < p < 1:
        return 0.0 if p <= 0 else 1.0

    lo, hi = 0.0, 1.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if _beta_incomplete_reg(mid, a, b) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def _beta_incomplete_reg(x: float, a: float, b: float) -> float:
    """Incomplete beta regularizada I_x(a, b) via continued fraction.

    Numerical Recipes 3rd Ed., §6.4. Implementacao em log-space para
    estabilidade quando a, b ~ 1e6+ (cobertura sobre milhoes de pixels).
    """
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0

    # Termo log para estabilidade numerica
    log_bt = (
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log(1 - x)
    )
    bt = math.exp(log_bt)

    # Selecao da forma para garantir convergencia rapida do CF
    if x < (a + 1) / (a + b + 2):
        return bt * _betacf(x, a, b) / a
    return 1 - bt * _betacf(1 - x, b, a) / b


def _betacf(
    x: float, a: float, b: float,
    max_iter: int = 200, eps: float = 1e-12,
) -> float:
    """Continued fraction de Lentz para incomplete beta. Numerical Recipes §6.4."""
    qab = a + b
    qap = a + 1
    qam = a - 1
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        # Termo par
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c
        # Termo impar
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h
