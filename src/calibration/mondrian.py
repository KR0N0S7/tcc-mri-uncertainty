# Autor: Massanori
# Data: 04/06/2026
# Descrição: Calibração conforme Mondrian (condicional a estrato discreto),
#            destino: src/calibration/mondrian.py. Motivado pelo achado do
#            S5-extras (item 3): a sub-cobertura em lesão é concentrada na
#            sequência AXT1. A calibração marginal usa um único q para todos
#            os pixels; a Mondrian (Vovk et al., 2005) particiona o espaço em
#            estratos discretos (aqui, a sequência: AXFLAIR/AXT1/AXT1POST) e
#            calibra um q por estrato, entregando cobertura condicional ao
#            estrato — atingível em distribution-free, ao contrário da
#            cobertura condicional plena (Barber et al., 2021).
#            Recebe: scores de não-conformidade já computados, agrupados por
#            estrato (saída de qualquer um dos calibradores scaled/cqr/cqr_norm).
#            Gera: dict {estrato -> q_hat}. A função de aplicação seleciona o q
#            do estrato do slice em test-time.
#            Fundamentos: Vovk et al. (2005, Mondrian CP); Romano et al. (2019);
#            Lei et al. (2018). A garantia: para cada estrato s com n_s pixels
#            de calibração trocáveis com os de test no mesmo estrato,
#            P(y in C(x) | estrato(x)=s) >= 1 - alpha.

"""Calibração conforme Mondrian condicional a estrato discreto.

Por que Mondrian
----------------
A garantia do conformal marginal (Romano et al., 2019, Teorema 1) é
P(y in C(x)) >= 1 - alpha, marginal sobre todos os pixels. Ela não impede
sub-cobertura sistemática em sub-populações — exatamente o que o S5-extras
observou em AXT1 (cobertura em lesão 0,61-0,73 vs ~0,87 em AXT1POST).
A calibração Mondrian (Vovk et al., 2005) recupera a garantia *condicional ao
estrato*: particiona o espaço em estratos discretos finitos {s} e calibra um
q_s independente por estrato, sobre os scores de calibração daquele estrato.
Isso é atingível em distribution-free (ao contrário da cobertura condicional
plena, impossível segundo Barber et al., 2021), ao custo de menos amostras
por estrato (q_s tem incerteza maior quando n_s é pequeno).

Estrato usado aqui: a sequência de aquisição (AXFLAIR/AXT1/AXT1POST), que é
constante por slice. Logo, em test-time, todos os pixels de um slice usam o
q da sua sequência.

Garantia (por estrato s, sob exchangeability cal-test dentro de s):
    P(y_pix in C(x) | sequence(x)=s) >= 1 - alpha
com C(x) montado pelo mesmo calibrador (scaled/cqr/cqr_norm), trocando o q
único por q_s.

Refs
----
    Vovk, V.; Gammerman, A.; Shafer, G. (2005). Algorithmic Learning in a
        Random World. Springer. (Mondrian conformal prediction, Cap. 4)
    Romano, Y.; Patterson, E.; Candes, E. (2019). Conformalized Quantile
        Regression. NeurIPS, 32:3543-3553.
    Barber, R.F. et al. (2021). The limits of distribution-free conditional
        predictive inference. Information and Inference, 10(2):455-482.
    Lei, J. et al. (2018). Distribution-Free Predictive Inference for
        Regression. JASA, 113(523):1094-1111.
"""
from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import torch

from src.calibration.conformal import cqr_score, scaled_cp_score
from src.calibration.adaptive_cqr import cqr_normalized_score, DEFAULT_EPS

logger = logging.getLogger(__name__)

VALID_CALIBRATORS = ('scaled', 'cqr', 'cqr_norm')


def score_and_widths(calibrator: str, forward_out, recon, target,
                     eps: float = DEFAULT_EPS):
    """Score de não-conformidade e parâmetros de largura por pixel.

    Unifica os três calibradores do projeto. A cobertura num threshold q é
    a fração de pixels com score <= q; a largura do intervalo é
    W(q) = base_width + 2*q*scale.

    - scaled (Grupo A): score = |y-x|/u ; base_width = 0 ; scale = u
    - cqr (B/C aditivo): score = max(lower-y, y-upper) ; base_width = upper-lower ; scale = 1
    - cqr_norm (B/C normalizado): score = max(lower-y, y-upper)/w ; base_width = w ; scale = w

    Returns
    -------
    (score, base_width, scale) : torch.Tensor x3, mesmo shape da entrada.
    """
    if calibrator not in VALID_CALIBRATORS:
        raise ValueError(f'calibrator inválido: {calibrator!r} '
                         f'(use um de {VALID_CALIBRATORS})')
    if calibrator == 'scaled':
        u = forward_out
        return scaled_cp_score(u, recon, target), torch.zeros_like(u), u
    lower, upper = forward_out['lower'], forward_out['upper']
    if calibrator == 'cqr':
        return cqr_score(lower, upper, target), (upper - lower), torch.ones_like(lower)
    w = (upper - lower).clamp_min(eps)
    return cqr_normalized_score(lower, upper, target, eps=eps), w, w


def empirical_conformal_quantile(scores: np.ndarray, alpha: float,
                                 finite_sample: bool = True) -> float:
    """Quantil conforme empírico de um array 1-D de scores.

    Seleciona o quantil (1-alpha) com correção finite-sample
    (1-alpha)(n+1)/n (Romano et al., 2019), via np.quantile method='higher'
    (k-ésimo menor), consistente com src/calibration/conformal.conformal_quantile.

    Parameters
    ----------
    scores : np.ndarray (1-D)
    alpha : float, nível de miscoverage (e.g. 0.10 -> cobertura 0.90).
    finite_sample : bool, aplica a correção (n+1)/n.

    Returns
    -------
    float : q_hat.
    """
    scores = np.asarray(scores).ravel()
    n = scores.size
    if n == 0:
        raise ValueError('scores vazio')
    level = 1.0 - alpha
    if finite_sample:
        level = min(level * (n + 1) / n, 1.0)
    return float(np.quantile(scores, level, method='higher'))


def mondrian_quantiles(scores_by_stratum: Dict[str, np.ndarray], alpha: float,
                       min_n: int = 1000,
                       fallback_q: float | None = None) -> Dict[str, dict]:
    """q conforme por estrato (Mondrian).

    Parameters
    ----------
    scores_by_stratum : dict {estrato -> np.ndarray de scores de calibração}
    alpha : float
    min_n : int
        Estratos com menos pixels que isto recebem fallback_q (se fornecido) e
        são sinalizados, pois a estimativa do quantil seria instável.
    fallback_q : float or None
        q a usar para estratos com n < min_n (tipicamente o q marginal).

    Returns
    -------
    dict {estrato -> {'q_hat': float, 'n_pixels': int, 'used_fallback': bool}}
    """
    out: Dict[str, dict] = {}
    for s, scores in scores_by_stratum.items():
        n = int(np.asarray(scores).size)
        if n < min_n and fallback_q is not None:
            out[s] = {'q_hat': float(fallback_q), 'n_pixels': n,
                      'used_fallback': True}
            logger.warning(f'Estrato {s!r}: n={n} < min_n={min_n}; '
                           f'usando fallback_q={fallback_q:.6f}')
        else:
            out[s] = {'q_hat': empirical_conformal_quantile(scores, alpha),
                      'n_pixels': n, 'used_fallback': False}
    return out
