# Autor: Massanori
# Data: 02/06/2026
# Descrição: CQR localmente adaptativo (normalized / locally-weighted CQR),
#            destino: src/calibration/adaptive_cqr.py. Estende o CQR aditivo
#            do Grupo B/C com um score normalizado pela largura predita do
#            intervalo, w(x) = upper(x) - lower(x), exatamente o resíduo
#            localmente ponderado de Lei et al. (2018, Sec. 5.2) aplicado ao
#            score CQR de Romano et al. (2019). Objetivo: isolar se a
#            superioridade do ResM em Coverage_lesion (achado S5.4.3) vem da
#            ADAPTATIVIDADE LOCAL da calibracao (multiplicativa) ou da
#            arquitetura. Se o CQR-normalizado, treinado igual ao B/C, recupera
#            a cobertura de lesão do ResM, então o mecanismo causal é a
#            calibracao, não o modelo.
#            Recebe: saídas lower(x)/upper(x) de um QuantileRegressionModule já
#            treinado (B ou C) e o split cal. Gera: q_hat escalar para o
#            intervalo calibrado [lower - q*w, upper + q*w].
#            A garantia de cobertura marginal pixelwise é preservada: w(x) é
#            função apenas de x (saída do modelo, ajustada no treino), logo o
#            score normalizado permanece exchangeable entre cal e test
#            (Lei et al., 2018; Vovk et al., 2005; Angelopoulos & Bates, 2023).

"""CQR localmente adaptativo (normalized / locally-weighted CQR).

Motivação
---------
No S5, o Grupo A (ResM, scaled CP locally-adaptive) superou os Grupos B/C
(CQR marginal aditivo) em Coverage_lesion por ~8 pp. A explicação estrutural
proposta foi a adaptatividade local: o intervalo do ResM, [x - q*u(x),
x + q*u(x)], dilata-se automaticamente onde u(x) e' maior (i.e. em lesoes),
enquanto o CQR usa um q constante (inflate aditivo). Este modulo testa essa
hipotese SEM trocar de arquitetura: aplica o mesmo principio locally-weighted
de Lei et al. (2018, Sec. 5.2) ao score CQR.

Definicoes
----------
Score CQR aditivo (Romano et al., 2019, eq. 9):
    E(x, y) = max(lower(x) - y, y - upper(x))
    intervalo calibrado: [lower(x) - q, upper(x) + q]

Score CQR normalizado (este modulo; Lei et al. 2018, Sec. 5.2):
    w(x) = max(upper(x) - lower(x), eps)        # escala local = largura predita
    E~(x, y) = E(x, y) / w(x)
    intervalo calibrado: [lower(x) - q*w(x), upper(x) + q*w(x)]

Equivalencia cobertura <-> score (porque w(x) > 0):
    y in [lower - q*w, upper + q*w]
      <=> (lower - y)/w <= q  e  (y - upper)/w <= q
      <=> max(lower - y, y - upper)/w <= q
      <=> E~(x, y) <= q
Logo q_hat = quantile (1-alpha)(n+1)/n empirico de {E~_i} garante
P(y_pix in [L, U]) >= 1 - alpha sob exchangeability (cal, test).

Largura do intervalo resultante: (upper - lower) + 2*q*w = w*(1 + 2q),
i.e. proporcional a w(x) — adaptativa por pixel, como no ResM.

Refs
----
    Lei, J. et al. (2018). Distribution-Free Predictive Inference for
        Regression. J. Amer. Statist. Assoc., 113(523):1094-1111. (Sec. 5.2,
        resíduo localmente ponderado R~ = |y - mu(x)| / sigma(x))
    Romano, Y.; Patterson, E.; Candes, E. (2019). Conformalized Quantile
        Regression. NeurIPS, 32:3543-3553. (score CQR, eq. 9; Teorema 1)
    Angelopoulos, A.N.; Bates, S. (2023). Conformal Prediction: A Gentle
        Introduction. FnT in ML, 16(4):494-591. (scores normalizados)
    Vovk, V.; Gammerman, A.; Shafer, G. (2005). Algorithmic Learning in a
        Random World. Springer.
"""
from __future__ import annotations

import logging
from typing import Tuple, Union

import torch
from torch.utils.data import DataLoader

from src.calibration.conformal import cqr_score, conformal_quantile

logger = logging.getLogger(__name__)

DEFAULT_EPS = 1e-6


def cqr_normalized_score(
    lower: torch.Tensor,
    upper: torch.Tensor,
    target: torch.Tensor,
    eps: float = DEFAULT_EPS,
) -> torch.Tensor:
    """Score CQR normalizado: E~ = max(lower - y, y - upper) / w(x).

    w(x) = max(upper - lower, eps) e' a largura predita do intervalo,
    usada como escala local (Lei et al., 2018, Sec. 5.2). Como w > 0,
    a normalizacao preserva a ordenacao do max e portanto a equivalencia
    cobertura <-> (score <= q).

    Parameters
    ----------
    lower, upper, target : torch.Tensor
        Mesmo shape (tipicamente (B, 1, H, W)).
    eps : float
        Piso para w(x), evita divisao por zero onde o modelo preve
        largura ~0.

    Returns
    -------
    torch.Tensor
        Score normalizado por pixel, mesmo shape da entrada.
    """
    if not (lower.shape == upper.shape == target.shape):
        raise ValueError(
            f'Shapes incompativeis: lower {tuple(lower.shape)}, '
            f'upper {tuple(upper.shape)}, target {tuple(target.shape)}'
        )
    width = (upper - lower).clamp_min(eps)
    return cqr_score(lower, upper, target) / width


def apply_cqr_normalized_interval(
    lower: torch.Tensor,
    upper: torch.Tensor,
    q_hat: float,
    eps: float = DEFAULT_EPS,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Intervalo CQR normalizado calibrado: [lower - q*w, upper + q*w].

    w(x) = max(upper - lower, eps). Espelha apply_cqr_interval do
    conformal.py, mas com inflate MULTIPLICATIVO por w(x) (locally-adaptive).

    Parameters
    ----------
    lower, upper : torch.Tensor
        Saidas do modulo QR.
    q_hat : float
        Saida de calibrate_qr_normalized.
    eps : float
        Piso para w(x); DEVE ser o mesmo usado na calibracao.

    Returns
    -------
    tuple
        (lower_cal, upper_cal), mesmo shape de lower/upper.
    """
    width = (upper - lower).clamp_min(eps)
    return lower - q_hat * width, upper + q_hat * width


@torch.no_grad()
def calibrate_qr_normalized(
    module: torch.nn.Module,
    cal_loader: DataLoader,
    alpha: float = 0.10,
    device: Union[str, torch.device] = 'cuda',
    eps: float = DEFAULT_EPS,
) -> dict:
    """Calibra um modulo QR (B ou C) via CQR localmente adaptativo.

    Espelha calibrate_qr do conformal.py, trocando o score por
    cqr_normalized_score. Itera o split cal, agrupa scores pixelwise em CPU
    (evita OOM em GPU) e retorna q_hat = quantile (1-alpha)(n+1)/n empirico.

    Intervalo em test-time: [lower - q_hat*w(x), upper + q_hat*w(x)].

    Parameters
    ----------
    module : nn.Module
        QuantileRegressionModule (ou alias QuantileRegressionLesionModule).
    cal_loader : DataLoader
        Sobre o split cal (slice-wise, batch=1).
    alpha : float, default 0.10
        Nivel de miscoverage.
    device : str or torch.device, default 'cuda'.
    eps : float
        Piso para w(x). DEVE coincidir com apply_cqr_normalized_interval.

    Returns
    -------
    dict
        Chaves: 'q_hat' (float), 'n_pixels' (int), 'n_batches' (int),
        'alpha' (float), 'mean_score' (float), 'eps' (float),
        'method' ('CQR-Norm').
    """
    module = module.to(device).eval()
    all_scores = []
    n_batches = 0

    for batch in cal_loader:
        recon = batch['recon'].to(device, non_blocking=True)
        target = batch['target'].to(device, non_blocking=True)
        pred = module(recon)
        scores = cqr_normalized_score(pred['lower'], pred['upper'], target, eps=eps)
        all_scores.append(scores.cpu().flatten())
        n_batches += 1

    if not all_scores:
        raise ValueError('cal_loader nao produziu batches')

    all_scores_t = torch.cat(all_scores)
    q_hat = conformal_quantile(all_scores_t, alpha)

    result = {
        'q_hat': q_hat,
        'n_pixels': int(all_scores_t.numel()),
        'n_batches': n_batches,
        'alpha': alpha,
        'mean_score': all_scores_t.mean().item(),
        'eps': eps,
        'method': 'CQR-Norm',
    }
    logger.info(
        f'CQR-Norm calibration: n_batches={n_batches}, '
        f'n_pixels={result["n_pixels"]:,}, alpha={alpha}, '
        f'q_hat={q_hat:.6f}, mean_score={result["mean_score"]:.6f}'
    )
    return result
