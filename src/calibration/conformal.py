# Autor: Massanori
# Data: 19/05/2026 (mod 21/05/2026: fallback np.partition para CPU large-tensor)
# Descrição: Calibracao conforme para os Grupos A/B/C do S5.7. Implementa
#            os dois paradigmas relevantes:
#            (1) CQR (Romano et al., 2019) para Grupos B e C (QR e QR-Lesion):
#                score = max(lower - y, y - upper); intervalo calibrado =
#                [lower - q_hat, upper + q_hat].
#            (2) Scaled CP locally adaptive (Lei et al., 2018) para Grupo A
#                (ResM): score = |y - x| / u(x); intervalo calibrado =
#                [x - q_hat * u(x), x + q_hat * u(x)].
#            Garantia formal de cobertura marginal pixelwise
#            P(y_pix in [L, U]) >= 1 - alpha sob exchangeability
#            (Vovk et al., 2005; Angelopoulos & Bates, 2023).
# Mod 21/05/2026: conformal_quantile() ganha fallback robusto com
#                 numpy.partition quando torch.quantile falha por tensor
#                 muito grande (limite ~16M em CPU). Necessario para o S5.7
#                 quando rodando em CPU (cota Kaggle GPU esgotada) com
#                 ~70M pixels no split cal.


"""Calibracao conforme para intervalos de predicao em regressao de imagens.

Dois sabores principais:
    1. CQR (Romano et al., 2019) — para QR (Grupo B) e QR-Lesion (Grupo C):
       inflate aditivo simetrico do intervalo predito por q_hat constante.
    2. Locally-adaptive scaled CP (Lei et al., 2018) — para ResM (Grupo A):
       escala multiplicativa usando a uncertainty predita como largura.

Ambos produzem um unico escalar q_hat a partir do split cal, aplicado
uniformemente em test-time. Isso garante cobertura marginal pixelwise:
    P(y_pixel in [L_pixel, U_pixel]) >= 1 - alpha
sob exchangeability dos exemplos (cal, test).

Memoria: pooling de scores pixelwise de ~46 volumes cal x ~16 slices x
320^2 pixels ~= 75M floats ~= 300 MB. Cabe em RAM CPU; se a memoria de
GPU for limite, move scores para CPU em cada batch.

Robustez do quantile: torch.quantile tem limite interno em CPU para
tensores > ~16M elementos (issue conhecido do PyTorch). Para n maior,
conformal_quantile() faz fallback com numpy.partition, que e O(n) com
selecao parcial e nao tem esse limite. Resultado matematicamente
equivalente (a diferenca para torch.quantile e <= 1 elemento entre vizinhos).

Refs:
    Romano, Y.; Patterson, E.; Candes, E. (2019). Conformalized Quantile
        Regression. NeurIPS, 32:3543-3553. (Teorema 1, eq. 9-10)
    Lei, J. et al. (2018). Distribution-Free Predictive Inference for
        Regression. J. Amer. Statist. Assoc., 113(523):1094-1111.
    Angelopoulos, A.N. et al. (2022). Image-to-Image Regression with
        Distribution-Free Uncertainty Quantification. ICML.
    Angelopoulos, A.N.; Bates, S. (2023). Conformal Prediction:
        A Gentle Introduction. FnT in ML, 16(4):494-591.
    Vovk, V.; Gammerman, A.; Shafer, G. (2005). Algorithmic Learning in
        a Random World. Springer.
    Press, W.H. et al. (2007). Numerical Recipes (3rd ed.), Sec. 8.5:
        Selection (partition-based, O(n)).
"""
from __future__ import annotations

import logging
import math
from typing import Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Nonconformity scores
# ---------------------------------------------------------------------------

def cqr_score(
    lower: torch.Tensor,
    upper: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """CQR nonconformity score: E = max(lower - y, y - upper).

    Sinal:
        - E > 0 quando y esta FORA do intervalo predito:
            * y < lower: E = lower - y (gap inferior)
            * y > upper: E = y - upper (gap superior)
        - E <= 0 quando y esta DENTRO de [lower, upper]:
            * E = -min(y - lower, upper - y) (profundidade negativa no intervalo)

    O quantile (1-alpha)(n+1)/n dos E_i e q_hat. Inflar o intervalo por
    q_hat garante: pelo menos 1-alpha fracao dos pontos test cai dentro.

    Ref: Romano, Patterson & Candes (2019), eq. 9.

    Parameters
    ----------
    lower, upper, target : torch.Tensor
        Mesmo shape, tipicamente (B, 1, H, W). Aceita qualquer shape.

    Returns
    -------
    torch.Tensor
        Mesmo shape, score por pixel.
    """
    if not (lower.shape == upper.shape == target.shape):
        raise ValueError(
            f'Shapes incompativeis: lower {tuple(lower.shape)}, '
            f'upper {tuple(upper.shape)}, target {tuple(target.shape)}'
        )
    return torch.maximum(lower - target, target - upper)


def scaled_cp_score(
    uncertainty: torch.Tensor,
    recon: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Locally adaptive CP score: E = |y - x| / u(x).

    Usado no ResM (Grupo A) que ja preve uma medida de uncertainty u(x).
    O intervalo calibrado em test-time e [x - q_hat * u(x), x + q_hat * u(x)].

    Dividir |y - x| por u(x) normaliza o score — dando intervalos
    localmente adaptativos (mais largos onde u(x) e maior).

    Ref: Lei et al. (2018), Secao 5.

    Parameters
    ----------
    uncertainty : torch.Tensor
        Uncertainty u(x) predita pelo ResM, shape (B, 1, H, W).
    recon : torch.Tensor
        Reconstrucao x, shape (B, 1, H, W).
    target : torch.Tensor
        Ground truth y, shape (B, 1, H, W).
    eps : float, default 1e-6
        Constante pequena para evitar divisao por zero.

    Returns
    -------
    torch.Tensor
        Score por pixel.
    """
    if not (uncertainty.shape == recon.shape == target.shape):
        raise ValueError(
            f'Shapes incompativeis: u {tuple(uncertainty.shape)}, '
            f'x {tuple(recon.shape)}, y {tuple(target.shape)}'
        )
    return (target - recon).abs() / (uncertainty + eps)


# ---------------------------------------------------------------------------
# Quantile com correcao finite-sample e fallback robusto
# ---------------------------------------------------------------------------

def _quantile_via_partition(scores_flat: torch.Tensor, alpha: float) -> float:
    """Fallback O(n) via numpy.partition quando torch.quantile falha.

    Computa o k-esimo menor elemento via algoritmo de selecao
    (introselect, Press et al. 2007), onde k = ceil((n+1)(1-alpha)).
    Matematicamente equivalente ao torch.quantile(scores, q_level)
    com q_level = (1-alpha)(n+1)/n (diferenca de no maximo 1 indice
    em n, ou seja, ~10^-7 para n=70M).

    Necessario para a calibracao do S5.7 em CPU: o split cal tem
    ~70M pixels (730 slices x 320^2), e torch.quantile crasha em
    CPU para n > ~16M elementos (limite interno do PyTorch:
    https://github.com/pytorch/pytorch/issues/64947).

    Parameters
    ----------
    scores_flat : torch.Tensor
        Tensor (possivelmente em GPU); sera movido para CPU para a
        operacao numpy. Aceita qualquer shape (sera flattened).
    alpha : float
        Nivel de miscoverage, no intervalo aberto (0, 1).

    Returns
    -------
    float
        q_hat escalar.
    """
    s = scores_flat.detach().reshape(-1).to(torch.float32).cpu().numpy()
    n = int(s.size)
    if n == 0:
        raise ValueError('Tensor de scores vazio')
    k = int(math.ceil((n + 1) * (1.0 - alpha)))
    k = min(max(k, 1), n)  # clip a [1, n]
    return float(np.partition(s, k - 1)[k - 1])


def conformal_quantile(scores: torch.Tensor, alpha: float = 0.10) -> float:
    """Quantile empirico (1-alpha)(n+1)/n com correcao finite-sample.

    Romano et al. (2019, Teorema 1): o fator (1+1/n) e necessario para a
    garantia de cobertura formal P(y in C_hat) >= 1 - alpha sob
    exchangeability. Para n grande, q_level ~= 1 - alpha.

    Robustez: torch.quantile tem limite interno de ~16M elementos em CPU.
    Para tensores maiores (e.g. 70M pixels do split cal), automatically
    fallbacks para numpy.partition (O(n), sem o limite interno).

    Parameters
    ----------
    scores : torch.Tensor
        Tensor flat ou multi-dim de scores de inconformidade.
    alpha : float, default 0.10
        Nivel de miscoverage. Cobertura garantida = 1 - alpha.

    Returns
    -------
    float
        Threshold q_hat (scalar).
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f'alpha deve estar em (0, 1), recebido {alpha}')

    scores_flat = scores.flatten()
    n = scores_flat.numel()
    if n == 0:
        raise ValueError('Tensor de scores vazio')

    # Clip a 1.0 caso (1-alpha)(n+1)/n > 1 para n muito pequeno
    q_level = min((1.0 - alpha) * (n + 1) / n, 1.0)

    try:
        return torch.quantile(scores_flat, q_level).item()
    except RuntimeError as e:
        # torch.quantile falha em CPU para n > ~16M com mensagem que contem
        # 'too large' (mensagem exata varia entre versoes do PyTorch).
        # Fallback equivalente via partition.
        if 'too large' in str(e).lower():
            logger.warning(
                f'torch.quantile falhou com n={n:,} (limite ~16M em CPU); '
                f'usando fallback numpy.partition.'
            )
            return _quantile_via_partition(scores_flat, alpha)
        # Outros RuntimeErrors propagam normalmente
        raise


# ---------------------------------------------------------------------------
# Calibracao end-to-end sobre um DataLoader
# ---------------------------------------------------------------------------

@torch.no_grad()
def calibrate_qr(
    module: torch.nn.Module,
    cal_loader: DataLoader,
    alpha: float = 0.10,
    device: Union[str, torch.device] = 'cuda',
) -> dict:
    """Calibra um modulo QR (ou QR-Lesion) via CQR.

    Itera o split cal, computa scores pixelwise via cqr_score, agrupa em
    um unico tensor (em CPU para evitar OOM em GPU) e retorna
    q_hat = quantile (1-alpha)(n+1)/n empirico.

    Intervalo em test-time: [lower(x) - q_hat, upper(x) + q_hat].

    Parameters
    ----------
    module : nn.Module
        QuantileRegressionModule ou QuantileRegressionLesionModule (alias).
    cal_loader : DataLoader
        Sobre o split cal (tipicamente slice-wise, batch=1).
    alpha : float, default 0.10
        Nivel de miscoverage.
    device : str or torch.device, default 'cuda'

    Returns
    -------
    dict
        Chaves: 'q_hat' (float), 'n_pixels' (int), 'n_batches' (int),
        'alpha' (float), 'mean_score' (float), 'method' ('CQR').
    """
    module = module.to(device).eval()
    all_scores = []
    n_batches = 0

    for batch in cal_loader:
        recon = batch['recon'].to(device, non_blocking=True)
        target = batch['target'].to(device, non_blocking=True)
        pred = module(recon)
        scores = cqr_score(pred['lower'], pred['upper'], target)
        # Move para CPU imediatamente para liberar memoria de GPU
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
        'method': 'CQR',
    }
    logger.info(
        f'CQR calibration: n_batches={n_batches}, '
        f'n_pixels={result["n_pixels"]:,}, alpha={alpha}, '
        f'q_hat={q_hat:.6f}, mean_score={result["mean_score"]:.6f}'
    )
    return result


@torch.no_grad()
def calibrate_resm(
    module: torch.nn.Module,
    cal_loader: DataLoader,
    alpha: float = 0.10,
    device: Union[str, torch.device] = 'cuda',
    eps: float = 1e-6,
) -> dict:
    """Calibra um modulo ResM via locally-adaptive scaled CP.

    Itera o split cal, computa scores pixelwise via scaled_cp_score,
    e retorna q_hat.

    Intervalo em test-time: [x - q_hat * u(x), x + q_hat * u(x)].

    Parameters
    ----------
    module : nn.Module
        ResidualMagnitudeModule.
    cal_loader, alpha, device : mesmo padrao de calibrate_qr.
    eps : float
        Para estabilidade numerica na divisao do score.

    Returns
    -------
    dict
        Chaves: 'q_hat', 'n_pixels', 'n_batches', 'alpha', 'mean_score',
        'method' ('ScaledCP').
    """
    module = module.to(device).eval()
    all_scores = []
    n_batches = 0

    for batch in cal_loader:
        recon = batch['recon'].to(device, non_blocking=True)
        target = batch['target'].to(device, non_blocking=True)
        uncertainty = module(recon)
        scores = scaled_cp_score(uncertainty, recon, target, eps=eps)
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
        'method': 'ScaledCP',
    }
    logger.info(
        f'Scaled CP calibration: n_batches={n_batches}, '
        f'n_pixels={result["n_pixels"]:,}, alpha={alpha}, '
        f'q_hat={q_hat:.6f}, mean_score={result["mean_score"]:.6f}'
    )
    return result


# ---------------------------------------------------------------------------
# Aplicacao do intervalo calibrado (test-time)
# ---------------------------------------------------------------------------

def apply_cqr_interval(
    lower: torch.Tensor,
    upper: torch.Tensor,
    q_hat: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Intervalo CQR calibrado: [lower - q_hat, upper + q_hat].

    Parameters
    ----------
    lower, upper : torch.Tensor
        Saidas do modulo QR.
    q_hat : float
        Saida de calibrate_qr.

    Returns
    -------
    tuple
        (lower_cal, upper_cal), mesmo shape de lower/upper.
    """
    return lower - q_hat, upper + q_hat


def apply_resm_interval(
    recon: torch.Tensor,
    uncertainty: torch.Tensor,
    q_hat: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Intervalo ResM calibrado: [x - q_hat * u, x + q_hat * u].

    Parameters
    ----------
    recon : torch.Tensor
        Reconstrucao x.
    uncertainty : torch.Tensor
        Uncertainty u(x) predita pelo ResM.
    q_hat : float
        Saida de calibrate_resm.

    Returns
    -------
    tuple
        (lower, upper), mesmo shape de recon/uncertainty.
    """
    return recon - q_hat * uncertainty, recon + q_hat * uncertainty
