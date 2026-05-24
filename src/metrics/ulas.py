# Autor: Massanori
# Data: 21/05/2026
# Descricao: ULAS — Uncertainty-Lesion Alignment Score (contribuicao original
#            do TCC). Mede alinhamento DIRECIONAL entre gradientes da
#            uncertainty predita e gradientes do erro ground truth, restrito
#            a regioes de lesao. Complementa o Pearson global (que mede so
#            correlacao de magnitudes) capturando se a uncertainty 'flui'
#            na mesma direcao que o erro. Recebe: uncertainty (H,W) ou
#            (B,1,H,W), error mesma shape, lesion_mask binaria mesma shape.
#            Retorna: float em [0, 1] (ULAS) ou tupla (mean_real, mean_null)
#            para versao com baseline. Necessario para o S5.8 / 4a entrega.


"""ULAS — Uncertainty-Lesion Alignment Score (contribuicao original do TCC).

Definicao
---------
.. math::

    \\mathrm{ULAS}(u, e, M) = \\frac{1}{|M|}\\sum_{i \\in M}
        \\left| \\cos\\left(\\nabla u_i, \\nabla e_i\\right) \\right|

onde
    u   = uncertainty predita pixelwise
    e   = error map = |y - x| ground truth
    M   = mascara de lesao binaria
    nabla = gradiente espacial via Sobel 3x3 isotropico
    cos(a, b) = (a . b) / (||a|| ||b||)  para vetores 2D
    |.|  = valor absoluto (alinhamento direcional, sinal nao importa)

Range [0, 1]:
    * 0 = gradientes ortogonais (sem alinhamento)
    * 1 = gradientes paralelos ou antiparalelos (alinhamento perfeito)

O que ULAS captura que Pearson nao
----------------------------------
Pearson global mede correlacao escalar de magnitudes pixelwise. ULAS mede
se a uncertainty FLUI na mesma direcao que o erro — ou seja, se onde
o erro aumenta, a uncertainty aumenta tambem, com a forma certa
localmente. Pixel-level direction vs scalar-level magnitude.

Null baseline
-------------
ULAS(u, pi(e), M) onde pi e permutacao aleatoria dos pixels do error.
Para gradientes 2D iid uniformes no circulo, E[|cos|] = 2/pi ~ 0.637.
Em imagens reais com correlacao espacial, esperado em [0.5, 0.7].
Se ULAS_real >> ULAS_null, a metrica e informativa.

Validacao sintetica (em tests/test_ulas.py)
-------------------------------------------
* u, e ambos radiais (paraboloide e cone) → ULAS proximo de 1.0.
* u radial, e ortogonal-radial → ULAS proximo de 0.0.
* u e e aleatorios → ULAS proximo do null baseline.

Refs:
    Sobel, I.; Feldman, G. (1968). A 3x3 isotropic gradient operator
        for image processing. Stanford AI Project (talk).
    Adler, J.; Oktem, O. (2018). Deep Bayesian Inversion. arXiv:1811.05910.
    Bishop (2006), PRML, Sec. 1.6 (entropia direcional como medida
        de informacao mutua).
    Cover & Thomas (2006), Elements of Information Theory.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Sobel kernels 3x3 isotropicos, normalizados (somam |1| em cada direcao).
_SOBEL_X = torch.tensor(
    [[-1.0, 0.0, 1.0],
     [-2.0, 0.0, 2.0],
     [-1.0, 0.0, 1.0]]
) / 8.0

_SOBEL_Y = torch.tensor(
    [[-1.0, -2.0, -1.0],
     [ 0.0,  0.0,  0.0],
     [ 1.0,  2.0,  1.0]]
) / 8.0


def sobel_gradient(
    f: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calcula gradiente espacial 2D via Sobel.

    Parameters
    ----------
    f : torch.Tensor
        Shape (H, W), (1, H, W), ou (B, 1, H, W). Float.

    Returns
    -------
    (gx, gy) : tuple of torch.Tensor
        Mesma shape de f. gx = derivada parcial em x (horizontal),
        gy = derivada parcial em y (vertical).

    Notes
    -----
    Usa padding='same' (replicate) para manter shape. Os Sobel kernels
    sao normalizados por 1/8 para que magnitude do gradiente em uma
    rampa linear de inclinacao 1 retorne 1.0.
    """
    original_ndim = f.ndim
    if f.ndim == 2:
        x = f.unsqueeze(0).unsqueeze(0)
    elif f.ndim == 3:
        # (1, H, W) ou (C, H, W) — assume 1 canal
        if f.shape[0] != 1:
            raise ValueError(
                f'Esperado shape (1, H, W) para input 3D, recebido {tuple(f.shape)}'
            )
        x = f.unsqueeze(0)
    elif f.ndim == 4:
        if f.shape[1] != 1:
            raise ValueError(
                f'Esperado 1 canal em (B, 1, H, W), recebido C={f.shape[1]}'
            )
        x = f
    else:
        raise ValueError(f'f.ndim deve ser 2, 3 ou 4, recebido {f.ndim}')

    kx = _SOBEL_X.to(device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    ky = _SOBEL_Y.to(device=x.device, dtype=x.dtype).view(1, 1, 3, 3)

    # Reflect padding evita artefatos de borda mais que zero-padding.
    x_padded = F.pad(x, (1, 1, 1, 1), mode='replicate')
    gx = F.conv2d(x_padded, kx)
    gy = F.conv2d(x_padded, ky)

    if original_ndim == 2:
        gx = gx.squeeze(0).squeeze(0)
        gy = gy.squeeze(0).squeeze(0)
    elif original_ndim == 3:
        gx = gx.squeeze(0)
        gy = gy.squeeze(0)
    return gx, gy


def ulas(
    uncertainty: torch.Tensor,
    error: torch.Tensor,
    lesion_mask: torch.Tensor,
    eps: float = 1e-8,
) -> float:
    """Uncertainty-Lesion Alignment Score (ver docstring do modulo).

    Parameters
    ----------
    uncertainty : torch.Tensor
        Uncertainty predita, shape (H, W) ou (B, 1, H, W).
    error : torch.Tensor
        Error map |y - x|, mesma shape de uncertainty.
    lesion_mask : torch.Tensor
        Mascara binaria (ou em [0, 1]; sera binarizada em > 0.5),
        mesma shape de uncertainty.
    eps : float, default 1e-8
        Estabilidade na divisao do coseno.

    Returns
    -------
    float
        ULAS em [0, 1]. Retorna 0.0 se a mascara nao tem pixels positivos
        (NaN-safe; o caller pode interpretar como 'sem lesao no slice').
    """
    if uncertainty.shape != error.shape:
        raise ValueError(
            f'Shapes incompativeis: u {tuple(uncertainty.shape)}, '
            f'e {tuple(error.shape)}'
        )
    if uncertainty.shape != lesion_mask.shape:
        raise ValueError(
            f'Shapes incompativeis: u {tuple(uncertainty.shape)}, '
            f'mask {tuple(lesion_mask.shape)}'
        )

    gx_u, gy_u = sobel_gradient(uncertainty)
    gx_e, gy_e = sobel_gradient(error)

    # Coseno entre vetores 2D (gx, gy)_u e (gx, gy)_e
    dot = gx_u * gx_e + gy_u * gy_e
    norm_u = torch.sqrt(gx_u ** 2 + gy_u ** 2 + eps)
    norm_e = torch.sqrt(gx_e ** 2 + gy_e ** 2 + eps)
    cos = dot / (norm_u * norm_e)
    cos_abs = cos.abs()

    mask_bool = lesion_mask > 0.5
    n_lesion = int(mask_bool.sum().item())
    if n_lesion == 0:
        return 0.0

    return cos_abs[mask_bool].mean().item()


def ulas_with_null(
    uncertainty: torch.Tensor,
    error: torch.Tensor,
    lesion_mask: torch.Tensor,
    n_permutations: int = 10,
    seed: int = 42,
    eps: float = 1e-8,
) -> dict:
    """Computa ULAS real + N permutacoes do error_map como null baseline.

    A permutacao destroi a estrutura espacial do error preservando a
    distribuicao marginal de magnitudes. Se ULAS_real >> media do null,
    o alinhamento direcional e estatisticamente informativo (Demsar, 2006,
    secao 2 sobre testes empiricos de significancia).

    Parameters
    ----------
    uncertainty, error, lesion_mask, eps : ver ulas().
    n_permutations : int, default 10
        Quantas permutacoes do error gerar para o null baseline.
        10 e suficiente para uma estimativa de media; aumente para
        intervalos de confianca empiricos.
    seed : int, default 42
        Para reprodutibilidade.

    Returns
    -------
    dict com:
        - 'ulas': float, score real.
        - 'null_mean': float, media do baseline.
        - 'null_std': float, desvio padrao do baseline.
        - 'null_scores': list of float, N valores individuais.
        - 'z_score': float, (ulas - null_mean) / null_std se std > 0.
        - 'n_lesion_pixels': int.
    """
    real = ulas(uncertainty, error, lesion_mask, eps=eps)

    g = torch.Generator(device=error.device).manual_seed(seed)
    n_total = error.numel()
    null_scores = []
    for _ in range(n_permutations):
        perm = torch.randperm(n_total, generator=g, device=error.device)
        error_perm = error.flatten()[perm].view(error.shape)
        null_scores.append(ulas(uncertainty, error_perm, lesion_mask, eps=eps))

    null_tensor = torch.tensor(null_scores)
    null_mean = null_tensor.mean().item()
    null_std = null_tensor.std(unbiased=True).item() if n_permutations > 1 else 0.0

    z = (real - null_mean) / null_std if null_std > 1e-6 else float('nan')

    n_lesion = int((lesion_mask > 0.5).sum().item())

    return {
        'ulas': real,
        'null_mean': null_mean,
        'null_std': null_std,
        'null_scores': null_scores,
        'z_score': z,
        'n_lesion_pixels': n_lesion,
    }
