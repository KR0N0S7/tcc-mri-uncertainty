# Autor: Massanori
# Data: 17/05/2026
# Descrição: Tres funcoes de loss com interface unificada (D3) para os Grupos
#            A/B/C do S5. Assinatura compartilhada:
#              loss(uncertainty_pred, recon, target, lesion_mask, **kwargs).
#            resm_loss (Grupo A): MSE entre predicao e |recon - target|,
#            baseline de Edupuganti et al. (2021) e Giannakopoulos et al.
#            (2026, eq. 3). qr_loss (Grupo B): pinball nos quantis alpha/2
#            e 1-alpha/2, replicacao de Giannakopoulos et al. (2026, eq. 2).
#            qr_lesion_loss (Grupo C, contribuicao original): pinball
#            ponderada por lambda nas regioes de lesao, generalizando
#            region-specific loss de Yeung et al. (2022) para CQR em imagem
#            medica. lesion_mask ignorada em A e B.


"""Loss functions com interface unificada para os Grupos A/B/C do S5.

Cada grupo tem sua propria loss, mas a assinatura e identica para permitir
trocar de modulo no loop de treino sem branches no codigo:

    loss = loss_fn(uncertainty_pred, recon, target, lesion_mask, **kwargs)

Onde uncertainty_pred e:
    - Grupo A (ResM): tensor (B, 1, H, W)
    - Grupos B (QR) e C (QR-Lesion): dict {'lower': tensor, 'upper': tensor}

Refs:
    Edupuganti, V. et al. (2021). Uncertainty Quantification in Deep MRI
        Reconstruction. IEEE Trans. Med. Imaging, 40(1):239-250.
    Giannakopoulos, C. et al. (2026). Pixelwise Uncertainty Quantification
        of Accelerated MRI Reconstruction. arXiv:2601.13236. (Eqs. 2, 3)
    Romano, Y.; Patterson, E.; Candes, E. (2019). Conformalized Quantile
        Regression. NeurIPS 32, 3543-3553.
    Yeung, M. et al. (2022). Unified Focal Loss: Generalising Dice and
        Cross Entropy-based Losses to Handle Class Imbalance in Medical
        Image Segmentation. Comput. Med. Imaging Graph., 95:102026.
    Isensee, F. et al. (2021). nnU-Net: A Self-Configuring Method for
        Deep Learning-based Biomedical Image Segmentation. Nat. Methods,
        18(2):203-211.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from src.losses.pinball import pinball_per_pixel

# Nivel padrao do quantile interval: alpha = 0.10 implica cobertura nominal 90%.
# Replicacao de Giannakopoulos et al. (2026, secao IV.A).
DEFAULT_ALPHA = 0.10


def resm_loss(
    uncertainty_pred: torch.Tensor,
    recon: torch.Tensor,
    target: torch.Tensor,
    lesion_mask: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    """Loss MSE do Grupo A (ResM) — baseline heuristico.

    L_ResM = MSE(u(x), |x - y|)  (Giannakopoulos et al., 2026, eq. 3).

    O modulo aprende a prever a magnitude do residuo de reconstrucao
    pixel a pixel. NAO tem garantia de cobertura — por isso e tratado
    como baseline pre-conformal (Edupuganti et al., 2021).

    Parameters
    ----------
    uncertainty_pred : torch.Tensor
        Saida do modulo ResM, shape (B, 1, H, W).
    recon : torch.Tensor
        Reconstrucao VarNet, shape (B, 1, H, W).
    target : torch.Tensor
        Ground truth, shape (B, 1, H, W).
    lesion_mask : torch.Tensor or None
        Ignorada neste grupo (interface unificada, D3).

    Returns
    -------
    torch.Tensor
        Loss escalar.
    """
    residual = torch.abs(recon - target)
    return F.mse_loss(uncertainty_pred, residual)


def qr_loss(
    uncertainty_pred: dict,
    recon: torch.Tensor,
    target: torch.Tensor,
    lesion_mask: Optional[torch.Tensor] = None,
    alpha: float = DEFAULT_ALPHA,
    **kwargs,
) -> torch.Tensor:
    """Loss CQR do Grupo B — pinball nos quantis alpha/2 e 1-alpha/2.

    L_QR = E[ L_{alpha/2}(lower, y) + L_{1-alpha/2}(upper, y) ]
    (Giannakopoulos et al., 2026, eq. 2).

    Treina os dois U-Nets do modulo QR. A calibracao conforme (S5.7)
    fecha a garantia de cobertura marginal (Romano et al., 2019).

    Parameters
    ----------
    uncertainty_pred : dict
        Dict com chaves 'lower' e 'upper', cada uma com shape (B, 1, H, W).
    recon : torch.Tensor
        Reconstrucao VarNet, shape (B, 1, H, W). Nao usado diretamente,
        mantido na assinatura para interface unificada.
    target : torch.Tensor
        Ground truth, shape (B, 1, H, W).
    lesion_mask : torch.Tensor or None
        Ignorada neste grupo (interface unificada, D3).
    alpha : float, default DEFAULT_ALPHA
        Nivel do intervalo. Cobertura nominal = 1 - alpha.

    Returns
    -------
    torch.Tensor
        Loss escalar.
    """
    lower = uncertainty_pred['lower']
    upper = uncertainty_pred['upper']
    loss_lower = pinball_per_pixel(lower, target, alpha / 2.0).mean()
    loss_upper = pinball_per_pixel(upper, target, 1.0 - alpha / 2.0).mean()
    return loss_lower + loss_upper


def qr_lesion_loss(
    uncertainty_pred: dict,
    recon: torch.Tensor,
    target: torch.Tensor,
    lesion_mask: torch.Tensor,
    alpha: float = DEFAULT_ALPHA,
    lambda_lesion: float = 5.0,
    **kwargs,
) -> torch.Tensor:
    """Loss CQR ponderada do Grupo C — contribuicao original do TCC.

    L_QR-Lesion = E_pixel[ w(pixel) * pinball_pixel ]

    onde pinball_pixel = pinball_{alpha/2}(lower) + pinball_{1-alpha/2}(upper)
    e w(pixel) = 1.0 fora da lesao, lambda dentro.

    Equivalentemente: w = 1 + (lambda - 1) * M, com M em {0, 1}.

    A motivacao clinica e que duvidar igualmente em todo o cerebro nao
    serve para diagnostico: o que importa e calibrar a incerteza onde
    estao as lesoes. Generaliza a ideia de region-specific loss (Yeung
    et al., 2022; Isensee et al., 2021) para o contexto de quantile
    regression em imagem medica acelerada.

    Por construcao, lambda=1 recupera qr_loss exatamente — protegido por
    teste em test_uncertainty_losses.py. Isso garante que C nao pode
    estar pior que B no limite, isolando o efeito da loss.

    Parameters
    ----------
    uncertainty_pred : dict
        Dict com chaves 'lower' e 'upper', cada uma com shape (B, 1, H, W).
    recon : torch.Tensor
        Reconstrucao VarNet. Nao usado diretamente.
    target : torch.Tensor
        Ground truth, shape (B, 1, H, W).
    lesion_mask : torch.Tensor
        Mascara binaria de lesao, shape (B, 1, H, W). Obrigatoria no Grupo C.
    alpha : float, default DEFAULT_ALPHA
        Nivel do intervalo.
    lambda_lesion : float, default 5.0
        Fator de ponderacao das regioes de lesao. lambda=1 reduz a loss
        ao Grupo B. MVP usa lambda=5.0.

    Returns
    -------
    torch.Tensor
        Loss escalar.
    """
    lower = uncertainty_pred['lower']
    upper = uncertainty_pred['upper']

    pinball_low = pinball_per_pixel(lower, target, alpha / 2.0)
    pinball_up = pinball_per_pixel(upper, target, 1.0 - alpha / 2.0)
    pixelwise = pinball_low + pinball_up  # (B, 1, H, W)

    # Pesos: 1.0 fora da lesao, lambda dentro. lambda=1 -> weight=1 em
    # todo lugar -> recupera qr_loss exatamente (teste no test suite).
    weight = 1.0 + (lambda_lesion - 1.0) * lesion_mask

    return (weight * pixelwise).mean()
