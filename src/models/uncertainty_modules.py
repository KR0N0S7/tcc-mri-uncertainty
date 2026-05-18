# Autor: Massanori
# Data: 17/05/2026
# Descrição: Tres modulos de incerteza para o S5 (Grupos A/B/C). Recebe:
#            reconstrucao da E2E-VarNet em (B, 1, H, W). Retorna:
#            - Grupo A (ResM): tensor (B, 1, H, W) com magnitude estimada
#              do residuo, baseline heuristico (Edupuganti et al., 2021).
#            - Grupos B (QR) e C (QR-Lesion): dict {'lower', 'upper'} com
#              dois tensores (B, 1, H, W) representando os quantis alpha/2
#              e 1-alpha/2 da distribuicao condicional p(y|x). Replicacao
#              de Giannakopoulos et al. (2026, secao II.B.2).
#            Grupo C reusa identicamente a arquitetura do Grupo B (alias)
#            — a unica variavel independente entre B e C e a loss aplicada
#            durante o treino, conservando a comparabilidade de capacidade.
#            Protegido por testes em tests/test_uncertainty_modules.py.


"""Modulos de incerteza dos Grupos A/B/C do S5.

Arquitetura segue o artigo base (Giannakopoulos et al., 2026):

Grupo A (ResM):
    1 U-Net (1 -> 1 canal). Output passa por sigmoid e e multiplicado pela
    reconstrucao para mapear ao mesmo range de intensidade. ~7.75M params
    com chans=32, num_pool_layers=4.

Grupos B (QR) e C (QR-Lesion):
    2 U-Nets identicas, uma para o quantil inferior, outra para o superior.
    Cada output: sigmoid(unet(x)) * x, parametrizando o quantil em [0, recon].
    ~15.5M params totais. Mesma arquitetura entre B e C — o que muda e a
    loss aplicada durante o treino (controle experimental).

Os tres modulos compartilham fastmri.models.Unet do pacote oficial do
fastMRI (Zbontar et al., 2018), evitando diferencas sutis de implementacao
que confundiriam a comparacao com o artigo base.

Refs:
    Edupuganti, V. et al. (2021). Uncertainty Quantification in Deep MRI
        Reconstruction. IEEE Trans. Med. Imaging, 40(1):239-250.
    Giannakopoulos, C. et al. (2026). Pixelwise Uncertainty Quantification
        of Accelerated MRI Reconstruction. arXiv:2601.13236.
    Ronneberger, O.; Fischer, P.; Brox, T. (2015). U-Net: Convolutional
        Networks for Biomedical Image Segmentation. MICCAI 2015 (LNCS 9351),
        234-241.
    Zbontar, J. et al. (2018). fastMRI: An Open Dataset and Benchmarks for
        Accelerated MRI. arXiv:1811.08839.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from fastmri.models import Unet


class ResidualMagnitudeModule(nn.Module):
    """Grupo A — modulo ResM (baseline heuristico).

    Aprende a prever a magnitude do residuo de reconstrucao u(x) ~ |x - y|.
    Treinado com MSE (Giannakopoulos et al., 2026, eq. 3). NAO tem garantia
    de cobertura: serve de referencia inferior para demonstrar a superioridade
    dos metodos calibrados (Grupos B e C).

    Parameters
    ----------
    chans : int, default 32
        Numero de canais da primeira camada da U-Net. Dobra a cada pool.
    num_pool_layers : int, default 4
        Profundidade da U-Net (numero de operacoes de pooling).

    Notes
    -----
    Saida no range [0, recon]: o sigmoid limita o coeficiente a [0, 1] e
    a multiplicacao por recon traz a saida ao mesmo range de intensidade,
    facilitando comparacao visual com a hero figure (Cap. 4 do TCC).
    """

    def __init__(self, chans: int = 32, num_pool_layers: int = 4) -> None:
        super().__init__()
        self.unet = Unet(
            in_chans=1,
            out_chans=1,
            chans=chans,
            num_pool_layers=num_pool_layers,
        )

    def forward(self, recon: torch.Tensor) -> torch.Tensor:
        raw = self.unet(recon)
        # sigmoid * recon: scale para a intensidade da reconstrucao
        # (Giannakopoulos et al., 2026, secao II.B.3).
        return torch.sigmoid(raw) * recon


class QuantileRegressionModule(nn.Module):
    """Grupo B — modulo CQR (replicacao do artigo base).

    Duas U-Nets identicas estimam os quantis alpha/2 (lower) e 1-alpha/2 (upper)
    da distribuicao condicional p(y|x), treinadas com pinball loss e calibradas
    via conformal prediction (S5.7) para garantia de cobertura marginal
    (Romano et al., 2019; Giannakopoulos et al., 2026, secao II.B.2).

    Parameters
    ----------
    chans : int, default 32
        Numero de canais da primeira camada de cada U-Net.
    num_pool_layers : int, default 4
        Profundidade de cada U-Net.

    Notes
    -----
    A escolha de duas U-Nets independentes (em vez de uma com 2 canais
    de saida) e do artigo base. Permite cada quantil ter capacidade
    dedicada e simplifica a interpretacao: cada U-Net e um estimador
    quantilico no sentido classico de Koenker (2005).
    """

    def __init__(self, chans: int = 32, num_pool_layers: int = 4) -> None:
        super().__init__()
        self.unet_lower = Unet(
            in_chans=1, out_chans=1,
            chans=chans, num_pool_layers=num_pool_layers,
        )
        self.unet_upper = Unet(
            in_chans=1, out_chans=1,
            chans=chans, num_pool_layers=num_pool_layers,
        )

    def forward(self, recon: torch.Tensor) -> dict:
        lower = torch.sigmoid(self.unet_lower(recon)) * recon
        upper = torch.sigmoid(self.unet_upper(recon)) * recon
        return {'lower': lower, 'upper': upper}


# Grupo C usa identicamente a arquitetura do Grupo B.
# A diferenca entre B e C esta EXCLUSIVAMENTE na loss (qr_loss vs
# qr_lesion_loss). Manter a mesma classe via alias deixa explicito que
# o controle experimental e a arquitetura, e a variavel independente
# e a loss. Qualquer divergencia entre B e C nas metricas finais e,
# por construcao, atribuivel a loss.
QuantileRegressionLesionModule = QuantileRegressionModule
