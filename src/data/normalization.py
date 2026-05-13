# Autor: Massanori
# Data: 13/05/2026
# Descrição: Estratégia de normalização max-volume para reconstruções RSS do
#            fastMRI (Giannakopoulos et al., 2026; Sriram et al., 2020). Recebe:
#            tensor de reconstrução para computar max_val, ou tensor + max_val
#            previamente computado. Retorna: tensor normalizado (divisão pelo
#            max do volume inteiro, não por fatia) ou desnormalizado (multiplicação).
#            Mantém consistência inter-fatias do mesmo paciente, eliminando
#            variabilidade entre scanners e coils. Protegido por 5 testes
#            unitários em tests/test_normalization.py.


"""
Estrategia de normalizacao: max-norm por volume.
Refs: Giannakopoulos et al. (2026), Sriram et al. (2020).
"""
from __future__ import annotations
import torch


def compute_volume_max(reconstruction_rss: torch.Tensor) -> float:
    """
    Calcula a constante de normalizacao para um volume inteiro.

    Parameters
    ----------
    reconstruction_rss : torch.Tensor
        Tensor (n_slices, H, W) ou (H, W) da reconstrucao RSS.

    Returns
    -------
    float
        max(abs(reconstruction_rss)) sobre todas as dimensoes.
    """
    return float(reconstruction_rss.abs().max().item())


def normalize(tensor: torch.Tensor, max_val: float) -> torch.Tensor:
    """
    Aplica normalizacao max-volume: x / max_val.

    Parameters
    ----------
    tensor : torch.Tensor
        Reconstrucao ou alvo a normalizar.
    max_val : float
        Constante computada via compute_volume_max para o MESMO volume.

    Returns
    -------
    torch.Tensor
        Tensor normalizado com valores tipicamente em [0, 1].
    """
    if max_val <= 0:
        raise ValueError(f'max_val deve ser positivo, recebido {max_val}')
    return tensor / max_val


def denormalize(tensor: torch.Tensor, max_val: float) -> torch.Tensor:
    """Inverte a normalizacao. Util para visualizacoes em escala original."""
    return tensor * max_val