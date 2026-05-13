# Autor: Massanori
# Data: 13/05/2026
# Descrição: Testes unitários para src/data/normalization.py. 5 testes que
#            blindam a estratégia max-volume: range [0, 1] pós-normalização,
#            denormalize é exatamente inverso de normalize, ValueError em
#            max_val <= 0, max é computado sobre todas as fatias (não por
#            fatia), preservação de proporções internas após normalização.
#            Recebe: tensores sintéticos. Retorna: assertions pass/fail.
#            Roda com: python -m pytest tests/test_normalization.py -v


import pytest
import torch

from src.data.normalization import compute_volume_max, normalize, denormalize


def test_max_norm_traz_valores_para_0_1():
    vol = torch.tensor([[[0.0, 50.0], [100.0, 200.0]]])  # max=200
    m = compute_volume_max(vol)
    assert m == 200.0
    norm = normalize(vol, m)
    assert norm.max().item() == 1.0
    assert norm.min().item() == 0.0


def test_denormalize_eh_inverso():
    vol = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    m = compute_volume_max(vol)
    assert torch.allclose(denormalize(normalize(vol, m), m), vol)


def test_normalize_falha_se_max_zero_ou_negativo():
    with pytest.raises(ValueError):
        normalize(torch.zeros(2, 2), 0.0)
    with pytest.raises(ValueError):
        normalize(torch.zeros(2, 2), -1.0)


def test_max_volume_sobre_todas_as_fatias():
    """Max deve ser sobre todo o volume, nao por fatia."""
    vol = torch.zeros(5, 10, 10)
    vol[2, 5, 5] = 100.0  # spike numa fatia
    assert compute_volume_max(vol) == 100.0


def test_normalizacao_preserva_proporcoes_internas():
    vol = torch.tensor([[10.0, 20.0, 30.0]])
    m = compute_volume_max(vol)
    norm = normalize(vol, m)
    # ratios internos preservados
    assert torch.allclose(norm[0, 1] / norm[0, 0], torch.tensor(2.0))
    assert torch.allclose(norm[0, 2] / norm[0, 0], torch.tensor(3.0))