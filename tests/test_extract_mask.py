# Autor: Massanori
# Data: 20/05/2026
# Descrição: Testes unitarios para _extract_mask_tensor em src/data/recons_dataset.py.
#            Cobre os 3 formatos comuns de .pt do S3 (tensor direto, dict com
#            chave conhecida, dict com 1 tensor unico) e os caminhos de erro
#            (dict sem chave conhecida + multiplos tensores, tipo nao suportado).
#            Necessario apos o bug 'Could not infer dtype of dict' no Grupo C
#            quando S3 salvava metadados junto da mascara.
# Roda com: python -m pytest tests/test_extract_mask.py -v


"""Testes para src.data.recons_dataset._extract_mask_tensor.

Garantem que o parser de .pt aceita todos os formatos plausiveis em que
as mascaras do S3 podem ter sido salvas, sem regressao acidental.
"""
from pathlib import Path

import pytest
import torch

from src.data.recons_dataset import _MASK_DICT_KEYS, _extract_mask_tensor


# ---------------------------------------------------------------------------
# Caminhos felizes
# ---------------------------------------------------------------------------

def test_extract_mask_tensor_aceita_tensor_direto():
    mask = torch.zeros(16, 320, 320, dtype=torch.bool)
    out = _extract_mask_tensor(mask, Path('fake.pt'))
    assert torch.is_tensor(out)
    assert out.shape == (16, 320, 320)
    assert torch.equal(out, mask)


@pytest.mark.parametrize('key', _MASK_DICT_KEYS)
def test_extract_mask_tensor_aceita_dict_com_chave_conhecida(key):
    """Cada uma das chaves declaradas em _MASK_DICT_KEYS deve funcionar."""
    mask = torch.ones(8, 256, 256)
    raw = {key: mask, 'volume_id': 'fake_id', 'n_slices': 8}
    out = _extract_mask_tensor(raw, Path('fake.pt'))
    assert torch.equal(out, mask)


def test_extract_mask_tensor_aceita_dict_com_unico_tensor_chave_desconhecida():
    """Fallback heuristico: se ha 1 unico tensor entre os valores, usa ele."""
    mask = torch.zeros(4, 128, 128)
    raw = {
        'minha_chave_customizada_xyz': mask,
        'metadata': {'foo': 'bar'},
        'n_slices': 4,
    }
    out = _extract_mask_tensor(raw, Path('fake.pt'))
    assert torch.equal(out, mask)


def test_extract_mask_tensor_aceita_numpy_array():
    import numpy as np
    arr = np.zeros((4, 64, 64), dtype=np.int32)
    out = _extract_mask_tensor(arr, Path('fake.pt'))
    assert torch.is_tensor(out)
    assert tuple(out.shape) == (4, 64, 64)


# ---------------------------------------------------------------------------
# Caminhos de erro
# ---------------------------------------------------------------------------

def test_extract_mask_tensor_falha_dict_com_multiplos_tensores_sem_chave_conhecida():
    raw = {
        'algum_tensor_a': torch.zeros(4, 64, 64),
        'algum_tensor_b': torch.ones(4, 64, 64),
        'metadata': {'foo': 'bar'},
    }
    with pytest.raises(ValueError, match='2 tensores'):
        _extract_mask_tensor(raw, Path('fake.pt'))


def test_extract_mask_tensor_falha_dict_sem_tensor_algum():
    raw = {'volume_id': 'fake', 'n_slices': 4, 'creator': 'me'}
    with pytest.raises(ValueError, match='0 tensores'):
        _extract_mask_tensor(raw, Path('fake.pt'))


def test_extract_mask_tensor_falha_tipo_nao_suportado():
    with pytest.raises(ValueError, match='tipo nao suportado'):
        _extract_mask_tensor({'arbitrary': object()}.values().__iter__(),
                              Path('fake.pt'))


# ---------------------------------------------------------------------------
# Prioridade entre chaves
# ---------------------------------------------------------------------------

def test_extract_mask_tensor_prioriza_chave_anterior_em_caso_de_conflito():
    """Se mais de uma chave conhecida tem tensor, usa a primeira em _MASK_DICT_KEYS.

    Cenario raro mas: garante comportamento deterministico.
    """
    primary = torch.zeros(4, 64, 64)
    secondary = torch.ones(4, 64, 64)
    # _MASK_DICT_KEYS[0] = 'mask', _MASK_DICT_KEYS[1] = 'masks'
    raw = {'mask': primary, 'masks': secondary}
    out = _extract_mask_tensor(raw, Path('fake.pt'))
    assert torch.equal(out, primary)


# ---------------------------------------------------------------------------
# Smoke: import publico (re-export indireto via __init__)
# ---------------------------------------------------------------------------

def test_extract_mask_tensor_e_modulo_level():
    """_extract_mask_tensor deve ser acessivel direto do modulo (nao classe)."""
    from src.data.recons_dataset import _extract_mask_tensor as fn
    assert callable(fn)
