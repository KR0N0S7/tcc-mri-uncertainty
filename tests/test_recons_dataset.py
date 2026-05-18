# Autor: Massanori
# Data: 17/05/2026
# Descrição: Testes unitários para src/data/recons_dataset.py. 8 testes que
#            blindam (1) extracao de sequencia AXFLAIR/AXT1/AXT1POST, (2)
#            contagem correta de slices via index, (3) shapes (1, H, W) na
#            saida, (4) normalizacao por max_val trazendo recon para [0, 1],
#            (5) bypass de normalizacao via apply_normalization=False, (6)
#            lesion_mask de zeros quando masks_dir e None, (7) lesion_mask
#            carregada quando .pt existe, (8) erro claro se recons_dir vazio.
#            Recebe: tensores sinteticos que mimetizam o schema do S4 num
#            tmp_path. Retorna: assertions pass/fail. Roda com:
#            python -m pytest tests/test_recons_dataset.py -v


"""Testes para ReconsSliceDataset.

Usa fixtures sinteticas com schema identico ao do S4
(scripts/precompute_reconstructions.py:save_volume) para nao depender do
dataset Kaggle real. Cada teste e isolado num tmp_path proprio do pytest.
"""
import numpy as np
import pytest
import torch

from src.data.recons_dataset import (
    KNOWN_SEQUENCES,
    ReconsSliceDataset,
    extract_sequence,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_synthetic_npz(out_dir, volume_id, n_slices=4, H=32, W=32,
                       max_val=10.0, split='train'):
    """Cria um .npz no schema do S4 com tensores sinteticos.

    recon e target sao linspaces previsiveis para que os testes possam
    verificar normalizacao matematicamente.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed=42)
    target = rng.uniform(0, max_val, size=(n_slices, H, W)).astype(np.float32)
    recon = target + rng.normal(0, 0.05 * max_val, size=target.shape).astype(np.float32)
    error_map = np.abs(target - recon).astype(np.float32)

    np.savez_compressed(
        out_dir / f'{volume_id}.npz',
        recon=recon,
        target=target,
        error_map=error_map,
        max_val=np.float32(max_val),
        volume_id=np.array(volume_id),
        split=np.array(split),
        acceleration=np.int32(4),
        center_fraction=np.float32(0.08),
        varnet_sha256=np.array('deadbeef' * 8),
    )
    return n_slices, H, W


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_extract_sequence_reconhece_AXFLAIR_AXT1_AXT1POST():
    assert extract_sequence('file_brain_AXFLAIR_200_6002460') == 'AXFLAIR'
    assert extract_sequence('file_brain_AXT1_200_1234567') == 'AXT1'
    assert extract_sequence('file_brain_AXT1POST_201_7654321') == 'AXT1POST'
    assert extract_sequence('nao_tem_padrao_AX') == 'UNKNOWN'
    for seq in KNOWN_SEQUENCES:
        assert extract_sequence(f'file_brain_{seq}_X_Y') == seq


def test_indexa_todos_os_slices_de_todos_os_npz(tmp_path):
    recons_dir = tmp_path / 'train'
    _make_synthetic_npz(recons_dir, 'file_brain_AXFLAIR_200_1', n_slices=4)
    _make_synthetic_npz(recons_dir, 'file_brain_AXT1_200_2', n_slices=5)
    _make_synthetic_npz(recons_dir, 'file_brain_AXT1POST_200_3', n_slices=3)

    ds = ReconsSliceDataset(recons_dir)
    assert len(ds) == 4 + 5 + 3, f'esperado 12 slices, obtido {len(ds)}'


def test_getitem_retorna_shapes_e_chaves_corretas(tmp_path):
    recons_dir = tmp_path / 'train'
    _make_synthetic_npz(recons_dir, 'file_brain_AXFLAIR_200_1',
                       n_slices=2, H=64, W=64)

    ds = ReconsSliceDataset(recons_dir)
    sample = ds[0]

    expected_keys = {'recon', 'target', 'error_map', 'max_val',
                     'volume_id', 'slice_idx', 'sequence', 'lesion_mask'}
    assert set(sample.keys()) == expected_keys
    assert sample['recon'].shape == (1, 64, 64)
    assert sample['target'].shape == (1, 64, 64)
    assert sample['error_map'].shape == (1, 64, 64)
    assert sample['lesion_mask'].shape == (1, 64, 64)
    assert sample['recon'].dtype == torch.float32
    assert sample['volume_id'] == 'file_brain_AXFLAIR_200_1'
    assert sample['sequence'] == 'AXFLAIR'
    assert sample['slice_idx'] == 0


def test_normalizacao_por_max_val_traz_target_para_0_1(tmp_path):
    recons_dir = tmp_path / 'train'
    _make_synthetic_npz(recons_dir, 'file_brain_AXFLAIR_200_1',
                       n_slices=2, max_val=20.0)

    ds = ReconsSliceDataset(recons_dir, apply_normalization=True)
    sample = ds[0]
    # target sintetico esta em [0, 20], dividido por 20 -> [0, 1]
    assert sample['target'].max() <= 1.0 + 1e-6
    assert sample['target'].min() >= 0.0
    assert sample['max_val'].item() == pytest.approx(20.0)


def test_bypass_normalizacao_preserva_magnitudes_brutas(tmp_path):
    recons_dir = tmp_path / 'train'
    _make_synthetic_npz(recons_dir, 'file_brain_AXFLAIR_200_1',
                       n_slices=2, max_val=20.0)

    ds_norm = ReconsSliceDataset(recons_dir, apply_normalization=True)
    ds_raw = ReconsSliceDataset(recons_dir, apply_normalization=False)

    s_norm = ds_norm[0]
    s_raw = ds_raw[0]
    # target raw / max_val == target normalizado
    assert torch.allclose(s_raw['target'] / 20.0, s_norm['target'], atol=1e-6)
    # raw deve ter max acima de 1.0 (sintetico foi gerado em [0, 20])
    assert s_raw['target'].max() > 5.0


def test_lesion_mask_de_zeros_quando_masks_dir_e_None(tmp_path):
    recons_dir = tmp_path / 'train'
    _make_synthetic_npz(recons_dir, 'file_brain_AXFLAIR_200_1', n_slices=2)

    ds = ReconsSliceDataset(recons_dir, masks_dir=None)
    sample = ds[0]
    assert sample['lesion_mask'].sum() == 0.0
    assert sample['lesion_mask'].shape == sample['recon'].shape


def test_lesion_mask_carregada_quando_pt_existe(tmp_path):
    recons_dir = tmp_path / 'train'
    masks_dir = tmp_path / 'masks'
    masks_dir.mkdir()

    n_slices, H, W = _make_synthetic_npz(
        recons_dir, 'file_brain_AXFLAIR_200_1',
        n_slices=3, H=16, W=16,
    )
    # Mascara sintetica: fatia 1 tem retangulo de 1s, demais zero.
    mask_vol = torch.zeros(n_slices, H, W, dtype=torch.float32)
    mask_vol[1, 4:10, 4:10] = 1.0
    torch.save(mask_vol, masks_dir / 'file_brain_AXFLAIR_200_1.pt')

    ds = ReconsSliceDataset(recons_dir, masks_dir=masks_dir)
    s0 = ds[0]
    s1 = ds[1]
    s2 = ds[2]
    assert s0['lesion_mask'].sum() == 0.0
    assert s1['lesion_mask'].sum() == 36.0   # 6x6 retangulo
    assert s2['lesion_mask'].sum() == 0.0


def test_erro_claro_se_recons_dir_vazio(tmp_path):
    empty = tmp_path / 'empty_split'
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match='Nenhum .npz'):
        ReconsSliceDataset(empty)
