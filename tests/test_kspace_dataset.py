# Autor: Massanori
# Data: 14/05/2026
# Descrição: Testes unitários do kspace_dataset. Cobre: 1) load_split lê
#            arquivo válido e retorna conjunto de IDs, 2) ignora linhas em
#            branco e whitespace, 3) rejeita split_name inválido, 4) FileNot
#            FoundError em path inválido, 5) ValueError em arquivo vazio,
#            6) make_volume_filter aceita/rejeita corretamente baseado em
#            stem do path, 7) tolera Path object ou string, 8) NÃO casa por
#            prefixo (igualdade exata do stem), 9) make_brain_mask_func tem
#            os hiperparâmetros corretos do checkpoint oficial, 10)
#            build_brain_kspace_dataset valida uso correto (mutuamente
#            exclusivo entre split_name e volume_ids). Testes que dependem
#            de IO real (SliceDataset varrendo .h5) NÃO estão aqui — esses
#            ficam no smoke test do sanity_check_varnet.py do passo 4.
#            Roda com: python -m pytest tests/test_kspace_dataset.py -v


"""Testes unitarios do src.data.kspace_dataset.

Roda 100% sem dados reais — testa apenas os componentes que sao funções
puras (carregamento de split, filtros, builder de mascara). A composicao
com SliceDataset (que precisa de .h5 reais) e validada no smoke test do
passo 4 (sanity_check_varnet.py).
"""
from pathlib import Path

import pytest

from src.data.kspace_dataset import (
    DEFAULT_ACCELERATION,
    DEFAULT_CENTER_FRACTION,
    VALID_SPLIT_NAMES,
    build_brain_kspace_dataset,
    load_split,
    make_brain_mask_func,
    make_volume_filter,
)


# -------------------- load_split --------------------


def test_load_split_reads_valid_file(tmp_path):
    f = tmp_path / 'val.txt'
    f.write_text(
        'file_brain_AXFLAIR_200_6002460\n'
        'file_brain_AXT1_201_6002804\n',
        encoding='utf-8',
    )
    result = load_split('val', tmp_path)
    assert result == {
        'file_brain_AXFLAIR_200_6002460',
        'file_brain_AXT1_201_6002804',
    }


def test_load_split_real_val_size(tmp_path):
    """Simula o val.txt real (46 volumes: 26 AXFLAIR + 8 AXT1POST + 12 AXT1)."""
    f = tmp_path / 'val.txt'
    lines = (
        [f'file_brain_AXFLAIR_200_{i:07d}' for i in range(26)]
        + [f'file_brain_AXT1POST_200_{i:07d}' for i in range(8)]
        + [f'file_brain_AXT1_200_{i:07d}' for i in range(12)]
    )
    f.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    result = load_split('val', tmp_path)
    assert len(result) == 46


def test_load_split_ignores_blank_lines(tmp_path):
    f = tmp_path / 'val.txt'
    f.write_text(
        'file_brain_AXFLAIR_200_6002460\n'
        '\n'
        '\n'
        'file_brain_AXT1_201_6002804\n'
        '   \n',
        encoding='utf-8',
    )
    assert len(load_split('val', tmp_path)) == 2


def test_load_split_strips_whitespace(tmp_path):
    f = tmp_path / 'val.txt'
    f.write_text(
        '  file_brain_AXFLAIR_200_6002460  \n'
        '\tfile_brain_AXT1_201_6002804\t\n',
        encoding='utf-8',
    )
    result = load_split('val', tmp_path)
    assert 'file_brain_AXFLAIR_200_6002460' in result
    assert 'file_brain_AXT1_201_6002804' in result


def test_load_split_rejects_invalid_split_name(tmp_path):
    """Salva-vidas contra typos tipo 'valid' (em vez de 'val')."""
    with pytest.raises(ValueError, match='split_name'):
        load_split('valid', tmp_path)


def test_load_split_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_split('val', tmp_path / 'inexistente')


def test_load_split_empty_file_raises(tmp_path):
    f = tmp_path / 'val.txt'
    f.write_text('   \n\n\t\n', encoding='utf-8')
    with pytest.raises(ValueError, match='vazio'):
        load_split('val', tmp_path)


def test_all_valid_split_names_present():
    assert set(VALID_SPLIT_NAMES) == {'train', 'val', 'cal', 'test'}


# -------------------- make_volume_filter --------------------


def test_filter_accepts_allowed_volume():
    f = make_volume_filter({'file_brain_AXFLAIR_200_6002460'})
    raw_sample = ('/data/file_brain_AXFLAIR_200_6002460.h5', 0, {})
    assert f(raw_sample)


def test_filter_rejects_disallowed_volume():
    f = make_volume_filter({'file_brain_AXFLAIR_200_6002460'})
    raw_sample = ('/data/file_brain_AXT1_201_6002804.h5', 0, {})
    assert not f(raw_sample)


def test_filter_accepts_path_object():
    """SliceDataset internamente passa pathlib.Path, nao string."""
    f = make_volume_filter({'file_brain_AXFLAIR_200_6002460'})
    raw_sample = (
        Path('D:/Mri/anotados/file_brain_AXFLAIR_200_6002460.h5'),
        5,
        {},
    )
    assert f(raw_sample)


def test_filter_no_prefix_match():
    """Casa por igualdade do stem, nao por prefixo. Evita que
    'file_brain_AXFLAIR_200_6002460' aceite um arquivo
    'file_brain_AXFLAIR_200_60024600.h5'.
    """
    f = make_volume_filter({'file_brain_AXFLAIR_200_6002460'})
    raw_sample = ('/data/file_brain_AXFLAIR_200_60024600.h5', 0, {})
    assert not f(raw_sample)


def test_filter_distinguishes_axflair_from_axt1post():
    """Garante que substrings sequenciais nao causam falso positivo."""
    f = make_volume_filter({'file_brain_AXFLAIR_200_6002460'})
    raw_sample = ('/data/file_brain_AXT1POST_200_6002460.h5', 0, {})
    assert not f(raw_sample)


def test_filter_empty_raises():
    with pytest.raises(ValueError, match='vazio'):
        make_volume_filter(set())


def test_filter_accepts_iterable_not_just_set():
    """Funciona com list, tuple, ou outro iteravel — converte para set."""
    f = make_volume_filter(['file_brain_AXFLAIR_200_6002460'])
    assert f(('/data/file_brain_AXFLAIR_200_6002460.h5', 0, {}))


# -------------------- make_brain_mask_func --------------------


def test_mask_func_defaults():
    """Defaults batem com o checkpoint brain 4x oficial."""
    mask = make_brain_mask_func()
    assert mask.center_fractions == [0.08]
    assert mask.accelerations == [4]


def test_mask_func_custom_acceleration():
    """Aceita outras aceleracoes (uso futuro: 8x para o brain 8x)."""
    mask = make_brain_mask_func(acceleration=8, center_fraction=0.04)
    assert mask.center_fractions == [0.04]
    assert mask.accelerations == [8]


# -------------------- build_brain_kspace_dataset (validação de uso) --------


def test_builder_requires_one_mode(tmp_path):
    """Sem volume_ids nem split_name deve falhar."""
    with pytest.raises(ValueError, match='EXATAMENTE UM'):
        build_brain_kspace_dataset(data_path=tmp_path)


def test_builder_rejects_both_modes(tmp_path):
    """Passar ambos volume_ids e split_name deve falhar."""
    with pytest.raises(ValueError, match='EXATAMENTE UM'):
        build_brain_kspace_dataset(
            data_path=tmp_path,
            volume_ids=['file_brain_AXFLAIR_200_6002460'],
            split_name='val',
        )


def test_builder_requires_splits_dir_when_split_name_given(tmp_path):
    """split_name sem splits_dir e erro de uso explicito."""
    with pytest.raises(ValueError, match='splits_dir'):
        build_brain_kspace_dataset(data_path=tmp_path, split_name='val')


def test_builder_default_constants():
    """Confirma que defaults publicos batem com o esperado."""
    assert DEFAULT_ACCELERATION == 4
    assert DEFAULT_CENTER_FRACTION == 0.08
