# Autor: Massanori
# Data: 13/05/2026
# Descrição: Testes unitários para src/data/lesion_masks.py. 7 testes que
#            blindam a correção de flip vertical de bboxes contra refatorações
#            futuras: validação sem flip, validação com flip, caso real do
#            diagnóstico (AXFLAIR_200_6002493 fatia 6), união binária de
#            múltiplas bboxes, clipping em bordas, suporte a imagens não
#            quadradas, e exclusão correta de anotações study-level. Recebe:
#            parâmetros sintéticos via parametrização do pytest. Retorna:
#            assertions pass/fail. Roda com:
#            python -m pytest tests/test_lesion_masks.py -v


"""
Testes unitarios da correcao de orientacao em lesion_masks.py
"""
import pandas as pd
import torch
import pytest

from src.data.lesion_masks import bbox_to_mask, slice_mask_from_annotations


def test_bbox_sem_flip_aparece_no_local_dicom():
    """Sem flip, bbox em (x=10, y=20, 5x4) deve aparecer em y=20."""
    mask = bbox_to_mask(10, 20, 5, 4, (100, 100), apply_y_flip=False)
    assert mask[20:24, 10:15].sum().item() == 20.0  # 5 * 4
    assert mask.sum().item() == 20.0


def test_bbox_com_flip_aparece_no_local_espelhado():
    """Com flip, bbox y=20 em imagem H=100 deve aparecer em y=100-20-4=76."""
    mask = bbox_to_mask(10, 20, 5, 4, (100, 100), apply_y_flip=True)
    assert mask[76:80, 10:15].sum().item() == 20.0
    assert mask[0:76, :].sum().item() == 0.0  # nada acima
    assert mask[80:, :].sum().item() == 0.0   # nada abaixo


def test_caso_real_da_validacao():
    """
    Caso real do diagnostico em AXFLAIR_200_6002493 fatia 6:
    Extra-axial mass com x=99, y=201, w=63, h=60 em 320x320.
    y_target esperado = 320 - 201 - 60 = 59.
    """
    mask = bbox_to_mask(99, 201, 63, 60, (320, 320), apply_y_flip=True)
    assert mask[59:119, 99:162].sum().item() == 63.0 * 60.0
    assert mask.sum().item() == 63.0 * 60.0


def test_uniao_de_multiplas_bboxes_eh_binaria():
    """Duas bboxes sobrepostas: resultado deve ter max=1.0 e area da uniao."""
    df = pd.DataFrame([
        {'x': 10, 'y': 10, 'width': 20, 'height': 20},
        {'x': 20, 'y': 20, 'width': 20, 'height': 20},  # sobreposicao 10x10
    ])
    mask = slice_mask_from_annotations(df, (100, 100), apply_y_flip=False)
    assert mask.max().item() == 1.0
    assert mask.sum().item() == 400 + 400 - 100  # 700 px


def test_bbox_fora_dos_limites_eh_clipado():
    mask = bbox_to_mask(95, 95, 20, 20, (100, 100), apply_y_flip=False)
    assert mask.sum().item() == 25.0  # so 5x5 dentro da imagem


def test_imagem_nao_quadrada():
    """Validar suporte a shape (320, 260) observado em alguns volumes."""
    mask = bbox_to_mask(50, 100, 20, 30, (320, 260), apply_y_flip=True)
    # y_target = 320 - 100 - 30 = 190
    assert mask[190:220, 50:70].sum().item() == 600.0


def test_anotacao_study_level_eh_ignorada():
    """Linhas com NaN em coordenadas (study-level labels) devem ser puladas."""
    df = pd.DataFrame([
        {'x': float('nan'), 'y': float('nan'),
         'width': float('nan'), 'height': float('nan')},
        {'x': 10, 'y': 10, 'width': 5, 'height': 5},
    ])
    mask = slice_mask_from_annotations(df, (100, 100), apply_y_flip=False)
    assert mask.sum().item() == 25.0  # so a bbox valida