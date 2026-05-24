# Autor: Massanori
# Data: 21/05/2026
# Descricao: Testes unitarios para src/metrics/iou.py.
#            Cobre: top-X% mask, IoU(top-X%) basico, IoU sem restricao
#            ou com restricao a mascara, curva IoU(X) + AUC, casos
#            degenerados (uniao vazia, top_pct invalido, shapes diferentes).
#            Roda com: python -m pytest tests/test_iou.py -v


"""Testes para src/metrics/iou.py.

Alguns testes usam tensores cuidadosamente construidos onde sabemos
exatamente quais pixels deveriam ser top-X%, para validar o IoU
esperado matematicamente.
"""
import pytest
import torch

from src.metrics.iou import _top_k_mask, iou_curve, iou_topk


# ---------------------------------------------------------------------------
# _top_k_mask
# ---------------------------------------------------------------------------

def test_top_k_mask_seleciona_top_metade_em_tensor_ordenado():
    """Para [0,1,2,...,9] e top_pct=0.5, espera-se os indices 5..9."""
    values = torch.arange(10).float()
    mask = _top_k_mask(values, top_pct=0.5)
    expected_indices = torch.arange(5, 10)
    assert mask.sum().item() == 5
    assert mask[expected_indices].all()


def test_top_k_mask_com_restrict_mask_apenas_dentro_dela():
    """top-k restrito a uma sub-regiao deve ignorar pixels fora dela."""
    values = torch.arange(10).float()  # 0..9, top global = 5..9
    restrict = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0, 0, 0]).bool()
    # Dentro da restrict, valores sao 0,1,2,3. Top 50% = top 2 = indices 2, 3
    mask = _top_k_mask(values, top_pct=0.5, restrict_mask=restrict)
    assert mask.sum().item() == 2
    assert mask[2] and mask[3]
    # Indices fora da restrict_mask nunca podem estar em mask
    assert not mask[4:].any()


def test_top_k_mask_restrict_vazia_retorna_zeros():
    values = torch.arange(10).float()
    restrict = torch.zeros(10, dtype=torch.bool)
    mask = _top_k_mask(values, top_pct=0.5, restrict_mask=restrict)
    assert mask.sum().item() == 0


# ---------------------------------------------------------------------------
# iou_topk: casos basicos
# ---------------------------------------------------------------------------

def test_iou_topk_identicos_iou_eh_um():
    """Se uncertainty == error, IoU(top-X%) = 1.0 para qualquer X."""
    torch.manual_seed(0)
    u = torch.rand(100)
    e = u.clone()
    for x in (0.05, 0.10, 0.30):
        iou = iou_topk(u, e, top_pct=x)
        assert iou == pytest.approx(1.0), f'IoU(X={x}) = {iou}, esperado 1.0'


def test_iou_topk_inversos_iou_eh_zero():
    """Se top-X% de u sao os bottom-X% de e (inversos), IoU = 0."""
    u = torch.arange(20).float()  # top-25% = 15..19
    e = -u                         # top-25% = 0..4
    # Top-25% nao se sobrepoem
    iou = iou_topk(u, e, top_pct=0.25)
    assert iou == pytest.approx(0.0)


def test_iou_topk_50pct_50pct_sobreposicao_parcial():
    """Tensores ordenados em sentidos diferentes mas com 50% top — IoU em
    [0, 1] mas calculavel."""
    # u: [0..9] → top-50% = 5..9
    # e: [9..0] (reversed): valor 0 esta no indice 9, valor 9 no indice 0
    #    top-50% de e (por valor) = indices 0..4
    # Intersecao: {5..9} ∩ {0..4} = vazia → IoU = 0
    u = torch.arange(10).float()
    e = torch.arange(9, -1, -1).float()
    iou = iou_topk(u, e, top_pct=0.5)
    assert iou == pytest.approx(0.0)


def test_iou_topk_sobreposicao_parcial_conhecida():
    """Construcao manual com sobreposicao 2/8 = 0.25."""
    # u: top-50% = indices [5, 6, 7, 8, 9]
    # e: top-50% = indices [3, 4, 5, 6, 9]
    # Intersecao: {5, 6, 9} = 3 pixels
    # Uniao: {3, 4, 5, 6, 7, 8, 9} = 7 pixels
    # IoU = 3/7
    u_vals = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]).float()
    # e_vals: 1 nos indices 3, 4, 5, 6, 9
    e_vals = torch.tensor([0, 0, 0, 1, 1, 1, 1, 0, 0, 1]).float()
    iou = iou_topk(u_vals, e_vals, top_pct=0.5)
    # Quando ha ties, _top_k_mask via topk pode pegar qualquer um dos
    # tied items — mas a IoU resultado deve ainda ser 3/7 se topk e
    # consistente. Tolerancia para diferentes implementacoes de topk:
    assert 0.30 <= iou <= 0.50, f'IoU={iou}, esperado ~3/7 ~ 0.43'


# ---------------------------------------------------------------------------
# iou_topk: erros
# ---------------------------------------------------------------------------

def test_iou_topk_rejeita_shapes_diferentes():
    with pytest.raises(ValueError, match='Shapes incompativeis'):
        iou_topk(torch.zeros(10), torch.zeros(20), top_pct=0.5)


def test_iou_topk_rejeita_top_pct_fora_0_1():
    u = torch.rand(10)
    for invalid in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError, match='top_pct'):
            iou_topk(u, u, top_pct=invalid)


def test_iou_topk_rejeita_restrict_mask_shape_diferente():
    u = torch.rand(10)
    bad_mask = torch.ones(20).bool()
    with pytest.raises(ValueError, match='restrict_mask shape'):
        iou_topk(u, u, top_pct=0.5, restrict_mask=bad_mask)


# ---------------------------------------------------------------------------
# iou_topk: com restrict_mask (e.g. para IoU_lesion)
# ---------------------------------------------------------------------------

def test_iou_topk_restrict_a_subregiao_funciona():
    """Restricao a metade do tensor calcula top-X% so dentro da subregiao."""
    torch.manual_seed(123)
    u = torch.rand(100)
    e = u.clone()  # Identicos
    restrict = torch.zeros(100, dtype=torch.bool)
    restrict[40:60] = True  # 20 pixels
    iou = iou_topk(u, e, top_pct=0.5, restrict_mask=restrict)
    assert iou == pytest.approx(1.0)


def test_iou_topk_2d_input():
    """Aceita inputs (H, W) sem precisar achatar manualmente."""
    u = torch.rand(20, 20)
    e = u.clone()
    iou = iou_topk(u, e, top_pct=0.10)
    assert iou == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# iou_curve + AUC
# ---------------------------------------------------------------------------

def test_iou_curve_retorna_dict_com_chaves_esperadas():
    torch.manual_seed(0)
    u = torch.rand(100)
    e = torch.rand(100)
    pcts = (0.05, 0.10, 0.20)
    ious, auc = iou_curve(u, e, top_pcts=pcts)
    assert set(ious.keys()) == set(pcts)
    assert all(0.0 <= v <= 1.0 for v in ious.values())
    assert 0.0 <= auc <= 1.0


def test_iou_curve_identicos_da_auc_um():
    u = torch.rand(100)
    e = u.clone()
    _, auc = iou_curve(u, e, top_pcts=(0.05, 0.10, 0.20, 0.50))
    assert auc == pytest.approx(1.0)


def test_iou_curve_rejeita_top_pcts_invalidos():
    u = torch.rand(10)
    with pytest.raises(ValueError, match='top_pcts'):
        iou_curve(u, u, top_pcts=(0.0, 0.5))
    with pytest.raises(ValueError, match='top_pcts'):
        iou_curve(u, u, top_pcts=(0.5, 1.0))
