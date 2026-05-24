# Autor: Massanori
# Data: 21/05/2026
# Descricao: Testes unitarios para src/metrics/ulas.py. Cobre:
#  (1) sobel_gradient: derivada exata em ramps lineares, simetria,
#  (2) ulas em casos sinteticos onde a resposta e calculavel:
#       - u = e (identidade): ULAS == 1.0
#       - u, e gradientes radiais coerentes: ULAS proximo de 1.0
#       - u radial, e ortogonal: ULAS proximo de 0.0
#       - u, e aleatorios: ULAS proximo do null baseline (~0.6)
#  (3) mascara vazia -> retorna 0.0 (NaN-safe),
#  (4) erros: shapes incompativeis,
#  (5) null baseline: media proxima do esperado teorico (~2/pi).
# Roda com: python -m pytest tests/test_ulas.py -v


"""Testes para src/metrics/ulas.py.

A validacao sintetica e essencial porque ULAS e a contribuicao original
do TCC — precisa demonstrar que ela mede o que claimamos antes de
reportar valores reais no S5.8. Banca vai perguntar 'como sabe que
essa metrica e informativa?' — esses testes sao a resposta.
"""
import math

import pytest
import torch

from src.metrics.ulas import sobel_gradient, ulas, ulas_with_null


# ---------------------------------------------------------------------------
# sobel_gradient: validacao numerica do operador
# ---------------------------------------------------------------------------

def test_sobel_gradient_em_ramp_linear_x():
    """Para f(x, y) = x, gx ~ 1.0 e gy ~ 0.0 em todos os pixels internos."""
    H = W = 20
    xx = torch.arange(W).float().unsqueeze(0).expand(H, W)  # (H, W), f(i,j) = j
    gx, gy = sobel_gradient(xx)
    # Pixels nas bordas tem replicate padding (artefato menor); olhamos interior
    interior_gx = gx[1:-1, 1:-1]
    interior_gy = gy[1:-1, 1:-1]
    assert torch.allclose(interior_gx, torch.ones_like(interior_gx), atol=1e-5)
    assert torch.allclose(interior_gy, torch.zeros_like(interior_gy), atol=1e-5)


def test_sobel_gradient_em_ramp_linear_y():
    """Para f(x, y) = y, gx ~ 0.0 e gy ~ 1.0 em pixels internos."""
    H = W = 20
    yy = torch.arange(H).float().unsqueeze(1).expand(H, W)  # f(i,j) = i
    gx, gy = sobel_gradient(yy)
    interior_gx = gx[1:-1, 1:-1]
    interior_gy = gy[1:-1, 1:-1]
    assert torch.allclose(interior_gx, torch.zeros_like(interior_gx), atol=1e-5)
    assert torch.allclose(interior_gy, torch.ones_like(interior_gy), atol=1e-5)


def test_sobel_gradient_aceita_3d_e_4d():
    """sobel_gradient aceita (H,W), (1,H,W), (B,1,H,W)."""
    f2d = torch.rand(10, 10)
    f3d = f2d.unsqueeze(0)
    f4d = f2d.unsqueeze(0).unsqueeze(0)

    gx2, gy2 = sobel_gradient(f2d)
    gx3, gy3 = sobel_gradient(f3d)
    gx4, gy4 = sobel_gradient(f4d)

    assert gx2.shape == f2d.shape
    assert gx3.shape == f3d.shape
    assert gx4.shape == f4d.shape
    # Valores equivalentes apos squeeze
    assert torch.allclose(gx2, gx3.squeeze(0), atol=1e-6)
    assert torch.allclose(gx2, gx4.squeeze(0).squeeze(0), atol=1e-6)


def test_sobel_gradient_rejeita_input_invalido():
    with pytest.raises(ValueError):
        sobel_gradient(torch.rand(2, 10, 10))  # 3D com 2 canais
    with pytest.raises(ValueError):
        sobel_gradient(torch.rand(2, 3, 10, 10))  # 4D com 3 canais
    with pytest.raises(ValueError):
        sobel_gradient(torch.rand(5, 5, 5, 5, 5))  # 5D


# ---------------------------------------------------------------------------
# ulas: casos sinteticos onde a resposta e calculavel
# ---------------------------------------------------------------------------

def test_ulas_identidade_eh_um():
    """Se u == e, gradientes identicos, ULAS = 1.0 exatamente."""
    torch.manual_seed(0)
    u = torch.rand(50, 50)
    e = u.clone()
    mask = torch.ones_like(u)
    score = ulas(u, e, mask)
    assert score == pytest.approx(1.0, abs=1e-5)


def test_ulas_negativo_eh_um_via_absoluto():
    """u e -u tem gradientes antiparalelos -> |cos| = 1.0.

    Demonstra que ULAS usa |cos|: direcao oposta tambem = alinhado.
    """
    torch.manual_seed(1)
    u = torch.rand(50, 50)
    e = -u
    mask = torch.ones_like(u)
    score = ulas(u, e, mask)
    assert score == pytest.approx(1.0, abs=1e-5)


def test_ulas_radial_coerente_alto():
    """u = r^2 (paraboloide), e = r (cone): ambos com gradientes radiais.

    ULAS deve ser proximo de 1.0 (gradientes apontam radialmente).
    """
    H = W = 100
    cy, cx = 50, 50
    yy, xx = torch.meshgrid(torch.arange(H).float(), torch.arange(W).float(), indexing='ij')
    r2 = (xx - cx) ** 2 + (yy - cy) ** 2
    u = r2  # paraboloide
    e = torch.sqrt(r2 + 1e-3)  # cone radial (eps para suavidade no centro)
    # Mascara exclui o centro (gradiente indefinido) e bordas
    mask = ((r2 > 25) & (r2 < 2000)).float()
    score = ulas(u, e, mask)
    assert score > 0.95, f'ULAS={score:.4f}, esperado > 0.95'


def test_ulas_ortogonal_baixo():
    """u = x (horizontal), e = y (vertical): gradientes ortogonais.

    ULAS deve ser proximo de 0.0.
    """
    H = W = 40
    xx = torch.arange(W).float().unsqueeze(0).expand(H, W)
    yy = torch.arange(H).float().unsqueeze(1).expand(H, W)
    u = xx
    e = yy
    # Mascara central (evita bordas)
    mask = torch.zeros_like(u)
    mask[5:-5, 5:-5] = 1.0
    score = ulas(u, e, mask)
    assert score < 0.05, f'ULAS={score:.4f}, esperado < 0.05'


def test_ulas_aleatorios_aproximam_null_baseline():
    """u e e aleatorios independentes -> ULAS proximo de E[|cos(U_2D, V_2D)|] = 2/pi.

    Para gradientes 2D iid uniformes no circulo:
        E[|cos|] = (2/pi) integral_0^1 1/sqrt(1-x^2) dx ... = 2/pi ~ 0.637.

    Como na pratica ha correlacao espacial (Sobel suaviza), o valor pode
    desviar um pouco — mas deve estar na faixa [0.4, 0.8].
    """
    torch.manual_seed(42)
    u = torch.rand(100, 100)
    e = torch.rand(100, 100)
    mask = torch.ones_like(u)
    score = ulas(u, e, mask)
    # Faixa ampla para conta com correlacao espacial do Sobel
    assert 0.4 < score < 0.8, (
        f'ULAS={score:.4f}, esperado proximo de 2/pi ~ 0.637 +/- 0.2'
    )


# ---------------------------------------------------------------------------
# ulas: mascara vazia + erros
# ---------------------------------------------------------------------------

def test_ulas_mascara_vazia_retorna_zero():
    u = torch.rand(20, 20)
    e = torch.rand(20, 20)
    mask = torch.zeros_like(u)
    score = ulas(u, e, mask)
    assert score == 0.0


def test_ulas_rejeita_shapes_diferentes():
    u = torch.rand(10, 10)
    e = torch.rand(20, 20)
    mask = torch.ones(10, 10)
    with pytest.raises(ValueError, match='Shapes incompativeis'):
        ulas(u, e, mask)

    with pytest.raises(ValueError, match='Shapes incompativeis'):
        ulas(torch.rand(10, 10), torch.rand(10, 10), torch.ones(20, 20))


# ---------------------------------------------------------------------------
# ulas_with_null: baseline empirico
# ---------------------------------------------------------------------------

def test_ulas_with_null_retorna_metadata_completa():
    torch.manual_seed(0)
    u = torch.rand(50, 50)
    e = u.clone()  # ULAS_real deve ser ~ 1.0
    mask = torch.ones_like(u)
    result = ulas_with_null(u, e, mask, n_permutations=5)

    assert set(result.keys()) == {
        'ulas', 'null_mean', 'null_std', 'null_scores', 'z_score',
        'n_lesion_pixels',
    }
    assert result['ulas'] == pytest.approx(1.0, abs=1e-5)
    assert len(result['null_scores']) == 5
    assert result['n_lesion_pixels'] == 50 * 50
    # Como real >> null, z_score deve ser positivo grande
    assert result['z_score'] > 5.0, f'z_score={result["z_score"]:.2f}'


def test_ulas_with_null_baseline_perto_de_2_sobre_pi():
    """Para u e e aleatorios independentes, ULAS_real e null_mean ambos
    devem ficar perto de 2/pi (a diferenca e que null e media de N perms).
    """
    torch.manual_seed(7)
    u = torch.rand(80, 80)
    e = torch.rand(80, 80)
    mask = torch.ones_like(u)
    result = ulas_with_null(u, e, mask, n_permutations=20)

    expected = 2.0 / math.pi  # ~ 0.637
    # Faixa ampla por causa de correlacao espacial do Sobel
    assert 0.4 < result['null_mean'] < 0.8, (
        f'null_mean={result["null_mean"]:.4f}, esperado ~{expected:.4f}'
    )
    # ulas_real e null_mean devem estar perto um do outro (ambos aleatorios)
    assert abs(result['ulas'] - result['null_mean']) < 0.15


def test_ulas_with_null_n_permutations_1_da_std_zero():
    """Edge case: 1 permutacao -> std nao definido, retorna 0.0."""
    torch.manual_seed(0)
    u = torch.rand(20, 20)
    e = torch.rand(20, 20)
    mask = torch.ones_like(u)
    result = ulas_with_null(u, e, mask, n_permutations=1)
    assert result['null_std'] == 0.0
    # z_score com std=0 -> NaN
    assert math.isnan(result['z_score'])
