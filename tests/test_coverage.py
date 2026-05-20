# Autor: Massanori
# Data: 19/05/2026
# Descrição: Testes unitarios para src/metrics/coverage.py. Cobre:
#   (1) empirical_coverage sem mascara,
#   (2) empirical_coverage com mascara (so conta pixels mascarados),
#   (3) empirical_coverage com mascara zerada (retorna NaN),
#   (4) clopper_pearson_interval em valores conhecidos (binom.confint do R),
#   (5) clopper_pearson edge cases k=0 e k=n,
#   (6) mean_interval_width sem e com mascara,
#   (7) erros claros para shapes incompativeis.
# Roda com: python -m pytest tests/test_coverage.py -v


"""Testes para src/metrics/coverage.py."""
import math

import pytest
import torch

from src.metrics import (
    clopper_pearson_interval,
    empirical_coverage,
    mean_interval_width,
)


# ---------------------------------------------------------------------------
# empirical_coverage
# ---------------------------------------------------------------------------

def test_empirical_coverage_sem_mascara():
    lower = torch.tensor([0.0, 0.0, 0.0])
    upper = torch.tensor([1.0, 1.0, 1.0])
    target = torch.tensor([0.5, 0.7, 1.5])  # 2 dentro, 1 fora
    result = empirical_coverage(lower, upper, target)
    assert result['n_covered'] == 2
    assert result['n_total'] == 3
    assert abs(result['coverage'] - 2/3) < 1e-6
    # CI 95% binomial Clopper-Pearson para k=2, n=3
    assert 0.0 <= result['ci_lower'] <= 2/3
    assert 2/3 <= result['ci_upper'] <= 1.0


def test_empirical_coverage_com_mascara():
    lower = torch.tensor([0.0, 0.0, 0.0])
    upper = torch.tensor([1.0, 1.0, 1.0])
    target = torch.tensor([0.5, 0.7, 1.5])
    mask = torch.tensor([1.0, 1.0, 0.0])  # exclui o terceiro pixel
    result = empirical_coverage(lower, upper, target, mask=mask)
    assert result['n_covered'] == 2
    assert result['n_total'] == 2
    assert result['coverage'] == 1.0


def test_empirical_coverage_mascara_zerada_retorna_nan():
    lower = torch.zeros(3)
    upper = torch.ones(3)
    target = torch.tensor([0.5, 0.7, 0.9])
    mask = torch.zeros(3)
    result = empirical_coverage(lower, upper, target, mask=mask)
    assert math.isnan(result['coverage'])
    assert result['n_total'] == 0
    assert math.isnan(result['ci_lower'])
    assert math.isnan(result['ci_upper'])


def test_empirical_coverage_target_no_boundary_inclusivo():
    # target == lower ou target == upper deve ser considerado dentro
    lower = torch.tensor([0.0, 1.0])
    upper = torch.tensor([1.0, 2.0])
    target = torch.tensor([0.0, 2.0])  # ambos no boundary
    result = empirical_coverage(lower, upper, target)
    assert result['n_covered'] == 2
    assert result['coverage'] == 1.0


def test_empirical_coverage_rejeita_shape_mismatch():
    with pytest.raises(ValueError, match='Shape mismatch'):
        empirical_coverage(
            torch.zeros(3), torch.ones(3), torch.zeros(5),
        )


# ---------------------------------------------------------------------------
# clopper_pearson_interval
# ---------------------------------------------------------------------------

def test_clopper_pearson_valores_conhecidos():
    """Valores comparados com binom.confint do R (Clopper-Pearson exato).

    R: binom.confint(60, 100, methods='exact')
        lower ~= 0.4972, upper ~= 0.6967
    """
    lo, hi = clopper_pearson_interval(60, 100, confidence=0.95)
    assert 0.49 < lo < 0.51
    assert 0.69 < hi < 0.71


def test_clopper_pearson_valor_50_50():
    """Para k=50, n=100, IC simetrico em torno de 0.5."""
    lo, hi = clopper_pearson_interval(50, 100, confidence=0.95)
    # R: ~0.3983, ~0.6017
    assert 0.39 < lo < 0.41
    assert 0.59 < hi < 0.61


def test_clopper_pearson_k_zero():
    lo, hi = clopper_pearson_interval(0, 100)
    assert lo == 0.0
    assert 0.0 < hi < 0.05  # R: ~0.0362


def test_clopper_pearson_k_igual_n():
    lo, hi = clopper_pearson_interval(100, 100)
    assert hi == 1.0
    assert 0.95 < lo < 1.0  # R: ~0.9638


def test_clopper_pearson_consistente_com_grande_n():
    """Para n grande e proporcao moderada, IC ~ proporcao +- ~2 SE.

    Wald-style SE = sqrt(p(1-p)/n) = sqrt(0.5*0.5/10000) = 0.005
    Half-width esperado: ~0.01.
    Clopper-Pearson e mais conservador, ent\u00e3o esperamos half-width ~ 0.01-0.011.
    """
    lo, hi = clopper_pearson_interval(5000, 10000, confidence=0.95)
    assert abs((hi - lo) / 2 - 0.0098) < 0.003


def test_clopper_pearson_rejeita_invalid_inputs():
    with pytest.raises(ValueError):
        clopper_pearson_interval(-1, 100)
    with pytest.raises(ValueError):
        clopper_pearson_interval(101, 100)
    with pytest.raises(ValueError):
        clopper_pearson_interval(50, 0)
    with pytest.raises(ValueError):
        clopper_pearson_interval(50, 100, confidence=1.5)


# ---------------------------------------------------------------------------
# mean_interval_width
# ---------------------------------------------------------------------------

def test_mean_interval_width_sem_mascara():
    lower = torch.tensor([0.0, 0.0])
    upper = torch.tensor([1.0, 3.0])
    w = mean_interval_width(lower, upper)
    assert w == 2.0  # (1 + 3) / 2


def test_mean_interval_width_com_mascara():
    lower = torch.tensor([0.0, 0.0, 0.0])
    upper = torch.tensor([1.0, 3.0, 10.0])
    mask = torch.tensor([1.0, 1.0, 0.0])  # exclui o 10
    w = mean_interval_width(lower, upper, mask=mask)
    assert w == 2.0  # (1 + 3) / 2


def test_mean_interval_width_mascara_zerada_retorna_nan():
    lower = torch.zeros(3)
    upper = torch.ones(3)
    mask = torch.zeros(3)
    w = mean_interval_width(lower, upper, mask=mask)
    assert math.isnan(w)


def test_mean_interval_width_rejeita_shape_mismatch():
    with pytest.raises(ValueError, match='Shape mismatch'):
        mean_interval_width(torch.zeros(3), torch.ones(5))
