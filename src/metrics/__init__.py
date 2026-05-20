# Autor: Massanori
# Data: 19/05/2026
# Descrição: Modulo de metricas de cobertura para o S5.8. Expoe
#            empirical_coverage (com IC Clopper-Pearson exato),
#            mean_interval_width, e o utilitario clopper_pearson_interval.
#            Sem dependencia em scipy: a inverse-beta para CP e
#            implementada via continued fraction (Numerical Recipes).

from src.metrics.coverage import (
    clopper_pearson_interval,
    empirical_coverage,
    mean_interval_width,
)

__all__ = [
    'clopper_pearson_interval',
    'empirical_coverage',
    'mean_interval_width',
]
