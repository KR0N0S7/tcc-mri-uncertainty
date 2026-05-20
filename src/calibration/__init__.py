# Autor: Massanori
# Data: 19/05/2026
# Descrição: Modulo de calibracao conforme para o S5.7. Expoe os scores
#            de inconformidade (CQR e scaled-CP locally adaptive),
#            o quantile com correcao finite-sample (1-alpha)(n+1)/n, e os
#            wrappers de calibracao end-to-end calibrate_qr (Grupos B/C)
#            e calibrate_resm (Grupo A). Inclui tambem os aplicadores de
#            intervalo calibrado para uso em test-time.

from src.calibration.conformal import (
    apply_cqr_interval,
    apply_resm_interval,
    calibrate_qr,
    calibrate_resm,
    conformal_quantile,
    cqr_score,
    scaled_cp_score,
)

__all__ = [
    'apply_cqr_interval',
    'apply_resm_interval',
    'calibrate_qr',
    'calibrate_resm',
    'conformal_quantile',
    'cqr_score',
    'scaled_cp_score',
]
