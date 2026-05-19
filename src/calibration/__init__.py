# Autor: Massanori
# Data: 19/05/2026
# Descrição: Modulo de calibracao conforme para o S5.7. Implementa scores
#            de nao-conformidade (Romano et al., 2019 para QR; Edupuganti et
#            al., 2021 para ResM) e a logica de cobertura empirica usada
#            em todos os 3 grupos. Garante cobertura marginal exata sob
#            permutability (exchangeability) dos splits cal/test. Suporta
#            apenas dispatch entre 'qr' (Grupos B/C) e 'resm' (Grupo A) —
#            duas familias de score que cobrem todos os 3 grupos do TCC.

"""Calibracao conforme para os 3 grupos do S5.

Referencias-base:
    - Romano, Y.; Patterson, E.; Candes, E. (2019). Conformalized Quantile
      Regression. NeurIPS 32, 3543-3553.
    - Angelopoulos, A.N.; Bates, S. (2023). Conformal Prediction: A Gentle
      Introduction. Foundations and Trends in ML, 16(4):494-591.
    - Edupuganti, V. et al. (2021). Uncertainty Quantification in Deep MRI
      Reconstruction. IEEE Trans. Med. Imaging, 40(1):239-250.
    - Angelopoulos, A.N. et al. (2022). Image-to-Image Regression with
      Distribution-Free Uncertainty Quantification and Applications in
      Imaging. ICML.
"""

from src.calibration.coverage import (
    calibrate,
    coverage_stats,
    evaluate,
)
from src.calibration.nonconformity import (
    apply_qhat_qr,
    apply_qhat_resm,
    compute_qhat,
    nonconformity_qr,
    nonconformity_resm,
)

__all__ = [
    'apply_qhat_qr',
    'apply_qhat_resm',
    'calibrate',
    'compute_qhat',
    'coverage_stats',
    'evaluate',
    'nonconformity_qr',
    'nonconformity_resm',
]
