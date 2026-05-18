# Autor: Massanori
# Data: 17/05/2026
# Descrição: Modulo de modelos. Expoe loaders de redes pre-treinadas usadas
#            no pipeline de pre-computacao do S4 (varnet_loader) e os quantile
#            networks treinados no S5 (Grupos A/B/C em uncertainty_modules).

from src.models.uncertainty_modules import (
    QuantileRegressionLesionModule,
    QuantileRegressionModule,
    ResidualMagnitudeModule,
)
from src.models.varnet_loader import (
    CHECKPOINT_URL,
    DEFAULT_NUM_CASCADES,
    DEFAULT_POOLS,
    DEFAULT_SENS_CHANS,
    DEFAULT_SENS_POOLS,
    DEFAULT_VARNET_CHANS,
    compute_sha256,
    load_pretrained_varnet,
)

__all__ = [
    'CHECKPOINT_URL',
    'DEFAULT_NUM_CASCADES',
    'DEFAULT_POOLS',
    'DEFAULT_SENS_CHANS',
    'DEFAULT_SENS_POOLS',
    'DEFAULT_VARNET_CHANS',
    'QuantileRegressionLesionModule',
    'QuantileRegressionModule',
    'ResidualMagnitudeModule',
    'compute_sha256',
    'load_pretrained_varnet',
]
