# Autor: Massanori
# Data: 14/05/2026
# Descrição: Modulo de modelos. Expoe loaders de redes pre-treinadas usadas
#            no pipeline de pre-computacao do S4 e, eventualmente, os
#            quantile networks treinados no S5.

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
    'compute_sha256',
    'load_pretrained_varnet',
]
