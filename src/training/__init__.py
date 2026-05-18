# Autor: Massanori
# Data: 17/05/2026
# Descrição: Modulo de infraestrutura de treino do S5. Expoe: setup
#            deterministico (random_seed), scheduler de warmup linear,
#            checkpointing atomico, logger dual TB+CSV. O train_loop
#            unificado e adicionado no commit seguinte.

from src.training.checkpointing import (
    CHECKPOINT_SCHEMA_VERSION,
    load_checkpoint,
    save_checkpoint,
)
from src.training.logging_utils import DualLogger
from src.training.random_seed import (
    get_rng_states,
    set_global_seeds,
    set_rng_states,
)
from src.training.scheduler import LinearWarmupScheduler

__all__ = [
    'CHECKPOINT_SCHEMA_VERSION',
    'DualLogger',
    'LinearWarmupScheduler',
    'get_rng_states',
    'load_checkpoint',
    'save_checkpoint',
    'set_global_seeds',
    'set_rng_states',
]
