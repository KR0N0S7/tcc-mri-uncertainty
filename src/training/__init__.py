# Autor: Massanori
# Data: 17/05/2026
# Descrição: Modulo de treino. Expoe set_global_seed (reprodutibilidade) e
#            o loop unificado train() + utilitarios (make_optimizer,
#            make_scheduler, save_checkpoint, load_checkpoint) usados pelos
#            Grupos A/B/C do S5. Toda a infra e polimorfica via interface
#            unificada D3: um unico loop atende os tres grupos.

from src.training.random_seed import set_global_seed
from src.training.train_loop import (
    cycle_loader,
    load_checkpoint,
    make_optimizer,
    make_scheduler,
    save_checkpoint,
    train,
    train_step,
    validate,
)

__all__ = [
    'cycle_loader',
    'load_checkpoint',
    'make_optimizer',
    'make_scheduler',
    'save_checkpoint',
    'set_global_seed',
    'train',
    'train_step',
    'validate',
]
