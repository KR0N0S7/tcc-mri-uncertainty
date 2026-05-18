# Autor: Massanori
# Data: 17/05/2026
# Descrição: Setup deterministico de seeds para reprodutibilidade entre runs
#            dos Grupos A/B/C (D4). Recebe: seed (int), deterministic (bool).
#            Retorna: None (side effect global). Aplica seed em torch, numpy,
#            random e cudnn. Tambem expoe get_rng_states/set_rng_states para
#            snapshot e restore de RNGs em checkpoints (resume bit-a-bit
#            identico ao run nao interrompido).


"""Setup deterministico de seeds.

Garante que dois runs com mesma seed inicializam pesos identicamente e
percorrem o mesmo caminho de SGD. Sem isso, diferencas observadas entre
A/B/C podem vir de cuDNN ou batch ordering, e nao da loss.

Refs:
    Paszke, A. et al. (2019). PyTorch: An Imperative Style, High-Performance
        Deep Learning Library. NeurIPS 32. (§4 — Reproducibility)
    Demsar, J. (2006). Statistical Comparisons of Classifiers over Multiple
        Data Sets. JMLR 7:1-30. (§3.2 — controle experimental)
"""
from __future__ import annotations

import logging
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_global_seeds(seed: int = 42, deterministic: bool = True) -> None:
    """Aplica seed em torch, numpy, random e cudnn.

    Parameters
    ----------
    seed : int, default 42
        Seed compartilhada entre A/B/C (Demsar, 2006).
    deterministic : bool, default True
        Se True, habilita modo deterministico do cuDNN (perda de ~10% de
        throughput em troca de bit-exact reproducibility).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(
        f'Seeds setadas: seed={seed}, deterministic={deterministic}'
    )


def get_rng_states() -> dict:
    """Snapshot dos RNGs atuais para inclusao em checkpoints.

    Captura torch (CPU), torch (CUDA, se disponivel), numpy e Python
    random. Restore com set_rng_states retoma a sequencia exata.
    """
    return {
        'torch': torch.get_rng_state(),
        'cuda': (torch.cuda.get_rng_state_all()
                 if torch.cuda.is_available() else None),
        'numpy': np.random.get_state(),
        'python': random.getstate(),
    }


def set_rng_states(states: dict) -> None:
    """Restaura RNGs a partir de snapshot. Usado em load_checkpoint."""
    torch.set_rng_state(states['torch'])
    if states.get('cuda') is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(states['cuda'])
    np.random.set_state(states['numpy'])
    random.setstate(states['python'])
