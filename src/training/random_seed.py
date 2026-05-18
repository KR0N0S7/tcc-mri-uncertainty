# Autor: Massanori
# Data: 18/05/2026
# Descrição: Setup determinístico de RNGs (Python random, numpy, torch CPU e
#            CUDA) e flags do cuDNN/CUBLAS. Recebe: int seed e flag
#            deterministic. Retorna: None (efeito colateral global).
#            Chamada uma vez no inicio de cada run de treino, com o mesmo
#            valor de seed entre Grupos A/B/C para garantir que diferencas
#            observadas nas metricas venham da loss, nao da inicializacao
#            (Demsar, 2006, secao 3.2).


"""Setup de seeds e flags para reprodutibilidade.

Reprodutibilidade em deep learning depende de tres frentes simultaneas:
    1. RNGs de Python (random.seed), numpy (np.random.seed) e torch
       (torch.manual_seed + torch.cuda.manual_seed_all)
    2. Algoritmos determinísticos do cuDNN (deterministic=True, benchmark=False)
    3. CUBLAS workspace para multiplicacoes determinísticas em CUDA >=10.2

Falhar em qualquer uma destas tres faz com que dois runs com mesma seed
produzam resultados diferentes em pequena magnitude (1e-5 a 1e-3), o
suficiente para invalidar testes estatisticos pareados entre grupos
(Paszke et al., 2019, secao 4).

Refs:
    Paszke, A. et al. (2019). PyTorch: An Imperative Style,
        High-Performance Deep Learning Library. NeurIPS 32.
    Demsar, J. (2006). Statistical Comparisons of Classifiers over Multiple
        Data Sets. JMLR 7:1-30.
    NVIDIA. CUDA Toolkit Documentation - cuBLAS reproducibility.
        https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
"""
from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_global_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Define seeds globais para reprodutibilidade.

    Parameters
    ----------
    seed : int, default 42
        Seed compartilhada por todos os RNGs (Python random, numpy, torch).
        Use o MESMO valor entre Grupos A/B/C para isolar o efeito da loss.
    deterministic : bool, default True
        Se True, ativa cuDNN.deterministic e desativa benchmark, alem de
        configurar CUBLAS_WORKSPACE_CONFIG. Mais lento (~10-20% overhead em
        T4), mas reprodutivel. Mantenha True para o TCC.

    Notes
    -----
    A flag CUBLAS_WORKSPACE_CONFIG=:4096:8 e necessaria em CUDA >=10.2 para
    que operacoes de matmul/conv2d em float32 sejam determinísticas. Definir
    aqui em vez de exigir o usuario configurar no shell e mais seguro contra
    esquecimentos.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Necessario para multiplicacoes determinísticas em CUDA >=10.2.
        # NVIDIA documenta dois valores aceitos (:4096:8 ou :16:8); :4096:8
        # usa mais memoria mas tem melhor performance.
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

    logger.info(
        f'Seed global = {seed}, deterministic={deterministic}, '
        f'cuda_available={torch.cuda.is_available()}'
    )
