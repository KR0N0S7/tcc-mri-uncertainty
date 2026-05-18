# Autor: Massanori
# Data: 17/05/2026
# Descrição: Save/load atomico de checkpoints com optimizer state, scheduler
#            state e RNG states (D5). Recebe: model, optimizer, scheduler,
#            metadata (iteration, best_metric, config). Retorna: arquivo .pt
#            atomico em disco / state restaurado. Escrita atomica via
#            tmp + rename: se o processo cair durante torch.save, o arquivo
#            destino nunca fica parcialmente escrito. Schema versionado
#            (CHECKPOINT_SCHEMA_VERSION) para detectar mismatches.


"""Save/load atomico de checkpoints.

Sem optimizer state, retomar treino corrompe o schedule de lr e os momentos
do AdamW. Sem scheduler state, o warmup recomeca do zero. Sem RNG states,
a ordem dos batches restantes muda.

Escrita atomica via tmp + rename: torch.save escreve em ckpt.pt.tmp, e
so depois renomeia para ckpt.pt. Crash no meio do save -> destino intacto.

Refs:
    Paszke, A. et al. (2019). PyTorch: An Imperative Style, High-Performance
        Deep Learning Library. NeurIPS 32. (§2.5 — checkpointing)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import torch

from src.training.random_seed import get_rng_states, set_rng_states

logger = logging.getLogger(__name__)

# Versao do schema. Incrementar se a estrutura do checkpoint mudar de forma
# incompativel — load_checkpoint rejeita schemas futuros para evitar
# corromper o estado por extrair campos errados.
CHECKPOINT_SCHEMA_VERSION = 1


def save_checkpoint(
    path: Union[str, Path],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    *,
    iteration: int,
    best_metric: float = float('inf'),
    config: Optional[dict] = None,
    extra: Optional[dict] = None,
) -> None:
    """Salva checkpoint atomicamente.

    Parameters
    ----------
    path : str | Path
        Caminho destino. Diretorio pai e criado se nao existir.
    model : nn.Module
    optimizer : torch.optim.Optimizer
    scheduler : LRScheduler or None
        Salvo apenas se diferente de None.
    iteration : int
        Numero da iteracao atual (para retomar).
    best_metric : float, default inf
        Melhor metrica de validacao ate agora (loss, por convencao).
    config : dict or None
        Snapshot da config do treino (auditoria de reprodutibilidade).
    extra : dict or None
        Campos adicionais por grupo (e.g. lambda_lesion no Grupo C).
    """
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    state = {
        'schema_version': CHECKPOINT_SCHEMA_VERSION,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler is not None else None,
        'iteration': iteration,
        'best_metric': best_metric,
        'config': config or {},
        'rng_states': get_rng_states(),
        'extra': extra or {},
    }

    # Escrita atomica: escreve em .tmp, depois renomeia (replace e atomic
    # em todos os FS POSIX e no NTFS). Se torch.save falhar, o destino
    # nao foi tocado.
    tmp_path = path.with_suffix(path.suffix + '.tmp')
    torch.save(state, tmp_path)
    tmp_path.replace(path)
    logger.info(f'Checkpoint salvo: {path.name} (iter={iteration})')


def load_checkpoint(
    path: Union[str, Path],
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    *,
    restore_rng: bool = False,
    map_location: Union[str, torch.device] = 'cpu',
) -> dict:
    """Carrega checkpoint, restaurando estados conforme passado.

    Parameters
    ----------
    path : str | Path
    model : nn.Module
        Estado e carregado in-place via model.load_state_dict.
    optimizer : torch.optim.Optimizer or None
        Se passado, restaura state.
    scheduler : LRScheduler or None
        Se passado E o checkpoint contem scheduler state, restaura.
    restore_rng : bool, default False
        Se True, restaura torch/numpy/random RNGs. Use em resume_from.
    map_location : str | torch.device, default 'cpu'
        Carregar em CPU e mover depois e mais robusto a maquinas sem GPU.

    Returns
    -------
    dict
        Conteudo completo do checkpoint (iteration, best_metric, config, ...).

    Raises
    ------
    FileNotFoundError
        Se path nao existe.
    ValueError
        Se schema_version ausente ou maior que CHECKPOINT_SCHEMA_VERSION.
    """
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Checkpoint nao encontrado: {path}')

    state = torch.load(path, map_location=map_location, weights_only=False)

    if 'schema_version' not in state:
        raise ValueError(
            f'Checkpoint sem schema_version em {path}. '
            f'Formato antigo nao suportado.'
        )
    if state['schema_version'] > CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f'schema_version={state["schema_version"]} > '
            f'CHECKPOINT_SCHEMA_VERSION={CHECKPOINT_SCHEMA_VERSION}. '
            f'Atualize o codigo antes de carregar.'
        )

    model.load_state_dict(state['model'])
    if optimizer is not None and state.get('optimizer'):
        optimizer.load_state_dict(state['optimizer'])
    if scheduler is not None and state.get('scheduler'):
        scheduler.load_state_dict(state['scheduler'])
    if restore_rng and 'rng_states' in state:
        set_rng_states(state['rng_states'])

    logger.info(
        f'Checkpoint carregado: {path.name} (iter={state.get("iteration")})'
    )
    return state
