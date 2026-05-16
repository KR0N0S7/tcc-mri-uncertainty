# Autor: Massanori
# Data: 14/05/2026
# Descrição: Carregamento determinístico da E2E-VarNet pré-treinada do fastMRI
#            oficial (Sriram et al., 2020). Recebe: caminho do checkpoint .pt;
#            opcionalmente os 5 hiperparâmetros da arquitetura para validação
#            via strict load_state_dict. Retorna: instância VarNet em modo
#            eval com gradientes desabilitados, mais SHA-256 do checkpoint
#            para auditoria de reprodutibilidade no TCC. Trata os formatos
#            de checkpoint comuns: state_dict puro (formato do
#            brain_leaderboard_state_dict.pt oficial), dict com chave
#            'state_dict' (formato PyTorch Lightning) e prefixos comuns
#            ('varnet.', 'model.', 'module.') que aparecem em exports
#            do Lightning ou DistributedDataParallel. Protegido por testes
#            em tests/test_varnet_loader.py. Hiperparametros confirmados
#            contra fastmri==0.3.0 e contra run_pretrained_varnet_inference.py
#            do repo facebookresearch/fastMRI (commit 91f2df47).


"""Carregamento da E2E-VarNet pre-treinada do fastMRI oficial.

Refs:
    Sriram, A. et al. (2020). End-to-End Variational Networks for
        Accelerated MRI Reconstruction. MICCAI.
        https://arxiv.org/abs/2004.06688
    fastMRI repo oficial (arquivado em ago/2025):
        https://github.com/facebookresearch/fastMRI
    Checkpoint brain 4x oficial:
        https://dl.fbaipublicfiles.com/fastMRI/trained_models/varnet/brain_leaderboard_state_dict.pt
"""
from __future__ import annotations
import hashlib
import logging
from pathlib import Path
from typing import Tuple, Union

import torch
from fastmri.models import VarNet

logger = logging.getLogger(__name__)

# Hiperparametros do checkpoint brain 4x oficial. Conferidos contra a linha 63
# de fastmri_examples/varnet/run_pretrained_varnet_inference.py em
# facebookresearch/fastMRI@91f2df47. Sao os 5 args necessarios para que
# load_state_dict com strict=True funcione no checkpoint oficial.
DEFAULT_NUM_CASCADES = 12
DEFAULT_POOLS = 4
DEFAULT_VARNET_CHANS = 18
DEFAULT_SENS_POOLS = 4
DEFAULT_SENS_CHANS = 8

# URL do checkpoint brain 4x oficial. Hospedado pela Facebook AI Research.
CHECKPOINT_URL = (
    'https://dl.fbaipublicfiles.com/fastMRI/trained_models/varnet/'
    'brain_leaderboard_state_dict.pt'
)

# Prefixos que aparecem em checkpoints exportados via PyTorch Lightning
# (LightningModule adiciona o nome do atributo como prefix) ou via DDP
# (que adiciona 'module.'). O checkpoint oficial NAO usa prefixo, mas
# este loader e robusto a re-empacotamentos comuns.
_KNOWN_PREFIXES = ('varnet.', 'model.', 'module.')


def compute_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 streaming, para checkpoints de centenas de MB sem estourar RAM.

    Parameters
    ----------
    path : Path
        Caminho do arquivo a hashear.
    chunk_size : int, default 1 MiB
        Tamanho do bloco de leitura.

    Returns
    -------
    str
        Hash hexadecimal de 64 caracteres.
    """
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def _strip_prefix(state_dict: dict, prefix: str) -> dict:
    """Remove prefixo do inicio das chaves do state_dict, se presente em
    pelo menos uma chave. Operacao idempotente: se nenhum prefixo bate,
    retorna o dict original sem copia.
    """
    if not any(k.startswith(prefix) for k in state_dict):
        return state_dict
    n = len(prefix)
    return {(k[n:] if k.startswith(prefix) else k): v for k, v in state_dict.items()}


def _extract_state_dict(raw: object, source: Path) -> dict:
    """Lida com os 2 formatos comuns de checkpoint:
    - state_dict puro: {layer.weight: tensor, ...} (formato oficial)
    - dict Lightning: {'state_dict': {...}, 'epoch': ..., 'optimizer': ...}
    """
    if isinstance(raw, dict) and 'state_dict' in raw:
        sd = raw['state_dict']
        if not isinstance(sd, dict):
            raise ValueError(
                f'Em {source}: chave "state_dict" existe mas nao e dict.'
            )
        return sd
    if isinstance(raw, dict) and all(isinstance(k, str) for k in raw):
        return raw
    raise ValueError(
        f'Formato de checkpoint desconhecido em {source}. '
        f'Esperado dict ou dict com chave "state_dict".'
    )


def load_pretrained_varnet(
    checkpoint_path: Union[str, Path],
    num_cascades: int = DEFAULT_NUM_CASCADES,
    pools: int = DEFAULT_POOLS,
    chans: int = DEFAULT_VARNET_CHANS,
    sens_pools: int = DEFAULT_SENS_POOLS,
    sens_chans: int = DEFAULT_SENS_CHANS,
    device: Union[str, torch.device] = 'cpu',
    strict: bool = True,
) -> Tuple[VarNet, str]:
    """Carrega a E2E-VarNet pre-treinada do fastMRI para inferencia apenas.

    O modelo retorna em modo eval com requires_grad=False em todos os
    parametros. NAO use este loader para fine-tuning: refaca o ciclo
    de gradientes manualmente se for esse o caso.

    Parameters
    ----------
    checkpoint_path : str | Path
        Caminho para o checkpoint .pt do fastMRI brain 4x oficial.
    num_cascades : int, default 12
        Numero de cascadas da VarNet. Fixo para o checkpoint oficial.
    pools : int, default 4
        Numero de pools da U-Net interna das cascadas.
    chans : int, default 18
        Numero de canais da U-Net interna das cascadas.
    sens_pools : int, default 4
        Numero de pools da U-Net de sensitivity map estimation.
    sens_chans : int, default 8
        Numero de canais da U-Net de sensitivity map estimation.
    device : str | torch.device, default 'cpu'
        Dispositivo final do modelo. Carregar em CPU primeiro e mover
        depois permite usar este loader em maquinas sem GPU durante
        desenvolvimento (debug local).
    strict : bool, default True
        Repassado para nn.Module.load_state_dict. Mantenha True para
        detectar mismatch de chaves cedo: e a salvaguarda principal
        contra carregar o checkpoint errado ou com hiperparametros errados.

    Returns
    -------
    model : fastmri.models.VarNet
        Modelo em modo eval, sem grad, no device solicitado.
    sha256 : str
        Hash hexadecimal do checkpoint para registro no MANIFEST do TCC.

    Raises
    ------
    FileNotFoundError
        Se checkpoint_path nao aponta para um arquivo existente.
    RuntimeError
        Se strict=True e ha mismatch entre state_dict e a arquitetura
        instanciada (provavel causa: hiperparametros errados).
    ValueError
        Se o conteudo do .pt nao e um state_dict reconhecivel.
    """
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f'Checkpoint nao encontrado: {checkpoint_path}')

    sha256 = compute_sha256(checkpoint_path)
    logger.info(f'Checkpoint: {checkpoint_path.name}  SHA-256: {sha256}')

    # map_location='cpu' garante carregamento em maquinas sem GPU. Mover
    # para o device final acontece no final, depois do load_state_dict.
    # weights_only=True (PyTorch >=2.0) recusa execucao arbitraria
    # durante o unpickle, blindando contra checkpoints maliciosos.
    try:
        raw = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    except TypeError:
        # PyTorch <2.0 nao tem weights_only; cai no comportamento default.
        raw = torch.load(checkpoint_path, map_location='cpu')

    state_dict = _extract_state_dict(raw, checkpoint_path)

    # Aplica os strips em sequencia. Cada um e idempotente, entao a ordem
    # nao importa contanto que todos sejam tentados.
    for prefix in _KNOWN_PREFIXES:
        state_dict = _strip_prefix(state_dict, prefix)

    model = VarNet(
        num_cascades=num_cascades,
        pools=pools,
        chans=chans,
        sens_pools=sens_pools,
        sens_chans=sens_chans,
    )
    # Com strict=True, load_state_dict levanta RuntimeError direto.
    # Com strict=False, ele retorna IncompatibleKeys(missing, unexpected)
    # e cabe a este loader logar diagnostico.
    result = model.load_state_dict(state_dict, strict=strict)
    missing = getattr(result, 'missing_keys', [])
    unexpected = getattr(result, 'unexpected_keys', [])
    if missing:
        logger.warning(
            f'load_state_dict: {len(missing)} chaves faltando '
            f'(primeiras: {missing[:3]})'
        )
    if unexpected:
        logger.warning(
            f'load_state_dict: {len(unexpected)} chaves inesperadas '
            f'(primeiras: {unexpected[:3]})'
        )

    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    model = model.to(device)
    return model, sha256
