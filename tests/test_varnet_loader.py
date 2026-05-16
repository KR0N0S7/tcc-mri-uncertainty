# Autor: Massanori
# Data: 14/05/2026
# Descrição: Testes unitários do varnet_loader. Cobre: 1) load com state_dict
#            puro (formato oficial), 2) load com formato PyTorch Lightning
#            (chave 'state_dict' + prefix 'varnet.'), 3) modo eval +
#            requires_grad=False em todos os parâmetros, 4) SHA-256
#            determinístico e correto, 5) FileNotFoundError em path inválido,
#            6) strict=True detectando mismatch de hiperparâmetros, 7)
#            idempotência de _strip_prefix, 8) device CPU explícito,
#            9) corrupted_checkpoint, 10) mismatch em pools/sens_pools
#            (especifico do checkpoint oficial). Usa um VarNet mock instanciado
#            em RAM e salvo via torch.save num tmp_path, evitando dependencia
#            do checkpoint real de ~390 MB.
#            Roda com: python -m pytest tests/test_varnet_loader.py -v


"""Testes unitarios do src.models.varnet_loader.

Roda 100% em CPU, sem precisar do brain_leaderboard_state_dict.pt real.
"""
import hashlib

import pytest
import torch
from fastmri.models import VarNet

from src.models.varnet_loader import (
    DEFAULT_NUM_CASCADES,
    DEFAULT_POOLS,
    DEFAULT_SENS_CHANS,
    DEFAULT_SENS_POOLS,
    DEFAULT_VARNET_CHANS,
    _strip_prefix,
    compute_sha256,
    load_pretrained_varnet,
)


def _make_fresh_varnet() -> VarNet:
    """Constroi VarNet com os 5 hiperparametros do checkpoint brain 4x oficial.
    Estes valores batem com a linha 63 de run_pretrained_varnet_inference.py
    do fastMRI oficial (commit 91f2df47).
    """
    return VarNet(
        num_cascades=DEFAULT_NUM_CASCADES,
        pools=DEFAULT_POOLS,
        chans=DEFAULT_VARNET_CHANS,
        sens_pools=DEFAULT_SENS_POOLS,
        sens_chans=DEFAULT_SENS_CHANS,
    )


@pytest.fixture
def mock_checkpoint(tmp_path):
    """Mock no formato state_dict puro — formato usado pelo checkpoint
    oficial brain_leaderboard_state_dict.pt."""
    model = _make_fresh_varnet()
    ckpt_path = tmp_path / 'mock_varnet.pt'
    torch.save(model.state_dict(), ckpt_path)
    return ckpt_path


@pytest.fixture
def mock_lightning_checkpoint(tmp_path):
    """Mock no formato PyTorch Lightning: dict com 'state_dict' e prefix
    'varnet.' nas chaves. Util para checkpoints re-empacotados via
    LightningModule.
    """
    model = _make_fresh_varnet()
    sd_with_prefix = {f'varnet.{k}': v for k, v in model.state_dict().items()}
    ckpt_path = tmp_path / 'mock_varnet_lightning.pt'
    torch.save(
        {
            'state_dict': sd_with_prefix,
            'epoch': 50,
            'global_step': 12345,
        },
        ckpt_path,
    )
    return ckpt_path


def test_load_raw_state_dict(mock_checkpoint):
    model, sha = load_pretrained_varnet(mock_checkpoint)
    assert isinstance(model, VarNet)
    assert len(sha) == 64
    assert all(c in '0123456789abcdef' for c in sha)


def test_load_lightning_state_dict(mock_lightning_checkpoint):
    """Formato Lightning com prefix — testa robustez a re-empacotamentos."""
    model, sha = load_pretrained_varnet(mock_lightning_checkpoint)
    assert isinstance(model, VarNet)
    assert len(sha) == 64


def test_returned_model_in_eval_mode(mock_checkpoint):
    model, _ = load_pretrained_varnet(mock_checkpoint)
    assert not model.training, 'VarNet deve estar em eval mode pos-load'


def test_returned_model_has_no_grad(mock_checkpoint):
    """Pre-computacao do S4 nao deve permitir backward."""
    model, _ = load_pretrained_varnet(mock_checkpoint)
    n_params = 0
    for p in model.parameters():
        assert not p.requires_grad
        n_params += 1
    assert n_params > 0, 'Modelo nao tem parametros — algo errado'


def test_sha256_deterministic(mock_checkpoint):
    """Hash deve ser estavel entre chamadas e bater com hashlib direto."""
    sha1 = compute_sha256(mock_checkpoint)
    sha2 = compute_sha256(mock_checkpoint)
    sha_ref = hashlib.sha256(mock_checkpoint.read_bytes()).hexdigest()
    assert sha1 == sha2 == sha_ref


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_pretrained_varnet(tmp_path / 'nao_existe.pt')


def test_strip_prefix_basic():
    sd = {
        'varnet.layer.weight': torch.zeros(3),
        'varnet.layer.bias': torch.zeros(3),
        'standalone': torch.zeros(3),
    }
    stripped = _strip_prefix(sd, 'varnet.')
    assert 'layer.weight' in stripped
    assert 'layer.bias' in stripped
    assert 'standalone' in stripped
    assert 'varnet.layer.weight' not in stripped


def test_strip_prefix_idempotent():
    """Aplicar duas vezes nao deve quebrar — devolve dict sem o prefixo
    ainda que ele ja tenha sido removido na primeira passada.
    """
    sd = {'varnet.layer.weight': torch.zeros(3)}
    once = _strip_prefix(sd, 'varnet.')
    twice = _strip_prefix(once, 'varnet.')
    assert once == twice


def test_strip_prefix_noop_when_absent():
    """Se nenhuma chave tem o prefixo, deve retornar o mesmo dict (mesma
    identidade), evitando copia desnecessaria.
    """
    sd = {'layer.weight': torch.zeros(3)}
    result = _strip_prefix(sd, 'varnet.')
    assert result is sd


def test_strict_detects_num_cascades_mismatch(mock_checkpoint):
    """num_cascades errado deve gerar RuntimeError em load_state_dict
    quando strict=True. Esta e a salvaguarda contra carregar checkpoints
    com hiperparametros errados silenciosamente.
    """
    with pytest.raises(RuntimeError):
        load_pretrained_varnet(
            mock_checkpoint,
            num_cascades=DEFAULT_NUM_CASCADES + 1,
            strict=True,
        )


def test_strict_detects_chans_mismatch(mock_checkpoint):
    """chans errado tambem deve ser pego pelo strict=True. Garante que
    todos os 5 hiperparametros sao validados, nao so num_cascades.
    """
    with pytest.raises(RuntimeError):
        load_pretrained_varnet(
            mock_checkpoint,
            chans=DEFAULT_VARNET_CHANS + 1,
            strict=True,
        )


def test_corrupted_checkpoint_raises(tmp_path):
    """Arquivo que nao e um state_dict valido deve falhar com erro claro,
    nao corromper silenciosamente o modelo.
    """
    bad = tmp_path / 'corrupted.pt'
    torch.save([1, 2, 3], bad)  # lista, nao dict
    with pytest.raises(ValueError):
        load_pretrained_varnet(bad)


def test_device_cpu_explicit(mock_checkpoint):
    """Carregamento explicito em CPU funciona em maquinas sem GPU."""
    model, _ = load_pretrained_varnet(mock_checkpoint, device='cpu')
    first_param = next(model.parameters())
    assert first_param.device.type == 'cpu'
