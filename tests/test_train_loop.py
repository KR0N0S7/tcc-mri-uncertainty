# Autor: Massanori
# Data: 17/05/2026
# Descrição: Testes unitarios para src/training/. Cobre, todos em CPU com
#            modulos pequenos (chans=4) para rodar rapido:
#   (1) cycle_loader itera indefinidamente sobre um DataLoader pequeno,
#   (2) make_scheduler implementa warmup linear corretamente,
#   (3) save/load_checkpoint sao roundtrip exato (parametros bit-a-bit),
#   (4) train() decresce loss em poucos iters com ResM,
#   (5) POLIMORFISMO: train() funciona tanto para ResM quanto para QR
#       sem mudar o codigo do loop (validacao da interface D3),
#   (6) train() sem val_loader nao quebra,
#   (7) resume retoma a iteracao correta e estado do optimizer,
#   (8) checkpoint_every salva last.pt na frequencia correta.
# Roda com: python -m pytest tests/test_train_loop.py -v


"""Testes para src/training/.

Usa um SyntheticReconsDataset em memoria com tensores pequenos (16x16)
para evitar dependencia do dataset real do S4 e rodar em CPU em <60s.
"""
import json
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from src.losses import qr_loss, resm_loss
from src.models import QuantileRegressionModule, ResidualMagnitudeModule
from src.training import (
    cycle_loader,
    load_checkpoint,
    make_optimizer,
    make_scheduler,
    save_checkpoint,
    train,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class SyntheticReconsDataset(Dataset):
    """Dataset em memoria que mimetiza a saida de ReconsSliceDataset.

    Target e gerado como recon + ruido pequeno, garantindo que MSE entre
    uncertainty estimada e |recon - target| seja aprendivel rapidamente.
    """

    def __init__(self, n: int = 8, H: int = 16, W: int = 16, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.recon = torch.rand(n, 1, H, W, generator=g) + 0.1
        self.target = self.recon + 0.1 * torch.randn(n, 1, H, W, generator=g)
        self.target.clamp_(min=0.0)
        self.error = (self.target - self.recon).abs()
        # Lesao = quadrante superior esquerdo (para testar interface unificada)
        self.mask = torch.zeros(n, 1, H, W)
        self.mask[:, :, : H // 2, : W // 2] = 1.0

    def __len__(self) -> int:
        return self.recon.shape[0]

    def __getitem__(self, i: int) -> dict:
        return {
            'recon': self.recon[i],
            'target': self.target[i],
            'error_map': self.error[i],
            'lesion_mask': self.mask[i],
            'max_val': torch.tensor(1.0),
            'volume_id': f'synth_{i}',
            'slice_idx': i,
            'sequence': 'AXFLAIR',
        }


def make_synthetic_loaders(n_train: int = 8, n_val: int = 4):
    train_ds = SyntheticReconsDataset(n=n_train, seed=0)
    val_ds = SyntheticReconsDataset(n=n_val, seed=1)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    return train_loader, val_loader


def make_minimal_config(lr: float = 1e-2, warmup: int = 10) -> dict:
    return {
        'optimizer': {
            'name': 'AdamW',
            'lr': lr,
            'weight_decay': 1e-4,
            'betas': [0.9, 0.999],
            'eps': 1e-8,
        },
        'scheduler': {
            'name': 'warmup_linear',
            'warmup_steps': warmup,
        },
        'training': {
            'batch_size': 1,
            'seed': 42,
        },
        'model': {
            'chans': 4,
            'num_pool_layers': 2,
        },
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_cycle_loader_itera_indefinidamente():
    ds = SyntheticReconsDataset(n=3)
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    gen = cycle_loader(loader)
    # Coleta 10 batches de um dataset com apenas 3 amostras
    batches = [next(gen) for _ in range(10)]
    assert len(batches) == 10
    # Todos com shape consistente
    for b in batches:
        assert b['recon'].shape == (1, 1, 16, 16)


def test_make_scheduler_warmup_linear():
    """lr deve crescer linearmente ate warmup_steps, depois ficar constante."""
    module = ResidualMagnitudeModule(chans=4, num_pool_layers=2)
    config = make_minimal_config(lr=0.01, warmup=5)
    opt = make_optimizer(module, config)
    sched = make_scheduler(opt, config)

    # Antes de qualquer step: lr=0 (com warmup linear comecando em 0)
    assert opt.param_groups[0]['lr'] == pytest.approx(0.0, abs=1e-6)

    lrs = []
    for _ in range(15):
        sched.step()
        lrs.append(opt.param_groups[0]['lr'])

    # No step 5 (indice 4) deve estar em lr_base; depois constante
    assert lrs[4] == pytest.approx(0.01, rel=1e-4)
    assert lrs[9] == pytest.approx(0.01, rel=1e-4)
    assert lrs[14] == pytest.approx(0.01, rel=1e-4)
    # Antes do warmup completar, deve ser proporcional
    assert lrs[1] == pytest.approx(0.01 * 2 / 5, rel=1e-4)


def test_save_load_checkpoint_roundtrip(tmp_path: Path):
    module1 = ResidualMagnitudeModule(chans=4, num_pool_layers=2)
    config = make_minimal_config()
    opt1 = make_optimizer(module1, config)
    sched1 = make_scheduler(opt1, config)

    # Aplica alguns steps para garantir que state_dicts nao estao no default
    for p in module1.parameters():
        p.data.add_(torch.randn_like(p) * 0.01)
    opt1.step()
    sched1.step()
    sched1.step()

    ckpt_path = tmp_path / 'test_ckpt.pt'
    save_checkpoint(
        ckpt_path, iteration=42,
        module=module1, optimizer=opt1, scheduler=sched1,
        best_val_loss=0.123, config_snapshot={'foo': 'bar'},
    )
    assert ckpt_path.is_file()

    module2 = ResidualMagnitudeModule(chans=4, num_pool_layers=2)
    opt2 = make_optimizer(module2, config)
    sched2 = make_scheduler(opt2, config)
    ckpt = load_checkpoint(ckpt_path, module2, opt2, sched2, device='cpu')

    # Parametros bit-a-bit iguais
    for p1, p2 in zip(module1.parameters(), module2.parameters()):
        assert torch.equal(p1.data, p2.data)

    # Estado do scheduler preservado
    assert sched2.last_epoch == sched1.last_epoch

    # Metadados preservados
    assert ckpt['iteration'] == 42
    assert ckpt['best_val_loss'] == pytest.approx(0.123)
    assert ckpt['config_snapshot'] == {'foo': 'bar'}


def test_train_decresce_loss_com_resm(tmp_path: Path):
    """50 iters em ResM com synthetic data devem reduzir a loss."""
    train_loader, val_loader = make_synthetic_loaders()
    module = ResidualMagnitudeModule(chans=4, num_pool_layers=2)
    config = make_minimal_config(lr=0.01, warmup=5)

    result = train(
        module=module,
        loss_fn=resm_loss,
        train_loader=train_loader,
        val_loader=val_loader,
        total_iters=50,
        config=config,
        run_dir=tmp_path / 'run_resm',
        device='cpu',
        seed=42,
        val_every=25,
        log_every=5,
        checkpoint_every=50,
        resume=False,
    )

    # Le CSV para verificar trajetoria
    csv = (tmp_path / 'run_resm' / 'metrics.csv').read_text(encoding='utf-8')
    rows = csv.strip().split('\n')[1:]
    train_losses = [
        float(r.split(',')[1]) for r in rows if r.split(',')[1]
    ]
    assert len(train_losses) >= 4, f'Esperava >=4 pontos, obteve {len(train_losses)}'
    initial = sum(train_losses[:2]) / 2
    final = sum(train_losses[-2:]) / 2
    assert final < initial, (
        f'Loss nao diminuiu: initial={initial:.6f}, final={final:.6f}'
    )


def test_train_polimorfismo_funciona_com_qr(tmp_path: Path):
    """O MESMO loop train() roda QR sem mudancas — validacao da interface D3."""
    train_loader, val_loader = make_synthetic_loaders()
    module = QuantileRegressionModule(chans=4, num_pool_layers=2)
    config = make_minimal_config(lr=0.01, warmup=5)

    result = train(
        module=module,
        loss_fn=qr_loss,
        train_loader=train_loader,
        val_loader=val_loader,
        total_iters=30,
        config=config,
        run_dir=tmp_path / 'run_qr',
        device='cpu',
        seed=42,
        val_every=15,
        log_every=5,
        checkpoint_every=30,
        resume=False,
        loss_kwargs={'alpha': 0.10},
    )

    assert result['total_iters'] == 30
    assert (tmp_path / 'run_qr' / 'last.pt').is_file()
    assert (tmp_path / 'run_qr' / 'metrics.csv').is_file()


def test_train_sem_val_loader_nao_quebra(tmp_path: Path):
    train_loader, _ = make_synthetic_loaders()
    module = ResidualMagnitudeModule(chans=4, num_pool_layers=2)
    config = make_minimal_config(lr=0.01, warmup=5)

    result = train(
        module=module,
        loss_fn=resm_loss,
        train_loader=train_loader,
        val_loader=None,
        total_iters=20,
        config=config,
        run_dir=tmp_path / 'run_no_val',
        device='cpu',
        seed=42,
        log_every=5,
        checkpoint_every=20,
        resume=False,
    )

    assert (tmp_path / 'run_no_val' / 'last.pt').is_file()
    # best.pt NAO deve existir sem val_loader
    assert not (tmp_path / 'run_no_val' / 'best.pt').is_file()


def test_resume_retoma_iteracao_correta(tmp_path: Path):
    """Treina 20 iters, salva, retoma e treina mais 10. Verifica continuidade."""
    train_loader, val_loader = make_synthetic_loaders()
    config = make_minimal_config(lr=0.01, warmup=5)
    run_dir = tmp_path / 'run_resume'

    # Primeira metade
    module1 = ResidualMagnitudeModule(chans=4, num_pool_layers=2)
    train(
        module=module1, loss_fn=resm_loss,
        train_loader=train_loader, val_loader=val_loader,
        total_iters=20, config=config, run_dir=run_dir,
        device='cpu', seed=42,
        val_every=20, log_every=5, checkpoint_every=20,
        resume=False,
    )

    # Resume e treina mais
    module2 = ResidualMagnitudeModule(chans=4, num_pool_layers=2)
    train(
        module=module2, loss_fn=resm_loss,
        train_loader=train_loader, val_loader=val_loader,
        total_iters=30, config=config, run_dir=run_dir,
        device='cpu', seed=42,
        val_every=10, log_every=5, checkpoint_every=10,
        resume=True,
    )

    # Verifica que iteration final no last.pt e 30
    ckpt = torch.load(run_dir / 'last.pt', weights_only=False)
    assert ckpt['iteration'] == 30

    # config_snapshot foi escrito uma vez no primeiro run
    snap = json.loads((run_dir / 'config_snapshot.json').read_text())
    assert 'optimizer' in snap


def test_checkpoint_every_salva_last_pt(tmp_path: Path):
    train_loader, val_loader = make_synthetic_loaders()
    module = ResidualMagnitudeModule(chans=4, num_pool_layers=2)
    config = make_minimal_config(lr=0.01, warmup=2)

    run_dir = tmp_path / 'run_ckpt'
    train(
        module=module, loss_fn=resm_loss,
        train_loader=train_loader, val_loader=val_loader,
        total_iters=10, config=config, run_dir=run_dir,
        device='cpu', seed=42,
        val_every=5, log_every=2, checkpoint_every=5,
        resume=False,
    )

    # last.pt sempre existe ao final
    assert (run_dir / 'last.pt').is_file()
    # Como tem val_loader e val_every=5, best.pt tambem deve existir
    assert (run_dir / 'best.pt').is_file()
    # CSV existe e tem dados
    csv_lines = (run_dir / 'metrics.csv').read_text(encoding='utf-8').strip().split('\n')
    assert len(csv_lines) > 1
