# Autor: Massanori
# Data: 19/05/2026
# Descrição: Testes unitarios para src/calibration/. Cobre, todos em CPU:
#   (1) nonconformity_qr e positivo fora do intervalo, negativo dentro,
#   (2) nonconformity_resm e sempre >= 0,
#   (3) compute_qhat com scores ~ N(0,1) e alpha=0.10 converge para
#       o quantil 90% da distribuicao (Phi^{-1}(0.9) ~= 1.282),
#   (4) compute_qhat com correcao finita: n=10 deve dar quantile_level > 0.9,
#   (5) compute_qhat rejeita alpha invalido ou scores vazio,
#   (6) apply_qhat_qr alarga o intervalo quando qhat>0, estreita quando qhat<0,
#   (7) apply_qhat_resm escala largura por uncertainty * qhat,
#   (8) coverage_stats em caso trivial (cobertura=100%, cobertura=0%),
#   (9) coverage_stats com lesion_mask conta apenas pixels da mascara,
#   (10) calibrate + evaluate end-to-end com dados sinteticos: coverage
#        no test ~= 1 - alpha.
# Roda com: python -m pytest tests/test_calibration.py -v

"""Testes para src/calibration/."""
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from src.calibration import (
    apply_qhat_qr,
    apply_qhat_resm,
    calibrate,
    compute_qhat,
    coverage_stats,
    evaluate,
    nonconformity_qr,
    nonconformity_resm,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class SyntheticQRDataset(Dataset):
    """Dataset onde target ~ N(recon, sigma^2) com intervalo fixo do modulo.

    Usado para testar end-to-end: o modulo \"trained\" gera lower=recon-k e
    upper=recon+k constantes. A cobertura bruta do intervalo [recon-k, recon+k]
    e P(|N(0, sigma^2)| <= k). Calibracao deve ajustar para 1-alpha exatamente.
    """

    def __init__(self, n=200, sigma=0.5, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.recon = torch.rand(n, 1, 8, 8, generator=g) + 0.1
        noise = sigma * torch.randn(n, 1, 8, 8, generator=g)
        self.target = self.recon + noise
        self.mask = torch.zeros(n, 1, 8, 8)
        # Lesao = quadrante superior esquerdo
        self.mask[:, :, :4, :4] = 1.0

    def __len__(self):
        return self.recon.shape[0]

    def __getitem__(self, i):
        return {
            'recon': self.recon[i],
            'target': self.target[i],
            'lesion_mask': self.mask[i],
        }


class ConstantIntervalModule(torch.nn.Module):
    """Modulo dummy que retorna intervalo de largura fixa [recon-k, recon+k].

    Usado para verificar matematicamente o pipeline de calibracao sem
    depender de treino real. O parametro k controla a cobertura bruta.
    """

    def __init__(self, k=0.5):
        super().__init__()
        self.k = k

    def forward(self, recon):
        return {'lower': recon - self.k, 'upper': recon + self.k}


class ConstantUncertaintyModule(torch.nn.Module):
    """Modulo dummy que retorna uncertainty constante c.

    Usado para testar apply_qhat_resm.
    """

    def __init__(self, c=0.3):
        super().__init__()
        self.c = c

    def forward(self, recon):
        return torch.full_like(recon, self.c)


# ---------------------------------------------------------------------------
# nonconformity tests
# ---------------------------------------------------------------------------

def test_nonconformity_qr_positivo_fora_negativo_dentro():
    lower = torch.zeros(2)
    upper = torch.ones(2)
    # target=2.0 esta acima de upper=1.0 -> score = 2.0 - 1.0 = 1.0 > 0
    # target=0.5 esta dentro de [0, 1] -> score = max(0-0.5, 0.5-1) = max(-0.5, -0.5) = -0.5
    target = torch.tensor([2.0, 0.5])
    scores = nonconformity_qr(lower, upper, target)
    assert scores[0].item() == pytest.approx(1.0)
    assert scores[1].item() == pytest.approx(-0.5)


def test_nonconformity_qr_rejeita_shapes_incompativeis():
    with pytest.raises(ValueError, match='Shapes'):
        nonconformity_qr(
            torch.zeros(3), torch.ones(3), torch.zeros(4)
        )


def test_nonconformity_resm_sempre_nao_negativo():
    torch.manual_seed(0)
    u = torch.rand(100) + 0.1
    recon = torch.rand(100)
    target = torch.rand(100)
    scores = nonconformity_resm(u, recon, target)
    assert (scores >= 0).all()


def test_nonconformity_resm_eps_evita_div_zero():
    u = torch.zeros(3)  # forca divisao por zero
    recon = torch.tensor([0.0, 1.0, 2.0])
    target = torch.tensor([0.5, 1.5, 2.5])
    scores = nonconformity_resm(u, recon, target, eps=1e-8)
    # |0.5 - 0| / (0 + 1e-8) = 5e7, gigantesco mas nao inf
    assert torch.isfinite(scores).all()
    assert (scores > 0).all()


# ---------------------------------------------------------------------------
# compute_qhat tests
# ---------------------------------------------------------------------------

def test_compute_qhat_aproxima_quantile_de_distribuicao_conhecida():
    """Para scores ~ N(0, 1) com n grande, qhat(alpha=0.10) ~= Phi^{-1}(0.9) ~= 1.282."""
    torch.manual_seed(42)
    scores = torch.randn(100_000)
    qhat = compute_qhat(scores, alpha=0.10)
    assert abs(qhat - 1.282) < 0.05, f'qhat={qhat:.4f}, esperado ~1.282'


def test_compute_qhat_correcao_finita_com_n_pequeno():
    """Para n=10 e alpha=0.10, quantile_level = 0.9 * 11/10 = 0.99.

    Logo qhat deve estar proximo do MAXIMO dos scores, nao do 90% quantil.
    """
    scores = torch.linspace(0.0, 1.0, 10)  # [0.0, 0.111, ..., 1.0]
    qhat = compute_qhat(scores, alpha=0.10)
    # quantile_level=0.99 sobre [0, 0.111, ..., 1.0]: interpolado entre o 9o e 10o
    # numpy.quantile com method='linear' (default) -> proximo de 1.0
    assert qhat > 0.9, f'qhat={qhat:.3f}, esperado > 0.9 (correcao finita)'


def test_compute_qhat_aceita_numpy_e_tensor():
    x_np = np.array([0.1, 0.5, 0.9, 1.2])
    x_torch = torch.tensor([0.1, 0.5, 0.9, 1.2])
    q_np = compute_qhat(x_np, alpha=0.1)
    q_torch = compute_qhat(x_torch, alpha=0.1)
    assert q_np == pytest.approx(q_torch)


def test_compute_qhat_rejeita_alpha_invalido():
    scores = torch.randn(100)
    with pytest.raises(ValueError, match='alpha'):
        compute_qhat(scores, alpha=0.0)
    with pytest.raises(ValueError, match='alpha'):
        compute_qhat(scores, alpha=1.0)
    with pytest.raises(ValueError, match='alpha'):
        compute_qhat(scores, alpha=-0.1)


def test_compute_qhat_rejeita_scores_vazio():
    with pytest.raises(ValueError, match='vazio'):
        compute_qhat(torch.tensor([]), alpha=0.1)


# ---------------------------------------------------------------------------
# apply_qhat tests
# ---------------------------------------------------------------------------

def test_apply_qhat_qr_alarga_intervalo_quando_qhat_positivo():
    lower = torch.tensor([0.3])
    upper = torch.tensor([0.7])
    l_cal, u_cal = apply_qhat_qr(lower, upper, qhat=0.1)
    assert l_cal.item() == pytest.approx(0.2)
    assert u_cal.item() == pytest.approx(0.8)
    assert (u_cal - l_cal).item() > (upper - lower).item()


def test_apply_qhat_qr_estreita_intervalo_quando_qhat_negativo():
    """Quando o intervalo bruto esta super conservador, qhat<0 e estreita corretamente."""
    lower = torch.tensor([0.1])
    upper = torch.tensor([0.9])
    l_cal, u_cal = apply_qhat_qr(lower, upper, qhat=-0.05)
    assert l_cal.item() == pytest.approx(0.15)
    assert u_cal.item() == pytest.approx(0.85)


def test_apply_qhat_resm_escala_largura_por_uncertainty():
    u = torch.tensor([0.5, 1.0, 2.0])
    recon = torch.tensor([0.3, 0.5, 0.7])
    l_cal, u_cal = apply_qhat_resm(u, recon, qhat=2.0)
    # u=0.5 -> intervalo [0.3-1.0, 0.3+1.0] = [-0.7, 1.3], largura=2.0 = 2*qhat*u
    widths = u_cal - l_cal
    expected_widths = 2 * 2.0 * u  # 2 * qhat * u
    assert torch.allclose(widths, expected_widths)


# ---------------------------------------------------------------------------
# coverage_stats tests
# ---------------------------------------------------------------------------

def test_coverage_stats_cobertura_total():
    """Intervalo gigante -> todos os pixels cobertos."""
    target = torch.rand(4, 1, 4, 4)
    lower_cal = torch.full_like(target, -10.0)
    upper_cal = torch.full_like(target, 10.0)
    stats = coverage_stats(lower_cal, upper_cal, target)
    assert stats['n_total'] == 64
    assert stats['n_covered'] == 64
    assert stats['sum_width'] == pytest.approx(64 * 20.0)


def test_coverage_stats_cobertura_zero():
    """Intervalo deslocado -> 0 pixels cobertos."""
    target = torch.rand(4, 1, 4, 4)  # em [0, 1]
    lower_cal = torch.full_like(target, 5.0)
    upper_cal = torch.full_like(target, 10.0)
    stats = coverage_stats(lower_cal, upper_cal, target)
    assert stats['n_covered'] == 0


def test_coverage_stats_com_mascara_de_lesao():
    target = torch.zeros(1, 1, 4, 4)
    # Intervalo cobre todos os pixels
    lower_cal = torch.full_like(target, -1.0)
    upper_cal = torch.full_like(target, 1.0)
    # Mascara cobre 1/4 dos pixels
    mask = torch.zeros_like(target)
    mask[0, 0, :2, :2] = 1.0  # 4 pixels de 16

    stats = coverage_stats(lower_cal, upper_cal, target, mask)
    assert stats['n_total'] == 16
    assert stats['n_covered'] == 16
    assert stats['n_lesion'] == 4
    assert stats['n_lesion_covered'] == 4


# ---------------------------------------------------------------------------
# Integration tests: calibrate + evaluate
# ---------------------------------------------------------------------------

def test_calibrate_e_evaluate_end_to_end_qr_recupera_cobertura_nominal():
    """Com dados sinteticos exchangeable, coverage(test) ~= 1 - alpha apos calibracao.

    Setup: target ~ N(recon, 0.5^2). O 'modulo' retorna intervalo bruto
    [recon-0.3, recon+0.3] (super estreito, cobertura bruta ~47%).
    Calibracao deve alargar para atingir 90% de cobertura empirica no test.
    """
    torch.manual_seed(123)
    alpha = 0.10
    cal_ds = SyntheticQRDataset(n=400, sigma=0.5, seed=1)
    test_ds = SyntheticQRDataset(n=400, sigma=0.5, seed=2)
    cal_loader = DataLoader(cal_ds, batch_size=4, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False)

    # Modulo gera intervalo estreito demais
    module = ConstantIntervalModule(k=0.3)

    # Calibracao
    qhat, scores = calibrate(module, cal_loader, alpha=alpha, kind='qr', device='cpu')
    assert qhat > 0, 'qhat deve ser positivo (intervalo estava estreito demais)'

    # Avaliacao
    metrics = evaluate(module, test_loader, qhat, kind='qr', device='cpu')

    # Cobertura empirica deve estar proxima de 1-alpha = 0.9
    # Tolerancia: 4 desvios padrao binomiais com n=400*64=25600 pixels e p=0.9 ~ 0.0075
    assert abs(metrics['coverage_global'] - (1 - alpha)) < 0.03, (
        f"coverage_global={metrics['coverage_global']:.4f}, esperado ~{1-alpha}"
    )


def test_calibrate_resm_termina_e_retorna_qhat_finito():
    """Sanidade end-to-end para o caminho ResM (calibration multiplicativa)."""
    torch.manual_seed(0)
    ds = SyntheticQRDataset(n=50, sigma=0.3, seed=0)
    loader = DataLoader(ds, batch_size=4, shuffle=False)
    module = ConstantUncertaintyModule(c=0.3)

    qhat, scores = calibrate(module, loader, alpha=0.10, kind='resm', device='cpu')
    assert isinstance(qhat, float)
    assert qhat > 0 and qhat < 100  # range razoavel

    metrics = evaluate(module, loader, qhat, kind='resm', device='cpu')
    assert 'coverage_global' in metrics
    assert 'mean_width' in metrics
    assert metrics['mean_width'] > 0


def test_kind_invalido_levanta_erro():
    ds = SyntheticQRDataset(n=10)
    loader = DataLoader(ds, batch_size=2)
    module = ConstantIntervalModule()
    with pytest.raises(ValueError, match='kind'):
        calibrate(module, loader, alpha=0.1, kind='invalid', device='cpu')
    with pytest.raises(ValueError, match='kind'):
        evaluate(module, loader, qhat=0.5, kind='invalid', device='cpu')
