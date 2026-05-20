# Autor: Massanori
# Data: 19/05/2026 (mod 20/05/2026: migrados 3 testes uteis do test_calibration.py)
# Descricao: Testes unitarios para src/calibration/conformal.py. Cobre:
#   (1) cqr_score: sinal correto (positivo se y fora, negativo se y dentro),
#   (2) scaled_cp_score: divisao por uncertainty + eps,
#   (3) conformal_quantile: correcao finite-sample (1-alpha)(n+1)/n,
#   (4) calibrate_qr: end-to-end com modulo fake + dataset sintetico,
#   (5) calibrate_resm: idem,
#   (6) GOLD STANDARD: calibrate -> apply -> cobertura empirica >= 1-alpha
#       em dados sinteticos com distribuicao conhecida (validacao da
#       garantia formal de Romano et al., 2019, Teorema 1),
#   (7) apply_cqr_interval e apply_resm_interval: aritmetica correta,
#   (8) erros claros para inputs invalidos,
#   (9) [migrados] convergencia para Phi^-1(0.9) com N(0,1), correcao
#       finite-sample para n pequeno, q_hat negativo para intervalo
#       conservador demais.
# Roda com: python -m pytest tests/test_conformal.py -v


"""Testes para src/calibration/conformal.py.

O teste mais critico e test_cqr_calibrate_then_apply_atinge_cobertura_marginal:
ele instancia um modulo fake com intervalos propositalmente estreitos
(undercovered), calibra com cal split sintetico, aplica ao test split,
e verifica que a cobertura empirica e >= 1 - alpha. Esse e o teste do
teorema de Romano et al. (2019, Teorema 1).
"""
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from src.calibration import (
    apply_cqr_interval,
    apply_resm_interval,
    calibrate_qr,
    calibrate_resm,
    conformal_quantile,
    cqr_score,
    scaled_cp_score,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class _SyntheticReconDS(Dataset):
    """Dataset sintetico para testar calibracao end-to-end.

    target = recon + noise. O scale do noise controla quao 'undercovered'
    o intervalo predito esta.
    """

    def __init__(self, n: int, noise_std: float = 0.3, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.recon = torch.rand(n, 1, 8, 8, generator=g)
        self.target = self.recon + noise_std * torch.randn(
            n, 1, 8, 8, generator=g,
        )

    def __len__(self) -> int:
        return self.recon.shape[0]

    def __getitem__(self, i: int) -> dict:
        return {
            'recon': self.recon[i],
            'target': self.target[i],
            'lesion_mask': torch.zeros_like(self.recon[i]),
            'max_val': torch.tensor(1.0),
            'volume_id': f'synth_{i}',
            'slice_idx': i,
            'sequence': 'AXFLAIR',
        }


class _FakeQRModule(torch.nn.Module):
    """Modulo QR que retorna intervalo [recon - h, recon + h], h fixo."""

    def __init__(self, half_width: float = 0.1):
        super().__init__()
        self.h = half_width

    def forward(self, recon):
        return {'lower': recon - self.h, 'upper': recon + self.h}


class _FakeResmModule(torch.nn.Module):
    """Modulo ResM que retorna u(x) constante."""

    def __init__(self, u: float = 0.1):
        super().__init__()
        self.u = u

    def forward(self, recon):
        return torch.full_like(recon, self.u)


# ---------------------------------------------------------------------------
# Tests: scores
# ---------------------------------------------------------------------------

def test_cqr_score_negativo_quando_target_dentro_do_intervalo():
    lower = torch.tensor([0.0, 0.0])
    upper = torch.tensor([1.0, 1.0])
    target = torch.tensor([0.3, 0.7])
    scores = cqr_score(lower, upper, target)
    assert (scores < 0).all()


def test_cqr_score_positivo_quando_target_abaixo_do_lower():
    lower = torch.tensor([1.0])
    upper = torch.tensor([2.0])
    target = torch.tensor([0.5])
    scores = cqr_score(lower, upper, target)
    # lower - target = 1.0 - 0.5 = 0.5
    assert torch.allclose(scores, torch.tensor([0.5]))


def test_cqr_score_positivo_quando_target_acima_do_upper():
    lower = torch.tensor([0.0])
    upper = torch.tensor([1.0])
    target = torch.tensor([1.5])
    scores = cqr_score(lower, upper, target)
    # target - upper = 1.5 - 1.0 = 0.5
    assert torch.allclose(scores, torch.tensor([0.5]))


def test_cqr_score_zero_no_boundary():
    lower = torch.tensor([0.0, 0.0])
    upper = torch.tensor([1.0, 1.0])
    # target exatamente nos limites
    target = torch.tensor([0.0, 1.0])
    scores = cqr_score(lower, upper, target)
    assert torch.allclose(scores, torch.zeros(2), atol=1e-6)


def test_cqr_score_rejeita_shapes_incompativeis():
    with pytest.raises(ValueError, match='Shapes incompativeis'):
        cqr_score(torch.zeros(2), torch.ones(2), torch.zeros(3))


def test_scaled_cp_score_divisao_basica():
    uncertainty = torch.tensor([1.0, 2.0, 0.5])
    recon = torch.tensor([0.0, 0.0, 0.0])
    target = torch.tensor([2.0, 4.0, 1.0])
    scores = scaled_cp_score(uncertainty, recon, target)
    # |y - x| / u = [2.0/1.0, 4.0/2.0, 1.0/0.5] = [2, 2, 2]
    assert torch.allclose(scores, torch.tensor([2.0, 2.0, 2.0]), atol=1e-5)


def test_scaled_cp_score_eps_evita_divisao_por_zero():
    uncertainty = torch.tensor([0.0])
    recon = torch.tensor([0.0])
    target = torch.tensor([1.0])
    scores = scaled_cp_score(uncertainty, recon, target, eps=1e-6)
    # 1.0 / 1e-6 = 1e6 (grande mas finito)
    assert torch.isfinite(scores).all()
    assert scores.item() > 1e5


# ---------------------------------------------------------------------------
# Tests: quantile
# ---------------------------------------------------------------------------

def test_conformal_quantile_basico():
    # 100 valores 1..100; quantile 90 deve estar perto de 90
    scores = torch.arange(1, 101).float()
    q = conformal_quantile(scores, alpha=0.10)
    # (1-0.10)(101)/100 = 0.909 -> ~91
    assert 89.0 <= q <= 92.0


def test_conformal_quantile_rejeita_alpha_fora_de_0_1():
    scores = torch.rand(100)
    with pytest.raises(ValueError, match='alpha'):
        conformal_quantile(scores, alpha=0.0)
    with pytest.raises(ValueError, match='alpha'):
        conformal_quantile(scores, alpha=1.0)


def test_conformal_quantile_rejeita_tensor_vazio():
    with pytest.raises(ValueError, match='vazio'):
        conformal_quantile(torch.zeros(0), alpha=0.10)


def test_conformal_quantile_aproxima_phi_inverse_para_n_grande():
    """Para scores ~ N(0, 1) com n >> 1, q -> Phi^-1(1 - alpha).

    Valida que a implementacao converge para o quantile teorico de uma
    distribuicao conhecida quando o numero de amostras e grande (a
    correcao finite-sample (n+1)/n ~= 1). Phi^-1(0.9) ~= 1.28155.

    Migrado do test_calibration.py legado (commit ae897c3 refactor).
    """
    torch.manual_seed(42)
    scores = torch.randn(100_000)
    q = conformal_quantile(scores, alpha=0.10)
    # Phi^-1(0.9) ~= 1.2815515655
    assert abs(q - 1.282) < 0.05, f'q_hat={q:.4f}, esperado ~1.282 (Phi^-1(0.9))'


def test_conformal_quantile_correcao_finite_sample_com_n_pequeno():
    """Para n=10 e alpha=0.10, q_level = 0.9 * 11/10 = 0.99.

    A correcao (n+1)/n eleva o quantile efetivo. Sobre scores uniformes
    em [0, 1] com n=10, q_hat deve estar bem proximo de 1.0 (perto do
    maximo), nao de 0.9. Isso garante cobertura formal sob exchangeability
    (Romano et al., 2019, Teorema 1).

    Migrado do test_calibration.py legado.
    """
    scores = torch.linspace(0.0, 1.0, 10)  # [0.0, 0.111, ..., 1.0]
    q = conformal_quantile(scores, alpha=0.10)
    assert q > 0.9, f'q_hat={q:.3f}, esperado > 0.9 (correcao finite-sample)'


# ---------------------------------------------------------------------------
# Tests: calibrate end-to-end
# ---------------------------------------------------------------------------

def test_calibrate_qr_retorna_metadata_completa():
    ds = _SyntheticReconDS(n=20, seed=0)
    loader = DataLoader(ds, batch_size=1)
    module = _FakeQRModule(half_width=0.1)
    result = calibrate_qr(module, loader, alpha=0.10, device='cpu')
    assert set(result.keys()) >= {
        'q_hat', 'n_pixels', 'n_batches', 'alpha', 'mean_score', 'method'
    }
    assert result['n_batches'] == 20
    assert result['n_pixels'] == 20 * 1 * 8 * 8
    assert result['alpha'] == 0.10
    assert result['method'] == 'CQR'
    # q_hat deve ser positivo: intervalo proposital estreito (h=0.1, noise=0.3)
    assert result['q_hat'] > 0


def test_calibrate_resm_retorna_metadata_completa():
    ds = _SyntheticReconDS(n=20, seed=1)
    loader = DataLoader(ds, batch_size=1)
    module = _FakeResmModule(u=0.1)
    result = calibrate_resm(module, loader, alpha=0.10, device='cpu')
    assert result['n_batches'] == 20
    assert result['method'] == 'ScaledCP'
    assert result['q_hat'] > 0


# ---------------------------------------------------------------------------
# GOLD STANDARD: cobertura empirica post-calibracao >= 1 - alpha
# ---------------------------------------------------------------------------

def test_cqr_calibrate_then_apply_atinge_cobertura_marginal():
    """Teste central da implementacao: o Teorema de Romano et al. (2019).

    Modulo proposital estreito (h=0.05) sobre dados ruidosos (noise=0.4).
    Apos calibrar em cal_loader e aplicar em test_loader, a cobertura
    empirica pixelwise deve atingir ~ 1 - alpha. Tolerancia ampla por
    causa de variabilidade de amostras pequenas.
    """
    torch.manual_seed(42)
    alpha = 0.10

    # Cal e test sao IID: mesma noise_std, seeds diferentes
    cal_ds = _SyntheticReconDS(n=100, noise_std=0.4, seed=0)
    test_ds = _SyntheticReconDS(n=200, noise_std=0.4, seed=1)
    cal_loader = DataLoader(cal_ds, batch_size=1)
    test_loader = DataLoader(test_ds, batch_size=1)

    module = _FakeQRModule(half_width=0.05)

    # Calibrate
    cal_result = calibrate_qr(module, cal_loader, alpha=alpha, device='cpu')
    q_hat = cal_result['q_hat']

    # Apply em test e mede cobertura
    n_covered = 0
    n_total = 0
    with torch.no_grad():
        for batch in test_loader:
            pred = module(batch['recon'])
            lower_cal, upper_cal = apply_cqr_interval(
                pred['lower'], pred['upper'], q_hat,
            )
            inside = (
                (batch['target'] >= lower_cal)
                & (batch['target'] <= upper_cal)
            )
            n_covered += int(inside.sum().item())
            n_total += int(batch['target'].numel())

    coverage = n_covered / n_total
    # Esperado: ~ 1 - alpha = 0.90. Margem ampla para variancia de amostras
    # pequenas (200 * 64 = 12800 pixels).
    assert 0.85 <= coverage <= 0.97, (
        f'Cobertura {coverage:.3f} fora do esperado [0.85, 0.97] '
        f'para 1-alpha = {1-alpha}'
    )


def test_resm_calibrate_then_apply_atinge_cobertura_marginal():
    """Mesmo teste para o caminho ResM (scaled CP locally adaptive)."""
    torch.manual_seed(43)
    alpha = 0.10

    cal_ds = _SyntheticReconDS(n=100, noise_std=0.4, seed=2)
    test_ds = _SyntheticReconDS(n=200, noise_std=0.4, seed=3)
    cal_loader = DataLoader(cal_ds, batch_size=1)
    test_loader = DataLoader(test_ds, batch_size=1)

    module = _FakeResmModule(u=0.1)

    cal_result = calibrate_resm(module, cal_loader, alpha=alpha, device='cpu')
    q_hat = cal_result['q_hat']

    n_covered = 0
    n_total = 0
    with torch.no_grad():
        for batch in test_loader:
            recon = batch['recon']
            target = batch['target']
            u = module(recon)
            lower_cal, upper_cal = apply_resm_interval(recon, u, q_hat)
            inside = (target >= lower_cal) & (target <= upper_cal)
            n_covered += int(inside.sum().item())
            n_total += int(target.numel())

    coverage = n_covered / n_total
    assert 0.85 <= coverage <= 0.97, (
        f'Cobertura ResM {coverage:.3f} fora do esperado [0.85, 0.97]'
    )


# ---------------------------------------------------------------------------
# Tests: apply
# ---------------------------------------------------------------------------

def test_apply_cqr_interval_aritmetica():
    lower = torch.tensor([0.0, 1.0])
    upper = torch.tensor([1.0, 2.0])
    q_hat = 0.5
    l, u = apply_cqr_interval(lower, upper, q_hat)
    assert torch.allclose(l, torch.tensor([-0.5, 0.5]))
    assert torch.allclose(u, torch.tensor([1.5, 2.5]))


def test_apply_cqr_interval_aceita_q_hat_negativo():
    """Caso degenerado: intervalo bruto super conservador, calibracao estreita.

    Se o modulo treinado sai com intervalos largos demais (overcovered),
    q_hat sai negativo e o intervalo calibrado [lower - q_hat, upper + q_hat]
    fica MAIS ESTREITO. Aritmetica permanece correta.

    Migrado do test_calibration.py legado.
    """
    lower = torch.tensor([0.1])
    upper = torch.tensor([0.9])
    q_hat = -0.05  # Negativo: estreita o intervalo
    l, u = apply_cqr_interval(lower, upper, q_hat)
    assert l.item() == pytest.approx(0.15)
    assert u.item() == pytest.approx(0.85)
    assert (u - l).item() < (upper - lower).item(), (
        'Com q_hat<0 o intervalo deve estreitar'
    )


def test_apply_resm_interval_aritmetica():
    recon = torch.tensor([1.0, 2.0])
    uncertainty = torch.tensor([0.5, 1.0])
    q_hat = 2.0
    l, u = apply_resm_interval(recon, uncertainty, q_hat)
    # l = recon - q_hat * u = [1 - 2*0.5, 2 - 2*1] = [0, 0]
    # u = recon + q_hat * u = [1 + 1, 2 + 2] = [2, 4]
    assert torch.allclose(l, torch.tensor([0.0, 0.0]))
    assert torch.allclose(u, torch.tensor([2.0, 4.0]))
