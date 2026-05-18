# Autor: Massanori
# Data: 17/05/2026
# Descrição: Testes unitarios para src/models/uncertainty_modules.py. Cobre:
#   (1) shape de saida do ResM e (B, 1, H, W),
#   (2) saida do ResM esta em [0, recon] (sigmoid * recon),
#   (3) QR retorna dict com chaves 'lower' e 'upper',
#   (4) lower e upper estao em [0, recon],
#   (5) QR-Lesion e o MESMO objeto que QR (alias) — controle experimental,
#   (6) numero de parametros em hiperparams oficiais bate com Giannakopoulos
#       et al. (2026): ResM ~7.75M, QR ~15.5M, ratio QR/ResM ~2,
#   (7) forward e diferenciavel (gradient flow ate as entradas),
#   (8) forward em GPU se CUDA disponivel.
# Roda com: python -m pytest tests/test_uncertainty_modules.py -v


"""Testes para os modulos de incerteza dos Grupos A/B/C."""
import pytest
import torch

from src.models import (
    QuantileRegressionLesionModule,
    QuantileRegressionModule,
    ResidualMagnitudeModule,
)


def test_resm_retorna_shape_correto():
    module = ResidualMagnitudeModule(chans=8, num_pool_layers=2)
    recon = torch.rand(2, 1, 32, 32)
    out = module(recon)
    assert out.shape == (2, 1, 32, 32)
    assert out.dtype == torch.float32


def test_resm_saida_esta_no_range_zero_recon():
    """sigmoid * recon limita a saida a [0, recon] elementwise.

    Tolerancia de 1e-5 para erros de ponto flutuante na multiplicacao.
    """
    torch.manual_seed(0)
    module = ResidualMagnitudeModule(chans=8, num_pool_layers=2)
    recon = torch.rand(2, 1, 32, 32) + 0.1  # garantir > 0
    module.eval()
    with torch.no_grad():
        out = module(recon)
    assert (out >= -1e-5).all(), 'saida com valor negativo fora de tolerancia'
    assert (out <= recon + 1e-5).all(), 'saida acima de recon'


def test_qr_retorna_dict_com_lower_e_upper():
    module = QuantileRegressionModule(chans=8, num_pool_layers=2)
    recon = torch.rand(2, 1, 32, 32)
    out = module(recon)
    assert isinstance(out, dict)
    assert set(out.keys()) == {'lower', 'upper'}
    assert out['lower'].shape == (2, 1, 32, 32)
    assert out['upper'].shape == (2, 1, 32, 32)


def test_qr_lower_e_upper_estao_em_zero_recon():
    torch.manual_seed(1)
    module = QuantileRegressionModule(chans=8, num_pool_layers=2)
    recon = torch.rand(2, 1, 32, 32) + 0.1
    module.eval()
    with torch.no_grad():
        out = module(recon)
    for key in ('lower', 'upper'):
        assert (out[key] >= -1e-5).all(), f'{key} com valor negativo'
        assert (out[key] <= recon + 1e-5).all(), f'{key} acima de recon'


def test_qr_lesion_e_qr_sao_mesma_classe():
    """Controle experimental: arquitetura identica entre Grupos B e C.

    A unica variavel independente entre B e C e a loss aplicada durante
    o treino. Manter a mesma classe via alias deixa esse controle explicito
    no codigo.
    """
    assert QuantileRegressionLesionModule is QuantileRegressionModule


def test_numero_de_parametros_em_hiperparams_oficiais():
    """ResM ~7.75M e QR ~15.5M params com chans=32, num_pool_layers=4.

    Valores reportados em Giannakopoulos et al. (2026, secao III.D).
    Tolerancia de +/- 10% para acomodar pequenas diferencas no head/tail
    do Unet entre versoes do pacote fastmri.
    """
    resm = ResidualMagnitudeModule(chans=32, num_pool_layers=4)
    qr = QuantileRegressionModule(chans=32, num_pool_layers=4)

    n_resm = sum(p.numel() for p in resm.parameters())
    n_qr = sum(p.numel() for p in qr.parameters())

    assert 7_000_000 <= n_resm <= 8_500_000, (
        f'ResM tem {n_resm:,} params, esperado ~7.75M'
    )
    assert 14_500_000 <= n_qr <= 16_500_000, (
        f'QR tem {n_qr:,} params, esperado ~15.5M'
    )
    # QR deve ter ~2x os params de ResM (duas U-Nets, mesma arquitetura).
    ratio = n_qr / n_resm
    assert 1.95 <= ratio <= 2.05, (
        f'ratio QR/ResM = {ratio:.3f}, esperado ~2.00'
    )


def test_forward_eh_diferenciavel():
    """Sanidade: gradient flow funciona ate as entradas, mesmo com sigmoid."""
    module = ResidualMagnitudeModule(chans=8, num_pool_layers=2)
    recon = torch.rand(1, 1, 32, 32, requires_grad=True)
    out = module(recon)
    out.sum().backward()
    assert recon.grad is not None
    assert recon.grad.abs().sum() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA nao disponivel')
def test_forward_em_gpu():
    module = ResidualMagnitudeModule(chans=8, num_pool_layers=2).cuda()
    recon = torch.rand(1, 1, 32, 32).cuda()
    out = module(recon)
    assert out.device.type == 'cuda'
    assert out.shape == (1, 1, 32, 32)
