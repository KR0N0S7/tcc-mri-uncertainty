# Autor: Massanori
# Data: 17/05/2026
# Descrição: Testes unitarios para src/losses/. Cobre:
#   (1) pinball converge para o quantil empirico de y (sanidade da
#       formulacao de Koenker & Bassett, 1978),
#   (2) pinball_loss(reduction='mean') == pinball_per_pixel.mean(),
#   (3) resm_loss e MSE entre uncertainty e |recon - target|,
#   (4) resm_loss ignora lesion_mask (interface unificada D3),
#   (5) qr_loss soma pinballs em alpha/2 e 1-alpha/2,
#   (6) qr_lesion_loss(lambda=1) == qr_loss EXATAMENTE — protege a
#       formulacao da loss hibrida e garante que C nao pode ser pior
#       que B no limite,
#   (7) qr_lesion_loss(lambda>1) amplifica regioes de lesao,
#   (8) qr_lesion_loss com mask=0 em todo lugar iguala qr_loss
#       independente de lambda,
#   (9) pinball rejeita alpha fora de (0, 1),
#   (10) pinball rejeita shapes incompativeis.
# Roda com: python -m pytest tests/test_uncertainty_losses.py -v


"""Testes para src/losses/."""
import pytest
import torch

from src.losses import (
    DEFAULT_ALPHA,
    pinball_loss,
    pinball_per_pixel,
    qr_lesion_loss,
    qr_loss,
    resm_loss,
)


def test_pinball_converge_para_quantil_empirico():
    """Minimizar pinball(alpha) sobre y faz q_hat convergir para o quantil-alpha.

    Para alpha=0.10 e y ~ N(0, 1), o quantil teorico e Phi^{-1}(0.10) ~= -1.282.
    Com 10000 amostras, o quantil empirico deve ficar a 0.05 do teorico, e a
    convergencia por SGD deve estar dentro de 0.1 apos 2000 passos.
    """
    torch.manual_seed(42)
    y = torch.randn(10000)
    q_hat = torch.nn.Parameter(torch.tensor(5.0))
    opt = torch.optim.SGD([q_hat], lr=0.05)
    for _ in range(2000):
        opt.zero_grad()
        loss = pinball_loss(q_hat.expand_as(y), y, alpha=0.10)
        loss.backward()
        opt.step()
    assert abs(q_hat.item() - (-1.282)) < 0.1, (
        f'q_hat={q_hat.item():.3f}, esperado ~-1.282 '
        f'(quantil 10% de N(0,1))'
    )


def test_pinball_loss_consistente_com_per_pixel_mean():
    q_pred = torch.randn(2, 1, 16, 16)
    target = torch.randn(2, 1, 16, 16)
    agg = pinball_loss(q_pred, target, alpha=0.10, reduction='mean')
    pp = pinball_per_pixel(q_pred, target, alpha=0.10).mean()
    assert torch.allclose(agg, pp, atol=1e-6)


def test_resm_loss_e_mse_entre_uncertainty_e_abs_residuo():
    torch.manual_seed(0)
    recon = torch.rand(2, 1, 8, 8)
    target = torch.rand(2, 1, 8, 8)
    pred = torch.rand(2, 1, 8, 8)

    expected = ((pred - torch.abs(recon - target)) ** 2).mean()
    actual = resm_loss(pred, recon, target, lesion_mask=None)
    assert torch.allclose(actual, expected, atol=1e-6)


def test_resm_loss_ignora_lesion_mask():
    """Interface unificada (D3): A ignora lesion_mask completamente."""
    torch.manual_seed(1)
    recon = torch.rand(2, 1, 8, 8)
    target = torch.rand(2, 1, 8, 8)
    pred = torch.rand(2, 1, 8, 8)

    l_none = resm_loss(pred, recon, target, lesion_mask=None)
    l_z = resm_loss(pred, recon, target, lesion_mask=torch.zeros_like(recon))
    l_o = resm_loss(pred, recon, target, lesion_mask=torch.ones_like(recon))

    assert torch.allclose(l_none, l_z, atol=1e-7)
    assert torch.allclose(l_none, l_o, atol=1e-7)


def test_qr_loss_e_soma_de_pinballs_em_alpha_meio_e_1_menos_alpha_meio():
    torch.manual_seed(2)
    lower = torch.rand(2, 1, 8, 8)
    upper = torch.rand(2, 1, 8, 8)
    target = torch.rand(2, 1, 8, 8)
    recon = torch.rand(2, 1, 8, 8)

    expected = (
        pinball_per_pixel(lower, target, DEFAULT_ALPHA / 2).mean()
        + pinball_per_pixel(upper, target, 1 - DEFAULT_ALPHA / 2).mean()
    )
    actual = qr_loss({'lower': lower, 'upper': upper},
                     recon, target, lesion_mask=None,
                     alpha=DEFAULT_ALPHA)
    assert torch.allclose(actual, expected, atol=1e-6)


def test_qr_lesion_loss_lambda_1_recupera_qr_loss_exatamente():
    """Sanidade central da formulacao: lambda=1 -> L_C == L_B.

    Garante que o Grupo C nao pode ter performance pior que o Grupo B
    no limite. Qualquer diferenca observada nas metricas e atribuivel
    ao efeito da ponderacao por lesao, nao a viees da implementacao.
    """
    torch.manual_seed(3)
    lower = torch.rand(2, 1, 8, 8)
    upper = torch.rand(2, 1, 8, 8)
    target = torch.rand(2, 1, 8, 8)
    recon = torch.rand(2, 1, 8, 8)
    mask = torch.randint(0, 2, recon.shape).float()  # mascara aleatoria

    l_b = qr_loss({'lower': lower, 'upper': upper},
                  recon, target, lesion_mask=None,
                  alpha=DEFAULT_ALPHA)
    l_c = qr_lesion_loss({'lower': lower, 'upper': upper},
                         recon, target, lesion_mask=mask,
                         alpha=DEFAULT_ALPHA, lambda_lesion=1.0)
    assert torch.allclose(l_b, l_c, atol=1e-6), (
        f'lambda=1 nao recupera qr_loss: l_b={l_b.item()}, l_c={l_c.item()}'
    )


def test_qr_lesion_loss_amplifica_regioes_de_lesao_com_lambda_maior_que_1():
    torch.manual_seed(4)
    target = torch.rand(1, 1, 8, 8)
    recon = torch.rand(1, 1, 8, 8)
    # Preditor proposital com erro grande dentro da lesao para destacar o efeito
    lower = torch.zeros_like(target)
    upper = torch.ones_like(target) * 2.0
    mask = torch.zeros_like(target)
    mask[0, 0, 0:4, 0:4] = 1.0  # quadrante superior esquerdo e lesao

    l_lambda1 = qr_lesion_loss({'lower': lower, 'upper': upper},
                               recon, target, lesion_mask=mask,
                               lambda_lesion=1.0)
    l_lambda5 = qr_lesion_loss({'lower': lower, 'upper': upper},
                               recon, target, lesion_mask=mask,
                               lambda_lesion=5.0)
    assert l_lambda5 > l_lambda1, (
        f'lambda=5 deveria amplificar loss: '
        f'l1={l_lambda1.item():.4f}, l5={l_lambda5.item():.4f}'
    )


def test_qr_lesion_loss_iguala_qr_loss_quando_mask_zerada():
    """Mask=0 em todo lugar deve recuperar qr_loss, independente de lambda."""
    torch.manual_seed(5)
    lower = torch.rand(1, 1, 8, 8)
    upper = torch.rand(1, 1, 8, 8)
    target = torch.rand(1, 1, 8, 8)
    recon = torch.rand(1, 1, 8, 8)
    mask = torch.zeros_like(target)

    l_b = qr_loss({'lower': lower, 'upper': upper}, recon, target, lesion_mask=None)
    l_c = qr_lesion_loss({'lower': lower, 'upper': upper},
                         recon, target, lesion_mask=mask,
                         lambda_lesion=10.0)
    assert torch.allclose(l_b, l_c, atol=1e-6)


def test_pinball_rejeita_alpha_fora_do_range():
    q = torch.rand(4)
    y = torch.rand(4)
    with pytest.raises(ValueError, match='alpha'):
        pinball_per_pixel(q, y, alpha=0.0)
    with pytest.raises(ValueError, match='alpha'):
        pinball_per_pixel(q, y, alpha=1.0)
    with pytest.raises(ValueError, match='alpha'):
        pinball_per_pixel(q, y, alpha=-0.1)


def test_pinball_rejeita_shapes_incompativeis():
    q = torch.rand(4)
    y = torch.rand(5)
    with pytest.raises(ValueError, match='Shapes'):
        pinball_per_pixel(q, y, alpha=0.1)
