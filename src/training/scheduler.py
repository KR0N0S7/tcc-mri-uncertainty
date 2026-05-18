# Autor: Massanori
# Data: 17/05/2026
# Descrição: Linear warmup scheduler usado nos 3 grupos do S5 (D4). Recebe:
#            torch.optim.Optimizer e warmup_steps (int). Retorna: objeto
#            LRScheduler que cresce linearmente de 0 a base_lr nas primeiras
#            warmup_steps iters, depois mantem constante (sem decay).
#            Replicacao exata de Giannakopoulos et al. (2026, secao III.D)
#            para preservar comparabilidade com o sanity do S5.7 (lambda_cal
#            ~1.54 no Grupo B). Protegido por testes em test_scheduler.py.


"""Linear warmup scheduler.

A lr cresce linearmente de 0 ate base_lr nas primeiras warmup_steps iters,
e mantem constante depois. Sem decay pos-warmup — desvios disso quebram
a comparabilidade com Giannakopoulos et al. (2026).

Refs:
    Goyal, P. et al. (2017). Accurate, Large Minibatch SGD: Training
        ImageNet in 1 Hour. arXiv:1706.02677. (warmup linear)
    Giannakopoulos, C. et al. (2026). Pixelwise Uncertainty Quantification
        of Accelerated MRI Reconstruction. arXiv:2601.13236. (§III.D)
"""
from __future__ import annotations

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class LinearWarmupScheduler(LRScheduler):
    """Warmup linear: lr = base_lr * min(1.0, (step + 1) / warmup_steps).

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Otimizador a controlar.
    warmup_steps : int
        Numero de passos do warmup. Apos esse ponto, lr = base_lr.
        Valores tipicos: 7500 (full, replicacao Giannakopoulos) ou
        100 (smoke test, para validar pipeline em 1000 iters).
    last_epoch : int, default -1
        Indice do ultimo step. -1 para inicializacao do zero.

    Notes
    -----
    Apos warmup, mantemos lr constante (sem decay). Replica Giannakopoulos
    et al. (2026), que tambem nao usa decay apos warmup. Esta decisao e
    pre-condicao para reproduzir lambda_cal ~1.54 no Grupo B (validacao
    de replicacao do S5.7).
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        last_epoch: int = -1,
    ) -> None:
        if warmup_steps < 1:
            raise ValueError(
                f'warmup_steps deve ser >= 1, recebido {warmup_steps}'
            )
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list:
        if self.last_epoch < self.warmup_steps:
            factor = (self.last_epoch + 1) / self.warmup_steps
        else:
            factor = 1.0
        return [base_lr * factor for base_lr in self.base_lrs]
