# Autor: Massanori
# Data: 17/05/2026
# Descrição: Modulo de funcoes de perda para os tres modulos de incerteza
#            do S5 (Grupos A/B/C). Expoe a interface unificada de loss (D3)
#            e a primitiva matematica reutilizavel (pinball). Recebe via
#            imports: torch.Tensor de predicoes, ground truth, mascaras.
#            Retorna: losses escalares prontas para .backward().

from src.losses.pinball import pinball_loss, pinball_per_pixel
from src.losses.uncertainty_losses import (
    DEFAULT_ALPHA,
    qr_lesion_loss,
    qr_loss,
    resm_loss,
)

__all__ = [
    'DEFAULT_ALPHA',
    'pinball_loss',
    'pinball_per_pixel',
    'qr_lesion_loss',
    'qr_loss',
    'resm_loss',
]
