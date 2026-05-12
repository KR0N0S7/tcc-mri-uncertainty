"""
Smoke test do pipeline de mascaras: 1 volume, verificacao visual.
Ref: Zhao et al. (2022) Scientific Data 9:152
"""
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.lesion_masks import volume_masks_from_h5

# ====== CONFIGURAR ======
ANOTADOS  = Path(r'D:\Mri\anotados')
BRAIN_CSV = Path(r'D:\Mri\brain.csv')
OUTPUT    = Path('figures/validacao_bbox/smoke_test_mask.png')
VOL = 'file_brain_AXFLAIR_200_6002493'  # volume da validacao
SL = 6                                   # fatia da validacao
# ========================

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
brain_df = pd.read_csv(BRAIN_CSV)
masks = volume_masks_from_h5(ANOTADOS / f'{VOL}.h5', brain_df, apply_y_flip=True)

with h5py.File(ANOTADOS / f'{VOL}.h5', 'r') as hf:
    img = hf['reconstruction_rss'][SL]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(img, cmap='gray')
axes[0].set_title('reconstruction_rss (fastMRI)')

axes[1].imshow(masks[SL], cmap='Reds')
axes[1].set_title(f'Mascara gerada pos-flip\n({int(masks[SL].sum().item())} pixels)')

axes[2].imshow(img, cmap='gray')
axes[2].imshow(masks[SL], cmap='Reds', alpha=0.4)
axes[2].set_title('Overlay (verificar alinhamento)')

for ax in axes:
    ax.axis('off')

plt.suptitle(f'{VOL} fatia {SL}', fontsize=11)
plt.tight_layout()
plt.savefig(OUTPUT, dpi=120, bbox_inches='tight')
plt.close()

# Estatisticas para confirmar
total_pixels_mascara = int(masks.sum().item())
fatias_com_lesao = int((masks.sum(dim=(1, 2)) > 0).sum().item())
print(f'\nVolume: {VOL}')
print(f'Shape do tensor de mascaras: {tuple(masks.shape)}')
print(f'Fatias com lesao anotada: {fatias_com_lesao}/{masks.shape[0]}')
print(f'Total de pixels de lesao no volume: {total_pixels_mascara}')
print(f'Fatia {SL}: {int(masks[SL].sum().item())} pixels marcados')
print(f'\nFigura salva em: {OUTPUT.resolve()}')