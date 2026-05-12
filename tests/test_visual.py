"""
Validacao visual de alinhamento bbox <-> reconstrucao do fastMRI+
Refs: Zhao et al. (2022) Scientific Data 9:152; Zbontar et al. (2020) arXiv:1811.08839
"""
import random
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ============ CONFIGURAR ============
ANOTADOS_DIR = Path(r'D:\Mri\anotados')
BRAIN_CSV    = Path(r'D:\Mri\brain.csv')
OUTPUT_DIR   = Path(r'D:\Mri\validacao_bbox')
SEED         = 42
TARGET_CROP  = 320              # tamanho ao qual o fastMRI+ converte para DICOM
QUOTAS = {'AXFLAIR': 4, 'AXT1': 4, 'AXT1POST': 2}  # total = 10
# ====================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
random.seed(SEED)
np.random.seed(SEED)

# 1. Ler CSV, filtrar so anotacoes com bbox (excluir study-level sem coordenadas)
df = pd.read_csv(BRAIN_CSV)
df = df.dropna(subset=['x', 'y', 'width', 'height'])
df = df[df['study_level'] == 'No']  # exclui anotacoes de estudo sem bbox
df['seq'] = df['file'].str.extract(r'file_brain_(AX[A-Z0-9]+)_')

# Filtrar volumes que voce realmente tem na pasta
have = {p.stem for p in ANOTADOS_DIR.glob('*.h5')}
df = df[df['file'].isin(have)]
print(f'Anotacoes com bbox em volumes disponiveis: {len(df)}')
print(df['seq'].value_counts())

# 2. Amostrar pares (volume, slice) estratificados por sequencia
chosen_pairs = []
for seq, n in QUOTAS.items():
    sub = df[df['seq'] == seq]
    pairs = (sub.groupby(['file', 'slice']).size().reset_index()[['file','slice']])
    picked = pairs.sample(n=min(n, len(pairs)), random_state=SEED)
    chosen_pairs.extend(picked.to_dict('records'))

print(f'\n{len(chosen_pairs)} pares selecionados:')
for p in chosen_pairs:
    print(f"  {p['file']}  fatia {p['slice']}")

# 3. Helper de center-crop (replica fastmri-to-dicom.py)
def center_crop(img, target):
    h, w = img.shape[-2:]
    if h < target or w < target:
        return img  # nao crop se imagem ja menor
    y0 = (h - target) // 2
    x0 = (w - target) // 2
    return img[..., y0:y0+target, x0:x0+target]

# 4. Gerar overlay para cada par
results = []
for i, pair in enumerate(chosen_pairs):
    vol, sl = pair['file'], int(pair['slice'])
    h5_path = ANOTADOS_DIR / f'{vol}.h5'

    with h5py.File(h5_path, 'r') as hf:
        if 'reconstruction_rss' not in hf:
            print(f'[PULAR] {vol}: sem reconstruction_rss')
            continue
        recon = hf['reconstruction_rss'][sl]   # (H, W)
        orig_shape = recon.shape

    # Aplicar center-crop se necessario
    cropped = recon.shape != (TARGET_CROP, TARGET_CROP)
    recon_show = center_crop(recon, TARGET_CROP) if cropped else recon

    # Anotacoes daquela fatia daquele volume
    rows = df[(df['file'] == vol) & (df['slice'] == sl)]

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    ax.imshow(recon_show, cmap='gray')
    for _, r in rows.iterrows():
        x, y, w, h = int(r['x']), int(r['y']), int(r['width']), int(r['height'])
        rect = patches.Rectangle((x, y), w, h, linewidth=2,
                                 edgecolor='lime', facecolor='none')
        ax.add_patch(rect)
        ax.text(x, max(y-4, 8), r['label'][:30],
                color='lime', fontsize=9,
                bbox=dict(facecolor='black', alpha=0.6, edgecolor='none', pad=1.5))
    ax.set_title(f'{vol}\nfatia {sl} | shape original {orig_shape} -> exibido {recon_show.shape}',
                 fontsize=9)
    ax.axis('off')

    out = OUTPUT_DIR / f'overlay_{i:02d}_{vol}_s{sl}.png'
    plt.savefig(out, dpi=130, bbox_inches='tight')
    plt.close()

    results.append({'file': vol, 'slice': sl,
                    'orig_shape': orig_shape,
                    'n_bboxes': len(rows),
                    'output': str(out)})
    print(f'[{i+1:2d}/{len(chosen_pairs)}] {vol} fatia {sl}: {len(rows)} bbox(es) -> {out.name}')

# 5. Salvar metadata para o documento de validacao
meta = pd.DataFrame(results)
meta.to_csv(OUTPUT_DIR / 'metadata.csv', index=False)
print(f'\nValidacao gerada em {OUTPUT_DIR}')
print(f'PROXIMO PASSO: abrir os {len(results)} PNGs, julgar 1 a 1, e preencher RESULTADO.md')

# Adicione ao final do script anterior — diagnostico do volume suspeito
SUSPEITO = 'file_brain_AXFLAIR_200_6002493'
for sl in [5, 6, 7]:
    rows_sl = df[(df['file'] == SUSPEITO) & (df['slice'] == sl)]
    print(f'Fatia {sl}: {len(rows_sl)} bbox(es)')
    for _, r in rows_sl.iterrows():
        print(f"   {r['label']:30s} x={r['x']} y={r['y']} w={r['width']} h={r['height']}")