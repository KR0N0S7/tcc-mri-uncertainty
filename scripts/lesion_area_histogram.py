"""
Histograma de areas de lesao no TRAINING SET dos 352 volumes selecionados.
Define formalmente as faixas para estratificacao de metricas regionais.

Refs:
- Demsar (2006). Statistical Comparisons of Classifiers over Multiple Data Sets. JMLR.
- Ottesen, Storas & Caan (2025). Evaluating Structural Uncertainty in Accelerated MRI.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ====== CONFIG ======
BRAIN_CSV = Path(r'D:\Mri\brain.csv')
TRAIN_TXT = Path('splits/train.txt')
OUT_FIG   = Path('figures/lesion_area_histogram.png')
OUT_CSV   = Path('data/lesion_area_stats.csv')
OUT_JSON  = Path('configs/lesion_thresholds.json')

THRESHOLDS = {'small': 200, 'medium': 2000}  # pixels^2
# =====================

for p in [OUT_FIG.parent, OUT_CSV.parent, OUT_JSON.parent]:
    p.mkdir(parents=True, exist_ok=True)

# 1. Carregar splits e filtrar brain.csv
train_files = set(TRAIN_TXT.read_text(encoding='utf-8').strip().split('\n'))
df = pd.read_csv(BRAIN_CSV).dropna(subset=['x','y','width','height'])
df_train = df[df['file'].isin(train_files)].copy()
print(f'Bboxes no train: {len(df_train)} (de {len(train_files)} volumes)')

# 2. Calcular area por bbox
df_train['area'] = df_train['width'] * df_train['height']

# 3. Estatisticas
stats = df_train['area'].describe(percentiles=[.1, .25, .5, .75, .9, .95, .99])
print('\nEstatisticas de area (px^2):')
print(stats.round(1))

# 4. Definir faixas
def assign_faixa(area, t=THRESHOLDS):
    if area < t['small']:  return f'pequena (<{t["small"]} px)'
    if area < t['medium']: return f'media ({t["small"]}-{t["medium"]} px)'
    return f'grande (>{t["medium"]} px)'

df_train['faixa'] = df_train['area'].apply(assign_faixa)
faixa_counts = df_train['faixa'].value_counts()
print('\nContagem por faixa (bboxes):')
print(faixa_counts)
print(f'\nFaixa percentuais:')
print((faixa_counts / len(df_train) * 100).round(1))

# 5. Contagem por faixa cruzada com sequencia
print('\nFaixa x Sequencia:')
import re
df_train['seq'] = df_train['file'].apply(lambda s: re.search(r'AX[A-Z0-9]+', s).group())
print(df_train.groupby(['faixa', 'seq']).size().unstack(fill_value=0))

# 6. Plot — histograma com escala log no eixo X
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
bins = np.logspace(0, 5, 60)  # 1 a 100k pixels^2, log-spaced
ax.hist(df_train['area'], bins=bins, color='#4472C4', edgecolor='black', alpha=0.85)
ax.set_xscale('log')
ax.set_xlabel('Area da lesao (px$^2$)', fontsize=12)
ax.set_ylabel('Frequencia (n bboxes)', fontsize=12)
ax.set_title(f'Distribuicao de areas de lesao no training set (n={len(df_train)} bboxes)',
             fontsize=13)

# Linhas verticais nos thresholds
for label, val in THRESHOLDS.items():
    ax.axvline(val, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.text(val, ax.get_ylim()[1] * 0.95, f'  {val} px',
            color='red', fontsize=10, ha='left', va='top')

ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_FIG, dpi=130, bbox_inches='tight')
plt.close()
print(f'\nHistograma salvo: {OUT_FIG.resolve()}')

# 7. Salvar estatisticas como CSV
df_train[['file','slice','label','width','height','area','faixa','seq']].to_csv(
    OUT_CSV, index=False, encoding='utf-8'
)
print(f'CSV de areas salvo: {OUT_CSV.resolve()}')

# 8. Salvar thresholds como JSON versionavel
thresholds_manifest = {
    'thresholds_px2': THRESHOLDS,
    'defined_on': 'training set only (n_bboxes={})'.format(len(df_train)),
    'distribution': {
        'pequena': int(faixa_counts.get('pequena (<50 px)', 0)),
        'media':   int(faixa_counts.get('media (50-200 px)', 0)),
        'grande':  int(faixa_counts.get('grande (>200 px)', 0)),
    },
    'rationale': (
        'Thresholds definidos apenas com training set para evitar data leakage '
        'metodologico. Mantem consistencia com Demsar (2006). Faixas escolhidas '
        'considerando: (i) lesoes <50 px aproximam a resolucao de bbox minima '
        'do fastMRI+; (ii) lesoes >200 px sao predominantemente massas e nao '
        'desafiam tanto a quantificacao de incerteza.'
    ),
}
OUT_JSON.write_text(json.dumps(thresholds_manifest, indent=2, ensure_ascii=False),
                    encoding='utf-8')
print(f'Thresholds salvos: {OUT_JSON.resolve()}')

# 9. Validar n por faixa no test set (poder estatistico)
print('\n=== VALIDACAO DO PODER ESTATISTICO ===')
test_files = set(Path('splits/test.txt').read_text(encoding='utf-8').strip().split('\n'))
df_test = df[df['file'].isin(test_files)].copy()
df_test['area'] = df_test['width'] * df_test['height']
df_test['faixa'] = df_test['area'].apply(assign_faixa)
n_por_faixa_test = df_test['faixa'].value_counts()
print('Bboxes por faixa no TEST set:')
for faixa, n in n_por_faixa_test.items():
    status = 'OK' if n >= 30 else ('USAR BOOTSTRAP' if n >= 10 else 'INSUFICIENTE')
    print(f'  {faixa:25s} n={n:4d}  [{status}]')