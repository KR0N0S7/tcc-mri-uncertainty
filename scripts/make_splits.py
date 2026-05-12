"""
Split estratificado por sequencia dos 352 volumes com bbox slice-level.
Output: splits/{train,val,cal,test}.txt + splits/manifest.json
Refs: Romano et al. (2019); Tibshirani et al. (2019)
"""
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# ====== CONFIGURAR ======
ANOTADOS  = Path(r'D:\Mri\anotados')
BRAIN_CSV = Path(r'D:\Mri\brain.csv')
OUT_DIR   = Path('splits')
SEED      = 42
SIZES     = {'test': 47, 'cal': 46, 'val': 46}  # train = resto (213)
# ========================

OUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Identificar os 352 com bbox slice-level
df = pd.read_csv(BRAIN_CSV)
df_bbox = df.dropna(subset=['x', 'y', 'width', 'height'])
volumes_bbox = set(df_bbox['file'].unique())
have = {p.stem for p in ANOTADOS.glob('*.h5')}
elegiveis = sorted(volumes_bbox & have)
print(f'Volumes elegiveis (com bbox e em disco): {len(elegiveis)}')
assert len(elegiveis) == 352, f'Esperado 352, obteve {len(elegiveis)}'

# 2. Tabela com coluna de sequencia
seqs = [re.search(r'AX[A-Z0-9]+', v).group() for v in elegiveis]
vols = pd.DataFrame({'file': elegiveis, 'seq': seqs})
print('\nDistribuicao por sequencia:')
print(vols['seq'].value_counts())

# 3. Split estratificado em 3 etapas, seed fixo
rest, test = train_test_split(vols, test_size=SIZES['test'],
                              stratify=vols['seq'], random_state=SEED)
rest, cal = train_test_split(rest, test_size=SIZES['cal'],
                             stratify=rest['seq'], random_state=SEED)
train, val = train_test_split(rest, test_size=SIZES['val'],
                              stratify=rest['seq'], random_state=SEED)

splits = {'train': train, 'val': val, 'cal': cal, 'test': test}

# 4. Verificacao de leak (erros fatais se falharem)
sets_files = {k: set(v['file']) for k, v in splits.items()}
for k1 in sets_files:
    for k2 in sets_files:
        if k1 < k2:  # cada par uma vez
            inter = sets_files[k1] & sets_files[k2]
            assert len(inter) == 0, f'LEAK entre {k1} e {k2}: {inter}'
print('\nVerificacao de leak: OK (todos os pares disjuntos)')

# 5. Salvar arquivos .txt (um nome por linha, ordenado)
for name, dfk in splits.items():
    out_file = OUT_DIR / f'{name}.txt'
    files_sorted = sorted(dfk['file'].tolist())
    out_file.write_text('\n'.join(files_sorted) + '\n', encoding='utf-8')

# 6. Manifesto JSON com tudo que importa para reproduzir
manifest = {
    'seed': SEED,
    'total_elegiveis': len(elegiveis),
    'criterio_inclusao': 'volumes com pelo menos uma bbox slice-level no brain.csv',
    'criterio_exclusao': '398 volumes apenas com anotacao study-level',
    'sizes': {k: len(v) for k, v in splits.items()},
    'seq_distribution': {
        k: Counter(v['seq'].tolist()) for k, v in splits.items()
    },
    'fastmri_plus_ref': 'Zhao et al. 2022, Sci Data 9:152',
    'stratification_ref': 'Tibshirani et al. 2019, Conformal Prediction Under Covariate Shift',
}
(OUT_DIR / 'manifest.json').write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8'
)

# 7. Print bonito do resultado final
print('\n' + '='*60)
print(f"{'Conjunto':<10} {'AXFLAIR':<10} {'AXT1':<10} {'AXT1POST':<10} {'Total':<10}")
print('='*60)
for name, dfk in splits.items():
    c = Counter(dfk['seq'].tolist())
    print(f"{name:<10} {c.get('AXFLAIR',0):<10} {c.get('AXT1',0):<10} "
          f"{c.get('AXT1POST',0):<10} {len(dfk):<10}")
print('='*60)
print(f'\nArquivos salvos em {OUT_DIR.resolve()}')