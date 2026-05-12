"""Audita o dataset: separa volumes por tipo de anotacao."""
from pathlib import Path
import pandas as pd

ANOTADOS = Path(r'D:\Mri\anotados')
BRAIN_CSV = Path(r'D:\Mri\brain.csv')

df = pd.read_csv(BRAIN_CSV)
have = {p.stem for p in ANOTADOS.glob('*.h5')}
print(f'Volumes em disco: {len(have)}')
print(f'Linhas no brain.csv: {len(df)}')

# Volumes com bbox slice-level (pelo menos uma)
com_bbox = df.dropna(subset=['x','y','width','height'])['file'].unique()
com_bbox_em_disco = set(com_bbox) & have

# Volumes apenas com study-level
so_study = (df[df['study_level']=='Yes']['file'].unique())
so_study_em_disco = (set(so_study) - set(com_bbox)) & have

# Volumes que estao em disco mas nao aparecem no CSV (deveria ser zero)
nao_no_csv = have - set(df['file'].unique())

print(f'\nVolumes COM bbox slice-level: {len(com_bbox_em_disco)}')
print(f'Volumes SO com study-level (sem bbox): {len(so_study_em_disco)}')
print(f'Volumes em disco mas fora do brain.csv: {len(nao_no_csv)}')

# Distribuicao por sequencia entre os 'com bbox'
import re
seq_dist = pd.Series([re.search(r'AX[A-Z0-9]+', v).group()
                      for v in com_bbox_em_disco]).value_counts()
print(f'\nDistribuicao por sequencia (so com bbox):')
print(seq_dist)

# Top study-level labels nos 398 "sem bbox"
study_labels = df[df['file'].isin(so_study_em_disco)]['label'].value_counts().head(10)
print(f'\nTop 10 labels study-level entre os "sem bbox":')
print(study_labels)