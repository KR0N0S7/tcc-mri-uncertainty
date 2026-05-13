# Autor: Massanori
# Data: 13/05/2026
# Descrição: Audita o dataset, separando volumes por tipo de anotação
#            (slice-level com bbox vs study-level sem coordenadas). Recebe:
#            pasta anotados/ e brain.csv (via env vars ou --anotados/--brain-csv).
#            Retorna: prints no console com contagens (volumes com bbox, só
#            study-level, fora do CSV), distribuição por sequência
#            (AXFLAIR/AXT1/AXT1POST) entre os elegíveis, e top 10 labels
#            study-level. Serve para validar a integridade do dataset antes
#            do split estratificado.


"""Audita o dataset: separa volumes por tipo de anotacao."""
import argparse
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config


def main(anotados: Path, brain_csv_path: Path):
    df = pd.read_csv(brain_csv_path)
    have = {p.stem for p in anotados.glob('*.h5')}
    print(f'Volumes em disco: {len(have)}')
    print(f'Linhas no brain.csv: {len(df)}')

    com_bbox = df.dropna(subset=['x', 'y', 'width', 'height'])['file'].unique()
    com_bbox_em_disco = set(com_bbox) & have

    so_study = df[df['study_level'] == 'Yes']['file'].unique()
    so_study_em_disco = (set(so_study) - set(com_bbox)) & have

    nao_no_csv = have - set(df['file'].unique())

    print(f'\nVolumes COM bbox slice-level: {len(com_bbox_em_disco)}')
    print(f'Volumes SO com study-level (sem bbox): {len(so_study_em_disco)}')
    print(f'Volumes em disco mas fora do brain.csv: {len(nao_no_csv)}')

    seq_dist = pd.Series(
        [re.search(r'AX[A-Z0-9]+', v).group() for v in com_bbox_em_disco]
    ).value_counts()
    print(f'\nDistribuicao por sequencia (so com bbox):')
    print(seq_dist)

    study_labels = (
        df[df['file'].isin(so_study_em_disco)]['label'].value_counts().head(10)
    )
    print(f'\nTop 10 labels study-level entre os "sem bbox":')
    print(study_labels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--anotados', type=Path, default=None,
                        help='Pasta com .h5 (default: $TCC_ANOTADOS_DIR)')
    parser.add_argument('--brain-csv', type=Path, default=None,
                        help='Caminho do brain.csv (default: $TCC_BRAIN_CSV)')
    args = parser.parse_args()

    main(
        anotados=args.anotados or config.anotados_dir(),
        brain_csv_path=args.brain_csv or config.brain_csv(),
    )