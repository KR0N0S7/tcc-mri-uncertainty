# Autor: Massanori
# Data: 13/05/2026
# Descrição: Gera split estratificado por sequência dos 352 volumes elegíveis
#            (com bbox slice-level) com seed=42. Aplicação de Tibshirani et al.
#            (2019) para garantir exchangeability entre calibração e teste no
#            CQR. Recebe: pasta anotados/, brain.csv e pasta de saída (todos
#            via env vars com defaults). Retorna: 4 arquivos .txt
#            (train/val/cal/test = 213/46/46/47) com nomes de volumes
#            ordenados, e manifest.json com seed, tamanhos, distribuição por
#            sequência e referências. Verifica não-leak entre conjuntos via
#            asserts.


"""Split estratificado por sequencia dos 352 volumes com bbox slice-level."""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config

SEED = 42
SIZES = {'test': 47, 'cal': 46, 'val': 46}


def main(anotados: Path, brain_csv_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(brain_csv_path)
    df_bbox = df.dropna(subset=['x', 'y', 'width', 'height'])
    volumes_bbox = set(df_bbox['file'].unique())
    have = {p.stem for p in anotados.glob('*.h5')}
    elegiveis = sorted(volumes_bbox & have)
    print(f'Volumes elegiveis: {len(elegiveis)}')
    assert len(elegiveis) == 352, f'Esperado 352, obteve {len(elegiveis)}'

    seqs = [re.search(r'AX[A-Z0-9]+', v).group() for v in elegiveis]
    vols = pd.DataFrame({'file': elegiveis, 'seq': seqs})
    print('\nDistribuicao por sequencia:')
    print(vols['seq'].value_counts())

    rest, test = train_test_split(vols, test_size=SIZES['test'],
                                  stratify=vols['seq'], random_state=SEED)
    rest, cal = train_test_split(rest, test_size=SIZES['cal'],
                                 stratify=rest['seq'], random_state=SEED)
    train, val = train_test_split(rest, test_size=SIZES['val'],
                                  stratify=rest['seq'], random_state=SEED)

    splits = {'train': train, 'val': val, 'cal': cal, 'test': test}
    sets_files = {k: set(v['file']) for k, v in splits.items()}
    for k1 in sets_files:
        for k2 in sets_files:
            if k1 < k2:
                assert len(sets_files[k1] & sets_files[k2]) == 0, \
                    f'LEAK entre {k1} e {k2}'
    print('\nVerificacao de leak: OK')

    for name, dfk in splits.items():
        (out_dir / f'{name}.txt').write_text(
            '\n'.join(sorted(dfk['file'].tolist())) + '\n', encoding='utf-8'
        )

    manifest = {
        'seed': SEED,
        'total_elegiveis': len(elegiveis),
        'criterio_inclusao': 'volumes com pelo menos uma bbox slice-level',
        'sizes': {k: len(v) for k, v in splits.items()},
        'seq_distribution': {k: Counter(v['seq'].tolist()) for k, v in splits.items()},
        'fastmri_plus_ref': 'Zhao et al. 2022, Sci Data 9:152',
        'stratification_ref': 'Tibshirani et al. 2019',
    }
    (out_dir / 'manifest.json').write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8'
    )

    print('\n' + '=' * 60)
    print(f"{'Conjunto':<10} {'AXFLAIR':<10} {'AXT1':<10} {'AXT1POST':<10} {'Total':<10}")
    print('=' * 60)
    for name, dfk in splits.items():
        c = Counter(dfk['seq'].tolist())
        print(f"{name:<10} {c.get('AXFLAIR', 0):<10} {c.get('AXT1', 0):<10} "
              f"{c.get('AXT1POST', 0):<10} {len(dfk):<10}")
    print('=' * 60)
    print(f'\nArquivos salvos em {out_dir.resolve()}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--anotados', type=Path, default=None)
    parser.add_argument('--brain-csv', type=Path, default=None)
    parser.add_argument('--out', type=Path, default=None,
                        help='Pasta de output (default: $TCC_SPLITS_DIR ou ./splits)')
    args = parser.parse_args()

    main(
        anotados=args.anotados or config.anotados_dir(),
        brain_csv_path=args.brain_csv or config.brain_csv(),
        out_dir=args.out or config.splits_dir(),
    )