# Autor: Massanori
# Data: 13/05/2026
# Descrição: Análise descritiva e definição data-driven dos thresholds de
#            estratificação por tamanho de lesão. Usa APENAS o training set
#            para evitar data leakage metodológico (Demšar, 2006). Recebe:
#            brain.csv, splits/train.txt e splits/test.txt (via env vars).
#            Retorna: figura PNG do histograma em escala logarítmica com as
#            linhas dos thresholds (200/2000 px²) marcadas no vale natural da
#            distribuição bimodal, CSV com áreas e faixas por bbox do train,
#            e JSON com thresholds documentados. Imprime no console a
#            validação do poder estatístico no test set (n por faixa).


"""Histograma de areas de lesao no training set."""
import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config

THRESHOLDS = {'small': 200, 'medium': 2000}


def main(brain_csv_path: Path, splits_dir: Path,
         out_fig: Path, out_csv: Path, out_json: Path):
    for p in [out_fig.parent, out_csv.parent, out_json.parent]:
        p.mkdir(parents=True, exist_ok=True)

    train_files = set((splits_dir / 'train.txt').read_text(encoding='utf-8').strip().split('\n'))
    df = pd.read_csv(brain_csv_path).dropna(subset=['x', 'y', 'width', 'height'])
    df_train = df[df['file'].isin(train_files)].copy()
    print(f'Bboxes no train: {len(df_train)}')

    df_train['area'] = df_train['width'] * df_train['height']
    stats = df_train['area'].describe(percentiles=[.1, .25, .5, .75, .9, .95, .99])
    print('\nEstatisticas de area (px^2):')
    print(stats.round(1))

    def assign_faixa(area, t=THRESHOLDS):
        if area < t['small']:  return f'pequena (<{t["small"]} px)'
        if area < t['medium']: return f'media ({t["small"]}-{t["medium"]} px)'
        return f'grande (>{t["medium"]} px)'

    df_train['faixa'] = df_train['area'].apply(assign_faixa)
    faixa_counts = df_train['faixa'].value_counts()
    print('\nContagem por faixa:')
    print(faixa_counts)

    import re
    df_train['seq'] = df_train['file'].apply(
        lambda s: re.search(r'AX[A-Z0-9]+', s).group()
    )
    print('\nFaixa x Sequencia:')
    print(df_train.groupby(['faixa', 'seq']).size().unstack(fill_value=0))

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    bins = np.logspace(0, 5, 60)
    ax.hist(df_train['area'], bins=bins, color='#4472C4',
            edgecolor='black', alpha=0.85)
    ax.set_xscale('log')
    ax.set_xlabel('Area da lesao (px$^2$)', fontsize=12)
    ax.set_ylabel('Frequencia (n bboxes)', fontsize=12)
    ax.set_title(f'Distribuicao de areas de lesao no training set '
                 f'(n={len(df_train)} bboxes)', fontsize=13)

    for label, val in THRESHOLDS.items():
        ax.axvline(val, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.text(val, ax.get_ylim()[1] * 0.95, f'  {val} px',
                color='red', fontsize=10, ha='left', va='top')

    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_fig, dpi=130, bbox_inches='tight')
    plt.close()

    df_train[['file', 'slice', 'label', 'width', 'height',
              'area', 'faixa', 'seq']].to_csv(out_csv, index=False)

    manifest = {
        'thresholds_px2': THRESHOLDS,
        'defined_on': f'training set only (n_bboxes={len(df_train)})',
        'distribution': dict(faixa_counts),
    }
    out_json.write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                        encoding='utf-8')

    print(f'\nHistograma: {out_fig.resolve()}')
    print(f'CSV: {out_csv.resolve()}')
    print(f'Thresholds: {out_json.resolve()}')

    # Validacao no test set
    test_files = set((splits_dir / 'test.txt').read_text(encoding='utf-8').strip().split('\n'))
    df_test = df[df['file'].isin(test_files)].copy()
    df_test['area'] = df_test['width'] * df_test['height']
    df_test['faixa'] = df_test['area'].apply(assign_faixa)
    print('\n=== VALIDACAO DO PODER ESTATISTICO ===')
    for faixa, n in df_test['faixa'].value_counts().items():
        status = 'OK' if n >= 30 else ('BOOTSTRAP' if n >= 10 else 'INSUFICIENTE')
        print(f'  {faixa:25s} n={n:4d}  [{status}]')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--brain-csv', type=Path, default=None)
    parser.add_argument('--splits-dir', type=Path, default=None)
    parser.add_argument('--figures-dir', type=Path, default=None)
    parser.add_argument('--data-dir', type=Path, default=None)
    parser.add_argument('--configs-dir', type=Path, default=None)
    args = parser.parse_args()

    figures = args.figures_dir or config.figures_dir()
    data = args.data_dir or config.data_dir()
    configs = args.configs_dir or config.configs_dir()

    main(
        brain_csv_path=args.brain_csv or config.brain_csv(),
        splits_dir=args.splits_dir or config.splits_dir(),
        out_fig=figures / 'lesion_area_histogram.png',
        out_csv=data / 'lesion_area_stats.csv',
        out_json=configs / 'lesion_thresholds.json',
    )