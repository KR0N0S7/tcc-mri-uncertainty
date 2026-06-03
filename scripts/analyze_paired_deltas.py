# Autor: Massanori
# Data: 02/06/2026
# Descrição: Tamanho de efeito e IC sobre o delta pareado, destino:
#            scripts/analyze_paired_deltas.py. Le os metrics_*.csv por slice do
#            S5.8, alinha por (volume_id, slice_idx) entre grupos e, para cada
#            metrica e cada par (A-B, A-C, B-C), reporta: media do delta
#            pareado, IC 95% BCa do delta (bootstrap, Efron & Tibshirani 1993),
#            correlacao rank-biserial pareada (tamanho de efeito do Wilcoxon,
#            Kerby 2014) e d_z de Cohen pareado. Complementa o S5.9 (que tinha
#            BCa sobre MEDIAS por grupo, nao sobre o DELTA). Para efeitos
#            pequenos, reportar IC do delta + tamanho de efeito e' mais honesto
#            e robusto que o p-valor isolado. Recebe via CLI: --metrics-dir ou
#            --csv, --output. Gera: CSV [metrica, par, delta_mean, ci_low,
#            ci_high, rank_biserial, cohen_dz, n_pares]. Sem dados pessoais.
#            Fundamentos: Efron & Tibshirani (1993); Kerby (2014); Demsar (2006).

"""IC BCa e tamanho de efeito sobre o delta pareado entre grupos.

Por que o delta pareado (e nao a media por grupo)
-------------------------------------------------
O S5.9 reportou BCa sobre as medias de cada grupo separadamente. Mas a
hipotese e' SOBRE A DIFERENCA (C > B etc.), e o desenho e' pareado por slice.
O objeto estatistico correto e' o delta d_i = metric_X(slice_i) -
metric_Y(slice_i), do qual se reporta:
  - media do delta e IC 95% BCa (bootstrap do delta) — magnitude com incerteza;
  - rank-biserial pareada — tamanho de efeito alinhado ao Wilcoxon usado no S5.9;
  - d_z de Cohen = mean(d)/std(d) — tamanho de efeito padronizado pareado.
Isso evita a leitura enganosa "p<0.001 logo grande": separa significancia
(ha efeito direcional) de magnitude (quao grande, com IC).

Refs:
    Efron, B.; Tibshirani, R.J. (1993). An Introduction to the Bootstrap.
        Chapman & Hall/CRC. (BCa)
    Kerby, D.S. (2014). The simple difference formula: an approach to teaching
        nonparametric correlation. Comprehensive Psychology, 3:1. (rank-biserial)
    Demsar, J. (2006). Statistical comparisons of classifiers over multiple
        data sets. JMLR 7:1-30.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import bootstrap, rankdata

KEY = ['volume_id', 'slice_idx']
PAIRS = [('A', 'B'), ('A', 'C'), ('B', 'C')]
METRICS = [
    'coverage_global', 'coverage_lesion',
    'mean_width_global', 'mean_width_lesion',
    'iou_topk_global', 'iou_topk_lesion',
    'ulas_lesion',
]
N_RESAMPLES = 10_000
SEED = 42


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='IC BCa e efeito sobre delta pareado.')
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--metrics-dir', type=Path,
                   help='Dir com metrics_A.csv, metrics_B.csv, metrics_C.csv.')
    g.add_argument('--csv', type=Path, nargs='+')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--pattern', default='metrics_*.csv')
    p.add_argument('--n-resamples', type=int, default=N_RESAMPLES)
    return p.parse_args()


def load_metrics(args) -> pd.DataFrame:
    files = (sorted(args.metrics_dir.glob(args.pattern))
             if args.metrics_dir is not None else list(args.csv))
    if not files:
        raise FileNotFoundError('Nenhum CSV de metricas encontrado.')
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    if 'group' not in df.columns:
        raise ValueError("Coluna 'group' ausente.")
    return df


def rank_biserial(d: np.ndarray) -> float:
    """Correlacao rank-biserial pareada (matched-pairs).

    r_rb = (R+ - R-) / (R+ + R-), onde R+/R- sao as somas dos ranks de |d|
    para deltas positivos/negativos (zeros descartados). Tamanho de efeito
    do Wilcoxon signed-rank (Kerby, 2014). Intervalo [-1, 1].
    """
    d = d[d != 0]
    if d.size == 0:
        return float('nan')
    ranks = rankdata(np.abs(d))
    r_pos = ranks[d > 0].sum()
    r_neg = ranks[d < 0].sum()
    total = r_pos + r_neg
    return float((r_pos - r_neg) / total) if total > 0 else float('nan')


def cohen_dz(d: np.ndarray) -> float:
    """d_z de Cohen pareado = mean(d) / std(d) (std amostral, ddof=1)."""
    if d.size < 2:
        return float('nan')
    sd = d.std(ddof=1)
    return float(d.mean() / sd) if sd > 0 else float('nan')


def bca_ci(d: np.ndarray, n_resamples: int):
    """IC 95% BCa da media do delta via scipy.stats.bootstrap."""
    if d.size < 2 or np.allclose(d, d[0]):
        m = float(d.mean()) if d.size else float('nan')
        return m, m
    rng = np.random.default_rng(SEED)
    res = bootstrap((d,), np.mean, method='BCa', n_resamples=n_resamples,
                    confidence_level=0.95, random_state=rng)
    return float(res.confidence_interval.low), float(res.confidence_interval.high)


def analyze(df: pd.DataFrame, n_resamples: int) -> pd.DataFrame:
    by_group = {g: gdf.set_index(KEY) for g, gdf in df.groupby('group')}
    rows = []
    for x, y in PAIRS:
        if x not in by_group or y not in by_group:
            continue
        gx, gy = by_group[x], by_group[y]
        common = gx.index.intersection(gy.index)
        for m in METRICS:
            if m not in gx.columns or m not in gy.columns:
                continue
            dx = gx.loc[common, m]
            dy = gy.loc[common, m]
            d = (dx - dy).to_numpy(dtype=float)
            d = d[~np.isnan(d)]  # descarta pares com NaN (e.g. lesion vazio)
            if d.size == 0:
                continue
            low, high = bca_ci(d, n_resamples)
            rows.append({
                'metric': m,
                'pair': f'{x}-{y}',
                'n_pairs': int(d.size),
                'delta_mean': float(d.mean()),
                'ci_low': low,
                'ci_high': high,
                'rank_biserial': rank_biserial(d),
                'cohen_dz': cohen_dz(d),
            })
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    df = load_metrics(args)
    res = analyze(df, args.n_resamples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.output, index=False)
    with pd.option_context('display.float_format', lambda x: f'{x:.4f}',
                           'display.max_rows', None):
        print(res.to_string(index=False))
    print('\nLeitura: o IC 95% BCa e\' sobre a MEDIA DO DELTA pareado; '
          'rank_biserial e\' o tamanho de efeito do Wilcoxon do S5.9; '
          'cohen_dz e\' o efeito padronizado pareado. Efeito direcional '
          'robusto com magnitude pequena => IC estreito proximo de zero, '
          'mas sem incluir zero.')
    print(f'\nCSV salvo: {args.output}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
