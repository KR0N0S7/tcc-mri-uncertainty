# Autor: Massanori
# Data: 02/06/2026
# Descrição: Analise de cobertura condicional, destino:
#            scripts/analyze_conditional_coverage.py. Le os metrics_*.csv por
#            slice do S5.8 (colunas coverage_global, coverage_lesion,
#            n_pixels_total, n_pixels_lesion, sequence) e reporta a cobertura
#            condicional como proporcao binomial exata (Clopper-Pearson) por
#            estrato: (i) lesao vs global (o "gap" condicional — resultado
#            central), (ii) por tipo de sequencia (AXFLAIR/AXT1/AXT1POST) e
#            (iii) por tercil de carga de lesao no slice (proxy slice-level;
#            estratificacao por AREA de lesao individual exige passada nas
#            mascaras e e' deixada como extensao). Recebe via CLI: --metrics-dir
#            ou --csv (1+ arquivos), --output. Gera: CSV com [grupo, estrato,
#            n_pixels, cobertura, ci_low, ci_high] + sumario impresso do gap.
#            IMPORTANTE: cobertura condicional EXATA e' impossivel em
#            distribution-free sem suposicoes extras (Barber et al., 2021);
#            aqui usamos estratos discretos finitos (estilo Mondrian, Vovk et
#            al., 2005), que entregam cobertura condicional ao estrato.
#            Fundamentos: Clopper & Pearson (1934); Barber et al. (2021);
#            Vovk et al. (2005). Sem dados pessoais.

"""Cobertura condicional por estrato com intervalos Clopper-Pearson.

O gap entre cobertura marginal (global) e condicional (em lesao) e' um dos
resultados centrais do trabalho: quantifica o quanto a garantia marginal do
conformal prediction "falha" justamente onde importa clinicamente. Barber et
al. (2021) provam que cobertura condicional exata e' inatingivel em
distribution-free sem suposicoes; por isso reportamos cobertura condicional a
estratos discretos finitos (lesao, sequencia, carga), que e' atingivel
(particionamento estilo Mondrian, Vovk et al., 2005).

Reconstrucao das contagens binomiais: o CSV do S5.8 guarda cobertura como
fracao por slice. Recuperamos contagens inteiras aproximadas por
  covered = round(coverage * n_pixels)
e agregamos por estrato (micro-average), aplicando Clopper-Pearson exato.
O arredondamento por slice introduz erro <= 0.5 pixel/slice, desprezivel
frente aos ~10^6-10^7 pixels por estrato.

Refs:
    Clopper, C.J.; Pearson, E.S. (1934). The use of confidence or fiducial
        limits illustrated in the case of the binomial. Biometrika 26(4).
    Barber, R.F. et al. (2021). The limits of distribution-free conditional
        predictive inference. Information and Inference 10(2):455-482.
    Vovk, V.; Gammerman, A.; Shafer, G. (2005). Algorithmic Learning in a
        Random World. Springer. (Mondrian conformal prediction)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

CONF_LEVEL = 0.95


def clopper_pearson(k: int, n: int, conf: float = CONF_LEVEL):
    """Intervalo de confianca binomial exato (Clopper-Pearson).

    Parameters
    ----------
    k : int   sucessos (pixels cobertos)
    n : int   tentativas (pixels no estrato)
    conf : float  nivel de confianca (default 0.95)

    Returns
    -------
    (p_hat, low, high)
    """
    if n == 0:
        return float('nan'), float('nan'), float('nan')
    alpha = 1.0 - conf
    p_hat = k / n
    low = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    high = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
    return float(p_hat), float(low), float(high)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Cobertura condicional por estrato.')
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--metrics-dir', type=Path,
                   help='Dir com metrics_A.csv, metrics_B.csv, metrics_C.csv.')
    g.add_argument('--csv', type=Path, nargs='+',
                   help='Lista explicita de CSVs por slice.')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--pattern', default='metrics_*.csv')
    return p.parse_args()


def load_metrics(args) -> pd.DataFrame:
    if args.metrics_dir is not None:
        files = sorted(args.metrics_dir.glob(args.pattern))
    else:
        files = list(args.csv)
    if not files:
        raise FileNotFoundError('Nenhum CSV de metricas encontrado.')
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    required = {'group', 'sequence', 'coverage_global', 'coverage_lesion',
                'n_pixels_total', 'n_pixels_lesion'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Colunas ausentes nos CSVs: {sorted(missing)}')
    return df


def _covered_counts(frac: pd.Series, n: pd.Series) -> np.ndarray:
    """Contagem inteira de cobertos = round(frac * n), tratando NaN como 0/0."""
    frac = frac.fillna(0.0).to_numpy()
    n = n.fillna(0).to_numpy()
    return np.rint(frac * n).astype(np.int64)


def _row(group: str, stratum: str, k: int, n: int) -> dict:
    p_hat, low, high = clopper_pearson(int(k), int(n))
    return {'group': group, 'stratum': stratum, 'covered': int(k),
            'n_pixels': int(n), 'coverage': p_hat,
            'ci_low': low, 'ci_high': high, 'ci_width': high - low}


def analyze(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group, gdf in df.groupby('group'):
        # ---- (i) global vs lesao ----
        cov_g = _covered_counts(gdf['coverage_global'], gdf['n_pixels_total'])
        n_g = gdf['n_pixels_total'].fillna(0).astype(np.int64).to_numpy()
        rows.append(_row(group, 'global', cov_g.sum(), n_g.sum()))

        les = gdf[gdf['n_pixels_lesion'] > 0]
        cov_l = _covered_counts(les['coverage_lesion'], les['n_pixels_lesion'])
        n_l = les['n_pixels_lesion'].astype(np.int64).to_numpy()
        rows.append(_row(group, 'lesion', cov_l.sum(), n_l.sum()))

        # ---- (ii) por sequencia (em lesao) ----
        for seq, sdf in les.groupby('sequence'):
            ck = _covered_counts(sdf['coverage_lesion'], sdf['n_pixels_lesion'])
            nn = sdf['n_pixels_lesion'].astype(np.int64).to_numpy()
            rows.append(_row(group, f'lesion::seq={seq}', ck.sum(), nn.sum()))

        # ---- (iii) por tercil de carga de lesao no slice (proxy) ----
        if len(les) >= 3:
            q1, q2 = les['n_pixels_lesion'].quantile([1/3, 2/3]).to_numpy()
            bins = [('carga_baixa', les['n_pixels_lesion'] <= q1),
                    ('carga_media', (les['n_pixels_lesion'] > q1) &
                                    (les['n_pixels_lesion'] <= q2)),
                    ('carga_alta', les['n_pixels_lesion'] > q2)]
            for name, mask in bins:
                bdf = les[mask]
                if bdf.empty:
                    continue
                ck = _covered_counts(bdf['coverage_lesion'], bdf['n_pixels_lesion'])
                nn = bdf['n_pixels_lesion'].astype(np.int64).to_numpy()
                rows.append(_row(group, f'lesion::{name}', ck.sum(), nn.sum()))
    return pd.DataFrame(rows)


def print_gap(res: pd.DataFrame) -> None:
    print('\n=== Gap de cobertura condicional (global - lesao) ===')
    for group in sorted(res['group'].unique()):
        g = res[res['group'] == group]
        cg = g[g['stratum'] == 'global']
        cl = g[g['stratum'] == 'lesion']
        if len(cg) and len(cl):
            gap = cg['coverage'].iloc[0] - cl['coverage'].iloc[0]
            print(f'  Grupo {group}: global={cg["coverage"].iloc[0]:.4f} '
                  f'[{cg["ci_low"].iloc[0]:.4f}, {cg["ci_high"].iloc[0]:.4f}] | '
                  f'lesao={cl["coverage"].iloc[0]:.4f} '
                  f'[{cl["ci_low"].iloc[0]:.4f}, {cl["ci_high"].iloc[0]:.4f}] | '
                  f'gap={gap:+.4f}')
    print('\nNota: cobertura condicional EXATA e\' impossivel em distribution-free '
          '(Barber et al., 2021); estes estratos sao a versao atingivel '
          '(condicional a estrato discreto, estilo Mondrian).')


def main() -> int:
    args = parse_args()
    df = load_metrics(args)
    res = analyze(df)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.output, index=False)
    with pd.option_context('display.float_format', lambda x: f'{x:.4f}',
                           'display.max_rows', None):
        print(res.to_string(index=False))
    print_gap(res)
    print(f'\nCSV salvo: {args.output}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
