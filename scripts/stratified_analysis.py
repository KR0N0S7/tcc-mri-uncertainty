#!/usr/bin/env python3
# Autor: Massanori
# Data: 12/06/2026
# Descricao: Estratificacao formal por tamanho de lesao (item 2.2 do Bloco 2,
#            "a tabela mais importante do TCC"). Le os 3 CSVs por slice do
#            S5.8 (metrics_A/B/C.csv), atribui cada slice a uma faixa de
#            tamanho a partir da AREA DA MASCARA do slice (coluna
#            n_pixels_lesion, que e o suporte exato onde Coverage_lesion,
#            Width_lesion e ULAS_lesion sao computados) usando os thresholds
#            data-driven ja commitados em configs/lesion_thresholds.json
#            (pequena <200 / media 200-2000 / grande >2000 px^2, definidos
#            so no train para evitar leakage; Demsar, 2006). Para cada
#            (faixa x grupo x metrica) reporta n, media, mediana e IC 95% por
#            bootstrap BCa, com flag de poder estatistico (OK n>=30,
#            BOOTSTRAP 10<=n<30, INSUFICIENTE n<10). Recebe via CLI:
#            --csv-dir, --thresholds (default config), --output-csv,
#            --output-md, --output-fig. Gera: CSV tidy auditavel, tabela
#            Markdown pronta para o docs/, e figura de barras de
#            Coverage_lesion por faixa com IC e linha de cobertura nominal.
#            Sem narrativa hardcoded; n por faixa e independente do grupo
#            (a mascara e a mesma), entao e reportado uma vez.

"""Tabela estratificada por tamanho de lesao (S5 / pre-entrega 5a).

Roda com:
    python scripts/stratified_analysis.py \\
        --csv-dir /caminho/com/metrics_A.csv,_B,_C \\
        --output-csv results/stratified_by_size.csv \\
        --output-md  docs/figures/stratified_by_size.md \\
        --output-fig figures/coverage_by_lesion_size.png

Refs:
    Demsar (2006), JMLR (thresholds definidos so no train; sem leakage).
    Efron & Tibshirani (1993) (BCa bootstrap para ICs com n pequeno).
    Romano et al. (2019), NeurIPS (cobertura nominal 1-alpha de referencia).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import bootstrap

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402

try:
    from src.losses import DEFAULT_ALPHA  # noqa: E402
except Exception:  # pragma: no cover - fallback se import pesado falhar
    DEFAULT_ALPHA = 0.10

GROUPS = ('A', 'B', 'C')
GROUP_LABELS = {'A': 'A (ResM)', 'B': 'B (QR)', 'C': 'C (QR-Lesion)'}

# Metricas de lesao a estratificar (chave no CSV -> rotulo legivel).
METRICS = (
    ('coverage_lesion', 'Coverage_lesion'),
    ('mean_width_lesion', 'Width_lesion'),
    ('ulas_lesion', 'ULAS_lesion'),
)

OUTPUT_FIELDS = [
    'faixa', 'group', 'metric', 'n', 'mean', 'median',
    'ci95_low', 'ci95_high', 'power_status',
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Estratificacao por tamanho de lesao (item 2.2).'
    )
    p.add_argument('--csv-dir', type=Path, required=True,
                   help='Dir com metrics_A.csv, metrics_B.csv, metrics_C.csv.')
    p.add_argument('--thresholds', type=Path, default=None,
                   help='lesion_thresholds.json. Default: configs/ do projeto.')
    p.add_argument('--alpha', type=float, default=DEFAULT_ALPHA,
                   help='alpha p/ linha de cobertura nominal (default 0.10).')
    p.add_argument('--n-bootstrap', type=int, default=10000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output-csv', type=Path,
                   default=ROOT / 'results' / 'stratified_by_size.csv')
    p.add_argument('--output-md', type=Path,
                   default=ROOT / 'docs' / 'figures' / 'stratified_by_size.md')
    p.add_argument('--output-fig', type=Path,
                   default=ROOT / 'figures' / 'coverage_by_lesion_size.png')
    return p.parse_args()


def load_thresholds(path: Path | None) -> dict:
    p = path or (cfg.configs_dir() / 'lesion_thresholds.json')
    payload = json.loads(Path(p).read_text(encoding='utf-8'))
    t = payload['thresholds_px2']
    return {'small': float(t['small']), 'medium': float(t['medium'])}


def make_faixa_fn(t: dict):
    """Fecha sobre os thresholds; bina pela area (px^2) da mascara do slice."""
    small, medium = t['small'], t['medium']
    labels = (
        f'pequena (<{int(small)} px)',
        f'media ({int(small)}-{int(medium)} px)',
        f'grande (>{int(medium)} px)',
    )

    def assign(area: float) -> str:
        if area < small:
            return labels[0]
        if area < medium:
            return labels[1]
        return labels[2]

    return assign, labels


def load_csvs(csv_dir: Path):
    import pandas as pd
    dfs = []
    for g in GROUPS:
        fp = csv_dir / f'metrics_{g}.csv'
        if not fp.is_file():
            print(f'ERRO: nao encontrado: {fp}', file=sys.stderr)
            sys.exit(2)
        df = pd.read_csv(fp)
        if 'group' not in df.columns:
            df['group'] = g
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def bca_ci(values: np.ndarray, n_resamples: int, seed: int) -> tuple:
    """IC 95% por BCa bootstrap. Ref: Efron & Tibshirani (1993).

    Espelha o helper do scripts/analyze_S5_9.py para consistencia. Degrada
    para percentil e depois para (mean, mean) quando o BCa nao e aplicavel
    (n<3 ou variancia ~0).
    """
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) < 3 or np.std(values, ddof=1) < 1e-12:
        m = float(np.mean(values)) if len(values) > 0 else float('nan')
        return m, m
    rng = np.random.default_rng(seed)
    try:
        res = bootstrap((values,), statistic=np.mean, confidence_level=0.95,
                        n_resamples=n_resamples, method='BCa', random_state=rng)
        return float(res.confidence_interval.low), float(res.confidence_interval.high)
    except Exception:
        try:
            res = bootstrap((values,), statistic=np.mean, confidence_level=0.95,
                            n_resamples=n_resamples, method='percentile',
                            random_state=rng)
            return float(res.confidence_interval.low), float(res.confidence_interval.high)
        except Exception:
            return float('nan'), float('nan')


def power_status(n: int) -> str:
    if n >= 30:
        return 'OK'
    if n >= 10:
        return 'BOOTSTRAP'
    return 'INSUFICIENTE'


def compute_rows(df, assign, labels, n_bootstrap: int, seed: int) -> list:
    """Uma linha por (faixa x grupo x metrica)."""
    df = df.copy()
    # Bina pelo tamanho da mascara do slice (px^2), independente do grupo.
    df = df[df['n_pixels_lesion'].fillna(0) > 0].copy()
    df['faixa'] = df['n_pixels_lesion'].astype(float).apply(assign)

    # Offset deterministico por metrica (hash() de str varia entre processos
    # via PYTHONHASHSEED, o que quebraria a reprodutibilidade do bootstrap).
    metric_offset = {key: i * 17 for i, (key, _l) in enumerate(METRICS)}

    rows = []
    for fi, faixa in enumerate(labels):
        for g in GROUPS:
            sub = df[(df['faixa'] == faixa) & (df['group'] == g)]
            for key, _label in METRICS:
                vals = sub[key].to_numpy(dtype=float)
                vals = vals[~np.isnan(vals)]
                n = int(len(vals))
                if n == 0:
                    rows.append({'faixa': faixa, 'group': g, 'metric': key,
                                 'n': 0, 'mean': float('nan'),
                                 'median': float('nan'),
                                 'ci95_low': float('nan'),
                                 'ci95_high': float('nan'),
                                 'power_status': 'INSUFICIENTE'})
                    continue
                seed_g = seed + ord(g) + metric_offset[key] + fi * 101
                lo, hi = bca_ci(vals, n_bootstrap, seed_g)
                rows.append({
                    'faixa': faixa, 'group': g, 'metric': key, 'n': n,
                    'mean': float(np.mean(vals)),
                    'median': float(np.median(vals)),
                    'ci95_low': lo, 'ci95_high': hi,
                    'power_status': power_status(n),
                })
    return rows


def write_csv(rows: list, out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({
                'faixa': r['faixa'], 'group': r['group'], 'metric': r['metric'],
                'n': r['n'],
                'mean': f'{r["mean"]:.6f}' if not np.isnan(r['mean']) else '',
                'median': f'{r["median"]:.6f}' if not np.isnan(r['median']) else '',
                'ci95_low': f'{r["ci95_low"]:.6f}' if not np.isnan(r['ci95_low']) else '',
                'ci95_high': f'{r["ci95_high"]:.6f}' if not np.isnan(r['ci95_high']) else '',
                'power_status': r['power_status'],
            })
    print(f'CSV tidy: {out_csv} ({len(rows)} linhas)')


def _cell(r: dict) -> str:
    if r['n'] == 0 or np.isnan(r['mean']):
        return 'n/a'
    flag = '' if r['power_status'] == 'OK' else (
        ' *' if r['power_status'] == 'BOOTSTRAP' else ' **')
    return (f'{r["mean"]:.3f} [{r["ci95_low"]:.3f}, {r["ci95_high"]:.3f}]'
            f' (n={r["n"]}){flag}')


def render_md(rows: list, labels: tuple, n_by_faixa: dict, alpha: float,
              thresholds: dict, out_md: Path):
    idx = {(r['faixa'], r['group'], r['metric']): r for r in rows}
    lines = []
    lines.append('# Estratificacao por tamanho de lesao (item 2.2)')
    lines.append('')
    lines.append(f'Faixas (px^2, area da mascara do slice): pequena <'
                 f'{int(thresholds["small"])}, media '
                 f'{int(thresholds["small"])}-{int(thresholds["medium"])}, '
                 f'grande >{int(thresholds["medium"])}. Thresholds '
                 'data-driven definidos so no train (Demsar, 2006). '
                 f'Cobertura nominal de referencia = {1 - alpha:.2f}.')
    lines.append('')
    lines.append('IC 95% por bootstrap BCa (Efron & Tibshirani, 1993). '
                 'Flags de poder: sem flag n>=30; `*` 10<=n<30 (BCa, '
                 'interpretar com cautela); `**` n<10 (INSUFICIENTE).')
    lines.append('')
    lines.append('n por faixa (identico entre grupos; a mascara e a mesma): '
                 + ', '.join(f'{k} = {v}' for k, v in n_by_faixa.items()))
    lines.append('')

    for key, label in METRICS:
        lines.append(f'## {label}')
        lines.append('')
        lines.append('| Faixa | A (ResM) | B (QR) | C (QR-Lesion) |')
        lines.append('|---|---|---|---|')
        for faixa in labels:
            cells = [_cell(idx[(faixa, g, key)]) for g in GROUPS]
            lines.append(f'| {faixa} | ' + ' | '.join(cells) + ' |')
        lines.append('')

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text('\n'.join(lines), encoding='utf-8')
    print(f'Markdown: {out_md}')


def render_fig(rows: list, labels: tuple, alpha: float, out_fig: Path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    idx = {(r['faixa'], r['group'], 'coverage_lesion'): r for r in rows}
    x = np.arange(len(labels))
    width = 0.26
    colors = {'A': '#4472C4', 'B': '#ED7D31', 'C': '#70AD47'}

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, g in enumerate(GROUPS):
        means, los, his = [], [], []
        for faixa in labels:
            r = idx.get((faixa, g, 'coverage_lesion'))
            m = r['mean'] if r and not np.isnan(r['mean']) else np.nan
            means.append(m)
            los.append((m - r['ci95_low']) if r and not np.isnan(r['ci95_low']) else 0)
            his.append((r['ci95_high'] - m) if r and not np.isnan(r['ci95_high']) else 0)
        ax.bar(x + (i - 1) * width, means, width, label=GROUP_LABELS[g],
               color=colors[g], yerr=[los, his], capsize=4, alpha=0.9)

    ax.axhline(1 - alpha, color='red', linestyle='--', linewidth=1.5,
               label=f'Cobertura nominal ({1 - alpha:.2f})')
    ax.set_xticks(x)
    ax.set_xticklabels([f.split(' ')[0] for f in labels], fontsize=12)
    ax.set_ylabel('Coverage_lesion', fontsize=12)
    ax.set_xlabel('Faixa de tamanho da lesao', fontsize=12)
    ax.set_title('Cobertura em lesao por faixa de tamanho (IC 95% BCa)',
                 fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Figura: {out_fig}')


def main() -> int:
    args = parse_args()
    thresholds = load_thresholds(args.thresholds)
    assign, labels = make_faixa_fn(thresholds)
    print(f'Thresholds (px^2): {thresholds}; alpha={args.alpha}')

    df = load_csvs(args.csv_dir)
    rows = compute_rows(df, assign, labels, args.n_bootstrap, args.seed)

    # n por faixa (do grupo A; identico entre grupos).
    n_by_faixa = {}
    for faixa in labels:
        rec = next((r for r in rows
                    if r['faixa'] == faixa and r['group'] == 'A'
                    and r['metric'] == 'coverage_lesion'), None)
        n_by_faixa[faixa.split(' ')[0]] = rec['n'] if rec else 0

    write_csv(rows, args.output_csv)
    render_md(rows, labels, n_by_faixa, args.alpha, thresholds, args.output_md)
    try:
        render_fig(rows, labels, args.alpha, args.output_fig)
    except Exception as e:  # figura e secundaria; nao derruba o pipeline
        print(f'AVISO: figura falhou ({e}); CSV/MD gerados.', file=sys.stderr)
    print('Estratificacao concluida.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
