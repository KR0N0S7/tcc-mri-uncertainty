#!/usr/bin/env python3
# Autor: Massanori
# Data: 12/06/2026
# Descricao: Orquestrador do ablation do fator de ponderacao lambda da loss
#            do Grupo C (item 2.1 do Bloco 2). Para cada lambda novo (default
#            {3,10,15}) executa, de forma IDEMPOTENTE e resumivel (estilo
#            Kaggle): treino (train_qr_lesion.py, 210k iters, mesma config
#            das ancoras) -> calibracao (calibrate.py, CQR no split cal) ->
#            metricas por slice (compute_metrics.py no split test). Reaproveita
#            os scripts ja commitados via subprocess (nao reimplementa treino).
#            As ancoras de custo zero entram da analise existente: lambda=1 e
#            o Grupo B (peso w=1+(lambda-1)*M => w=1; loss identica ao qr_loss)
#            e lambda=5 e o Grupo C ja treinado. Ao final monta
#            results/ablation_lambda.csv (lambda x metrica: n, media, IC 95%
#            BCa) e plota ULAS_lesion, Width_lesion e Coverage_lesion vs lambda
#            (figures/ablation_lambda_*.png, 300 DPI). Justificativa academica:
#            a curva separa "escolhi 5" de "5 e o melhor trade-off". CRITICO:
#            todos os lambda usam os MESMOS 210k iters das ancoras, senao a
#            curva confunde efeito do lambda com tempo de treino.
#            Recebe via CLI: --lambdas, --total-iters, --recons-dir,
#            --masks-dir, --config, --work-dir, --metrics-b, --metrics-c,
#            --no-train (so monta a curva do que ja existe). Gera CSV + PNGs.

"""Ablation do lambda da loss do Grupo C (item 2.1).

Roda no Kaggle (treino) — idempotente, retoma de last.pt:
    python scripts/ablation_lambda.py \\
        --lambdas 3 10 15 --total-iters 210000 \\
        --recons-dir /kaggle/input/.../recons \\
        --masks-dir  /kaggle/input/.../masks \\
        --work-dir   /kaggle/working/ablation \\
        --metrics-b  /kaggle/input/.../metrics_B.csv \\
        --metrics-c  /kaggle/input/.../metrics_C.csv

So montar a curva/figuras a partir de CSVs ja calculados (CPU):
    python scripts/ablation_lambda.py --no-train \\
        --work-dir results/ablation \\
        --metrics-b metrics_B.csv --metrics-c metrics_C.csv --lambdas 3 10 15

Refs:
    Romano et al. (2019), NeurIPS (CQR). Efron & Tibshirani (1993) (BCa).
    Demsar (2006), JMLR (seed fixa entre runs isola o efeito do lambda).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.stats import bootstrap

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Ancoras de custo zero: lambda -> grupo ja treinado/analisado.
ANCHORS = {1.0: 'B', 5.0: 'C'}

# Metricas plotadas (chave no CSV -> (rotulo, "maior melhor"?)).
METRICS = (
    ('ulas_lesion', 'ULAS_lesion'),
    ('mean_width_lesion', 'Width_lesion'),
    ('coverage_lesion', 'Coverage_lesion'),
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Ablation do lambda (item 2.1).')
    p.add_argument('--lambdas', type=float, nargs='+', default=[3.0, 10.0, 15.0],
                   help='Lambdas NOVOS a treinar (default 3 10 15).')
    p.add_argument('--total-iters', type=int, default=210000,
                   help='MESMO valor das ancoras (NAO reduzir; confunde a curva).')
    p.add_argument('--recons-dir', type=Path, default=None)
    p.add_argument('--masks-dir', type=Path, default=None)
    p.add_argument('--config', type=Path,
                   default=ROOT / 'configs' / 'training_base.json')
    p.add_argument('--work-dir', type=Path, default=ROOT / 'results' / 'ablation',
                   help='Onde ficam run-dirs, q_hats e metrics_lambda*.csv.')
    p.add_argument('--metrics-b', type=Path, default=None,
                   help='metrics_B.csv (ancora lambda=1).')
    p.add_argument('--metrics-c', type=Path, default=None,
                   help='metrics_C.csv (ancora lambda=5).')
    p.add_argument('--no-train', action='store_true',
                   help='Nao treina; so monta a curva dos CSVs existentes.')
    p.add_argument('--alpha', type=float, default=0.10)
    p.add_argument('--n-bootstrap', type=int, default=10000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda', choices=['cuda', 'cpu', 'auto'])
    p.add_argument('--chans', type=int, default=32)
    p.add_argument('--num-pool-layers', type=int, default=4)
    p.add_argument('--num-workers', type=int, default=2)
    p.add_argument('--output-csv', type=Path, default=None)
    p.add_argument('--output-fig-dir', type=Path, default=ROOT / 'figures')
    return p.parse_args()


def _run(cmd: list) -> int:
    print('>>> ' + ' '.join(str(c) for c in cmd), flush=True)
    return subprocess.call([str(c) for c in cmd])


# --------------------------------------------------------------------------- #
# Etapa GPU: treina -> calibra -> metricas, por lambda novo (idempotente)
# --------------------------------------------------------------------------- #
def metrics_path_for(work_dir: Path, lam: float) -> Path:
    return work_dir / f'metrics_lambda{lam:g}.csv'


def process_lambda(lam: float, args) -> Path:
    """Garante metrics_lambda{lam}.csv. Pula etapas com saida ja existente."""
    work = args.work_dir
    work.mkdir(parents=True, exist_ok=True)
    run_dir = work / f'group_c_lambda{lam:g}'
    ckpt = run_dir / 'best.pt'
    qhat = work / f'q_hat_lambda{lam:g}.json'
    metrics = metrics_path_for(work, lam)

    if metrics.is_file():
        print(f'[lambda={lam:g}] metrics ja existe -> pulando treino/cal.')
        return metrics

    if args.no_train:
        raise FileNotFoundError(
            f'--no-train, mas {metrics} nao existe. Rode o treino primeiro.')

    py = sys.executable
    # 1) Treino (resume automatico de last.pt; re-run apos 210k e no-op).
    if not ckpt.is_file():
        rc = _run([py, ROOT / 'scripts' / 'train_qr_lesion.py',
                   '--lambda-lesion', lam, '--run-dir', run_dir,
                   '--recons-dir', args.recons_dir, '--masks-dir', args.masks_dir,
                   '--total-iters', args.total_iters, '--config', args.config,
                   '--device', args.device, '--num-workers', args.num_workers,
                   '--alpha', args.alpha])
        if rc != 0 or not ckpt.is_file():
            raise RuntimeError(f'[lambda={lam:g}] treino falhou (rc={rc}).')

    # 2) Calibracao CQR no split cal.
    if not qhat.is_file():
        rc = _run([py, ROOT / 'scripts' / 'calibrate.py', '--group', 'C',
                   '--checkpoint', ckpt, '--recons-dir', args.recons_dir,
                   '--masks-dir', args.masks_dir, '--output', qhat,
                   '--alpha', args.alpha, '--device', args.device,
                   '--chans', args.chans, '--num-pool-layers', args.num_pool_layers])
        if rc != 0 or not qhat.is_file():
            raise RuntimeError(f'[lambda={lam:g}] calibracao falhou (rc={rc}).')

    # 3) Metricas por slice no split test.
    rc = _run([py, ROOT / 'scripts' / 'compute_metrics.py', '--group', 'C',
               '--checkpoint', ckpt, '--qhat', qhat,
               '--recons-dir', args.recons_dir, '--masks-dir', args.masks_dir,
               '--output', metrics, '--alpha', args.alpha,
               '--device', args.device, '--num-workers', args.num_workers,
               '--chans', args.chans, '--num-pool-layers', args.num_pool_layers])
    if rc != 0 or not metrics.is_file():
        raise RuntimeError(f'[lambda={lam:g}] compute_metrics falhou (rc={rc}).')
    return metrics


# --------------------------------------------------------------------------- #
# Etapa CPU: monta a curva e plota
# --------------------------------------------------------------------------- #
def bca_ci(values: np.ndarray, n_resamples: int, seed: int) -> tuple:
    """IC 95% BCa. Espelha analyze_S5_9/stratified_analysis."""
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


def summarize_csv(csv_path: Path, lam: float, source: str,
                  n_bootstrap: int, seed: int) -> list:
    """Uma linha por metrica para um dado lambda (so slices com lesao)."""
    import pandas as pd
    df = pd.read_csv(csv_path)
    df = df[df['n_pixels_lesion'].fillna(0) > 0]
    rows = []
    for mi, (key, _label) in enumerate(METRICS):
        vals = df[key].to_numpy(dtype=float)
        vals = vals[~np.isnan(vals)]
        n = int(len(vals))
        if n == 0:
            rows.append({'lambda': lam, 'source': source, 'metric': key,
                         'n': 0, 'mean': float('nan'),
                         'ci95_low': float('nan'), 'ci95_high': float('nan')})
            continue
        lo, hi = bca_ci(vals, n_bootstrap, seed + mi * 17 + int(lam * 7))
        rows.append({'lambda': lam, 'source': source, 'metric': key, 'n': n,
                     'mean': float(np.mean(vals)),
                     'ci95_low': lo, 'ci95_high': hi})
    return rows


def assemble(args) -> list:
    """Coleta resumos de todas as ancoras + lambdas novos."""
    rows = []
    # Ancoras (lambda=1 -> B, lambda=5 -> C).
    anchor_csv = {1.0: args.metrics_b, 5.0: args.metrics_c}
    for lam, group in ANCHORS.items():
        csvp = anchor_csv.get(lam)
        if csvp is None or not Path(csvp).is_file():
            print(f'AVISO: ancora lambda={lam:g} (grupo {group}) sem CSV; '
                  f'ponto ausente na curva.', file=sys.stderr)
            continue
        rows += summarize_csv(Path(csvp), lam, f'ancora ({group})',
                              args.n_bootstrap, args.seed)
    # Lambdas novos.
    for lam in args.lambdas:
        mp = metrics_path_for(args.work_dir, lam)
        if not mp.is_file():
            print(f'AVISO: metrics de lambda={lam:g} ausente ({mp}).',
                  file=sys.stderr)
            continue
        rows += summarize_csv(mp, lam, 'novo', args.n_bootstrap, args.seed)
    return rows


def write_csv(rows: list, out_csv: Path):
    import csv as _csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ['lambda', 'source', 'metric', 'n', 'mean', 'ci95_low', 'ci95_high']
    with out_csv.open('w', newline='', encoding='utf-8') as f:
        w = _csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in sorted(rows, key=lambda d: (d['metric'], d['lambda'])):
            w.writerow({k: (f'{r[k]:.6f}' if isinstance(r[k], float)
                            and k in ('mean', 'ci95_low', 'ci95_high')
                            and not np.isnan(r[k]) else
                            ('' if (isinstance(r[k], float) and np.isnan(r[k]))
                             else r[k])) for k in fields})
    print(f'CSV da curva: {out_csv} ({len(rows)} linhas)')


def plot_curves(rows: list, alpha: float, out_dir: Path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    by_metric = {}
    for r in rows:
        by_metric.setdefault(r['metric'], []).append(r)

    for key, label in METRICS:
        pts = sorted([r for r in by_metric.get(key, []) if r['n'] > 0],
                     key=lambda d: d['lambda'])
        if not pts:
            continue
        xs = [p['lambda'] for p in pts]
        ys = [p['mean'] for p in pts]
        lo = [p['mean'] - p['ci95_low'] for p in pts]
        hi = [p['ci95_high'] - p['mean'] for p in pts]
        is_anchor = [p['source'].startswith('ancora') for p in pts]

        fig, ax = plt.subplots(figsize=(7.5, 5))
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt='-o', color='#4472C4',
                    capsize=4, linewidth=1.5, markersize=7, label=label)
        # Destaca as ancoras (B/C) com marcador diferente.
        ax_anchor_x = [x for x, a in zip(xs, is_anchor) if a]
        ax_anchor_y = [y for y, a in zip(ys, is_anchor) if a]
        if ax_anchor_x:
            ax.scatter(ax_anchor_x, ax_anchor_y, s=160, facecolors='none',
                       edgecolors='red', linewidths=1.8, zorder=5,
                       label='ancora (B: lambda=1, C: lambda=5)')
        if key == 'coverage_lesion':
            ax.axhline(1 - alpha, color='gray', linestyle='--', linewidth=1.2,
                       label=f'cobertura nominal ({1 - alpha:.2f})')
        ax.set_xlabel('lambda (peso da lesao na loss)', fontsize=12)
        ax.set_ylabel(label, fontsize=12)
        ax.set_title(f'{label} vs lambda (IC 95% BCa)', fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        out_png = out_dir / f'ablation_lambda_{key}.png'
        fig.savefig(out_png, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f'Figura: {out_png}')


def main() -> int:
    args = parse_args()
    print(f'Lambdas novos: {args.lambdas} | total_iters={args.total_iters} | '
          f'no_train={args.no_train}')
    print(f'Ancoras: lambda=1 -> Grupo B, lambda=5 -> Grupo C (custo zero)')

    if not args.no_train:
        for lam in args.lambdas:
            if lam in ANCHORS:
                print(f'[lambda={lam:g}] e ancora ({ANCHORS[lam]}); '
                      f'nao retreina.')
                continue
            process_lambda(lam, args)

    rows = assemble(args)
    out_csv = args.output_csv or (args.work_dir / 'ablation_lambda.csv')
    write_csv(rows, out_csv)
    try:
        plot_curves(rows, args.alpha, args.output_fig_dir)
    except Exception as e:
        print(f'AVISO: plot falhou ({e}); CSV gerado.', file=sys.stderr)
    print('Ablation concluido.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
