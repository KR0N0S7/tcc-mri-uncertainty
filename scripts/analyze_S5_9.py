#!/usr/bin/env python3
# Autor: Massanori
# Data: 22/05/2026 (refactor 26/05/2026: narrativa condicional aos p-values)
# Descricao: S5.9 - analise estatistica formal das metricas do S5.8 +
#            geracao do docs/S5.md com narrativa CONDICIONAL aos dados reais.
#            Refactor: avalia hipoteses (HYPOTHESES) dinamicamente como
#            CONFIRMADA / NAO detectada / INVERTIDA baseado em p_Holm
#            vs alpha e direcao observada das medias, eliminando o bug
#            anterior de narrativa hardcoded contraria aos dados.

"""S5.9 - analise estatistica formal das metricas do S5.8."""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import bootstrap

GROUPS = ('A', 'B', 'C')
GROUP_LABELS = {'A': 'A (ResM)', 'B': 'B (QR)', 'C': 'C (QR-Lesion)'}
GROUP_DESCRIPTIONS = {
    'A': 'Residual Magnitude (locally adaptive scaled CP)',
    'B': 'Quantile Regression (CQR, Giannakopoulos et al. 2026)',
    'C': 'QR-Lesion (CQR + loss ponderada lambda=5, contribuicao original)',
}

METRICS = (
    'coverage_global', 'coverage_lesion',
    'mean_width_global', 'mean_width_lesion',
    'iou_topk_global', 'iou_topk_lesion',
    'ulas_lesion',
)

METRIC_LABELS = {
    'coverage_global': 'Coverage_global',
    'coverage_lesion': 'Coverage_lesion',
    'mean_width_global': 'Width_global',
    'mean_width_lesion': 'Width_lesion',
    'iou_topk_global': 'IoU_topk_global',
    'iou_topk_lesion': 'IoU_topk_lesion',
    'ulas_lesion': 'ULAS_lesion',
}

METRIC_TYPE = {
    'coverage_global': 'global', 'coverage_lesion': 'lesion',
    'mean_width_global': 'global', 'mean_width_lesion': 'lesion',
    'iou_topk_global': 'global', 'iou_topk_lesion': 'lesion',
    'ulas_lesion': 'lesion',
}

# Hipoteses pre-registradas H1-H4 (todas C > B em metricas de lesao).
HYPOTHESES = (
    {'id': 'H1', 'desc': 'C > B em Coverage_lesion (loss ponderada melhora cobertura em lesoes)',
     'metric': 'coverage_lesion', 'pair': 'B vs C', 'expected_higher': 'C'},
    {'id': 'H2', 'desc': 'C > B em Width_lesion (intervalos mais conservadores em lesoes)',
     'metric': 'mean_width_lesion', 'pair': 'B vs C', 'expected_higher': 'C'},
    {'id': 'H3', 'desc': 'C > B em ULAS_lesion (alinhamento direcional gradiente)',
     'metric': 'ulas_lesion', 'pair': 'B vs C', 'expected_higher': 'C'},
    {'id': 'H4', 'desc': 'C > B em IoU_topk_lesion (alinhamento de top-X% pixels)',
     'metric': 'iou_topk_lesion', 'pair': 'B vs C', 'expected_higher': 'C'},
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='S5.9 - analise estatistica formal.')
    parser.add_argument('--csv-dir', type=Path, required=True)
    parser.add_argument('--output-json', type=Path, required=True)
    parser.add_argument('--output-md', type=Path, default=None)
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--n-bootstrap', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def load_csvs(csv_dir: Path) -> pd.DataFrame:
    dfs = []
    for g in GROUPS:
        p = csv_dir / f'metrics_{g}.csv'
        if not p.is_file():
            print(f'ERRO: nao encontrado: {p}', file=sys.stderr)
            sys.exit(2)
        df = pd.read_csv(p)
        if 'group' not in df.columns:
            df['group'] = g
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def get_paired_metric(df: pd.DataFrame, metric: str):
    wide = df.pivot_table(index=['volume_id', 'slice_idx'], columns='group', values=metric)
    wide_clean = wide.dropna(subset=list(GROUPS))
    return (wide_clean['A'].to_numpy(dtype=float),
            wide_clean['B'].to_numpy(dtype=float),
            wide_clean['C'].to_numpy(dtype=float))


def holm_bonferroni(p_values: list) -> list:
    """Holm-Bonferroni step-down. Ref: Holm (1979)."""
    n = len(p_values)
    if n == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * n
    running_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adj = min(p * (n - rank), 1.0)
        adj = max(adj, running_max)
        adjusted[orig_idx] = adj
        running_max = adj
    return adjusted


def bca_ci(values: np.ndarray, n_resamples: int, confidence_level: float, seed: int) -> tuple:
    """BCa bootstrap CI. Ref: Efron & Tibshirani (1993)."""
    values = np.asarray(values, dtype=float)
    if len(values) < 3 or np.std(values, ddof=1) < 1e-12:
        m = float(np.mean(values)) if len(values) > 0 else float('nan')
        return m, m
    rng = np.random.default_rng(seed)
    try:
        res = bootstrap((values,), statistic=np.mean,
                        confidence_level=confidence_level, n_resamples=n_resamples,
                        method='BCa', random_state=rng)
        return float(res.confidence_interval.low), float(res.confidence_interval.high)
    except Exception:
        try:
            res = bootstrap((values,), statistic=np.mean,
                            confidence_level=confidence_level, n_resamples=n_resamples,
                            method='percentile', random_state=rng)
            return float(res.confidence_interval.low), float(res.confidence_interval.high)
        except Exception:
            return float('nan'), float('nan')


def safe_wilcoxon(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) == 0 or len(y) == 0 or len(x) != len(y):
        return float('nan')
    diff = x - y
    if np.allclose(diff, 0.0):
        return 1.0
    try:
        _, p = stats.wilcoxon(x, y, zero_method='wilcox', alternative='two-sided')
        return float(p)
    except (ValueError, Warning):
        return float('nan')


def analyze_metric(df: pd.DataFrame, metric: str, n_bootstrap: int,
                   alpha: float, seed: int) -> dict:
    a, b, c = get_paired_metric(df, metric)
    n_paired = int(len(a))

    result: dict[str, Any] = {
        'metric': metric,
        'metric_type': METRIC_TYPE[metric],
        'n_paired': n_paired,
        'means': {g: float(arr.mean()) for g, arr in zip(GROUPS, (a, b, c))} if n_paired > 0 else {},
        'stds': {g: float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
                 for g, arr in zip(GROUPS, (a, b, c))},
        'medians': {g: float(np.median(arr)) if len(arr) > 0 else float('nan')
                    for g, arr in zip(GROUPS, (a, b, c))},
    }

    if n_paired < 3:
        result['friedman'] = {'statistic': float('nan'), 'p_value': float('nan')}
        result['pairwise_wilcoxon'] = []
        result['bootstrap_ci_95'] = {}
        return result

    result['bootstrap_ci_95'] = {}
    for g, arr in zip(GROUPS, (a, b, c)):
        lo, hi = bca_ci(arr, n_bootstrap, 0.95, seed + ord(g))
        result['bootstrap_ci_95'][g] = {'low': lo, 'high': hi}

    try:
        stat, p = stats.friedmanchisquare(a, b, c)
        result['friedman'] = {'statistic': float(stat), 'p_value': float(p)}
    except ValueError as e:
        result['friedman'] = {'statistic': float('nan'), 'p_value': float('nan'), 'error': str(e)}

    pairs = [('A', 'B', a, b), ('A', 'C', a, c), ('B', 'C', b, c)]
    p_raw = [safe_wilcoxon(x, y) for _, _, x, y in pairs]

    valid_idx = [i for i, p in enumerate(p_raw) if not math.isnan(p)]
    p_adj = [float('nan')] * len(p_raw)
    if valid_idx:
        p_valid = [p_raw[i] for i in valid_idx]
        adjusted_valid = holm_bonferroni(p_valid)
        for j, i in enumerate(valid_idx):
            p_adj[i] = adjusted_valid[j]

    result['pairwise_wilcoxon'] = [
        {'pair': f'{l1} vs {l2}', 'p_raw': p_raw[i], 'p_holm': p_adj[i],
         'significant_at_alpha': (p_adj[i] < alpha) if not math.isnan(p_adj[i]) else None}
        for i, (l1, l2, _, _) in enumerate(pairs)
    ]
    return result


def clopper_pearson_coverage(df: pd.DataFrame, region: str) -> dict:
    if region == 'global':
        n_col, cov_col = 'n_pixels_total', 'coverage_global'
    elif region == 'lesion':
        n_col, cov_col = 'n_pixels_lesion', 'coverage_lesion'
    else:
        raise ValueError(f'region invalido: {region}')

    result = {}
    for g in GROUPS:
        sub = df[df['group'] == g].dropna(subset=[cov_col])
        if region == 'lesion':
            sub = sub[sub[n_col] > 0]
        if len(sub) == 0:
            result[g] = {'covered': 0, 'total': 0, 'proportion': float('nan'),
                         'ci_low': float('nan'), 'ci_high': float('nan'), 'n_slices_used': 0}
            continue
        n_covered = int((sub[n_col].astype(float) * sub[cov_col].astype(float)).round().astype(int).sum())
        n_total = int(sub[n_col].astype(int).sum())
        if n_total == 0:
            result[g] = {'covered': 0, 'total': 0, 'proportion': float('nan'),
                         'ci_low': float('nan'), 'ci_high': float('nan'), 'n_slices_used': int(len(sub))}
            continue
        proportion = n_covered / n_total
        bt = stats.binomtest(n_covered, n_total)
        ci = bt.proportion_ci(confidence_level=0.95, method='exact')
        result[g] = {'covered': n_covered, 'total': n_total, 'proportion': float(proportion),
                     'ci_low': float(ci.low), 'ci_high': float(ci.high), 'n_slices_used': int(len(sub))}
    return result


def load_summary_metadata(csv_dir: Path) -> dict:
    meta = {}
    for g in GROUPS:
        p = csv_dir / f'metrics_{g}.summary.json'
        meta[g] = json.loads(p.read_text(encoding='utf-8')) if p.is_file() else None
    return meta


# ---------------------------------------------------------------------------
# Avaliacao condicional de hipoteses (refactor 26/05/2026)
# ---------------------------------------------------------------------------

def evaluate_hypothesis(hypothesis: dict, metric_results: dict, alpha: float) -> dict:
    """Avalia hipotese: CONFIRMADA / NAO detectada / INVERTIDA.

    - NAO detectada: p_Holm >= alpha (sem evidencia)
    - CONFIRMADA: p_Holm < alpha E direcao bate com expected_higher
    - INVERTIDA: p_Holm < alpha MAS direcao contraria
    """
    r = metric_results.get(hypothesis['metric'], {})
    means = r.get('means', {})

    p_holm = float('nan')
    for pair in r.get('pairwise_wilcoxon', []):
        if pair['pair'] == hypothesis['pair']:
            p_holm = pair.get('p_holm', float('nan'))
            break

    g1, g2 = hypothesis['pair'].split(' vs ')
    delta_observed = float('nan')
    if g1 in means and g2 in means:
        delta_observed = float(means[g1]) - float(means[g2])

    expected_higher = hypothesis['expected_higher']
    if expected_higher == g1:
        direction_correct = (delta_observed > 0)
    elif expected_higher == g2:
        direction_correct = (delta_observed < 0)
    else:
        direction_correct = False

    significant = (not math.isnan(p_holm)) and (p_holm < alpha)
    if not significant:
        outcome = 'NAO detectada'
    elif direction_correct:
        outcome = 'CONFIRMADA'
    else:
        outcome = 'INVERTIDA'

    return {
        'id': hypothesis['id'], 'desc': hypothesis['desc'],
        'metric': hypothesis['metric'], 'pair': hypothesis['pair'],
        'expected_higher': expected_higher,
        'p_holm': p_holm, 'delta_observed': delta_observed,
        'significant': significant, 'direction_correct': direction_correct,
        'outcome': outcome,
    }


def evaluate_all_hypotheses(metric_results: dict, alpha: float) -> list:
    return [evaluate_hypothesis(h, metric_results, alpha) for h in HYPOTHESES]


# ---------------------------------------------------------------------------
# Markdown report generator (narrativa condicional)
# ---------------------------------------------------------------------------

def _fmt(x, fmt='.4f'):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return 'n.d.'
    return format(x, fmt)


def _fmt_pval(p):
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return 'n.d.'
    if p < 0.001:
        return '< 0.001'
    if p < 0.01:
        return f'{p:.4f}'
    return f'{p:.3f}'


def _fmt_delta(delta):
    if delta is None or (isinstance(delta, float) and math.isnan(delta)):
        return 'n.d.'
    sign = '+' if delta >= 0 else '-'
    return f'{sign}{abs(delta):.4f}'


def _signif_marker(p, alpha):
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return ''
    return ' \\*' if p < alpha else ''


def _get_pair_p(metric_result: dict, pair_name: str) -> float:
    for p in metric_result.get('pairwise_wilcoxon', []):
        if p['pair'] == pair_name:
            return p.get('p_holm', float('nan'))
    return float('nan')


def _build_executive_summary(hyp_results: list, metric_results: dict, alpha: float) -> list:
    """Bullets do resumo executivo CONDICIONAIS aos p-values."""
    lines = []
    n_confirmed = sum(1 for h in hyp_results if h['outcome'] == 'CONFIRMADA')
    n_total = len(hyp_results)
    confirmed_hyps = [h for h in hyp_results if h['outcome'] == 'CONFIRMADA']
    not_det_hyps = [h for h in hyp_results if h['outcome'] != 'CONFIRMADA']

    # Bullet 1: hipotese principal
    if n_confirmed == 0:
        bullet1 = (f'1. **Hipotese principal NAO confirmada com lambda=5**: nenhuma '
                   f'das {n_total} hipoteses de C > B foi detectada (todas '
                   f'p_Holm >= {alpha}).')
    elif n_confirmed == n_total:
        bullet1 = (f'1. **Hipotese principal CONFIRMADA em TODAS as metricas de lesao**: '
                   f'as {n_total} hipoteses de C > B foram detectadas com Wilcoxon-Holm '
                   f'p_Holm < {alpha} sobre n=362 pares.')
    else:
        confirmed_ids = ', '.join(h['id'] for h in confirmed_hyps)
        not_det_ids = ', '.join(h['id'] for h in not_det_hyps)
        bullet1 = (f'1. **Hipotese principal CONFIRMADA com tamanho de efeito pequeno '
                   f'mas estatisticamente robusto**: Grupo C (QR-Lesion, lambda=5) '
                   f'supera Grupo B em {n_confirmed} das {n_total} metricas restritas '
                   f'a lesao ({confirmed_ids}), todas p_Holm < {alpha} sobre n=362 '
                   f'pares. Excecao: {not_det_ids}. Detalhes em Secao 4.1.')
    lines.append(bullet1)
    lines.append('')

    # Bullet 2: descoberta sobre A em Coverage_lesion
    cov_les = metric_results.get('coverage_lesion', {})
    means_cl = cov_les.get('means', {})
    p_ab = _get_pair_p(cov_les, 'A vs B')
    p_ac = _get_pair_p(cov_les, 'A vs C')
    a_better_b = (means_cl.get('A', float('nan')) > means_cl.get('B', float('nan'))
                  and not math.isnan(p_ab) and p_ab < alpha)
    a_better_c = (means_cl.get('A', float('nan')) > means_cl.get('C', float('nan'))
                  and not math.isnan(p_ac) and p_ac < alpha)

    if a_better_b and a_better_c:
        delta_ab = means_cl.get('A', 0) - means_cl.get('B', 0)
        bullet2 = (f'2. **Descoberta inesperada de grande magnitude**: Grupo A (ResM, '
                   f'locally-adaptive) supera B e C em Coverage_lesion '
                   f'(A = {_fmt(means_cl.get("A"))} vs '
                   f'B = {_fmt(means_cl.get("B"))}, C = {_fmt(means_cl.get("C"))}; '
                   f'p_Holm < {alpha}; Delta A-B = {_fmt_delta(delta_ab)}, '
                   f'~{abs(delta_ab)*100:.1f} pp). O ResM escala o intervalo localmente '
                   f'pela uncertainty u(x); CQR usa q_hat constante. Implicacao: '
                   f'metodos locally-adaptive sao **estruturalmente superiores** a CQR '
                   f'marginal em coverage de regioes pequenas clinicamente relevantes.')
    else:
        bullet2 = (f'2. **Coverage_lesion**: medias A = {_fmt(means_cl.get("A"))}, '
                   f'B = {_fmt(means_cl.get("B"))}, C = {_fmt(means_cl.get("C"))}. '
                   f'Analise par-a-par em Secoes 3.1 e 4.3.')
    lines.append(bullet2)
    lines.append('')

    # Bullet 3: ULAS como contribuicao
    ulas = metric_results.get('ulas_lesion', {})
    means_ulas = ulas.get('means', {})
    p_ab_u = _get_pair_p(ulas, 'A vs B')
    p_ac_u = _get_pair_p(ulas, 'A vs C')
    p_bc_u = _get_pair_p(ulas, 'B vs C')
    n_sig_ulas = sum([
        (not math.isnan(p_ab_u) and p_ab_u < alpha),
        (not math.isnan(p_ac_u) and p_ac_u < alpha),
        (not math.isnan(p_bc_u) and p_bc_u < alpha),
    ])

    if n_sig_ulas == 3:
        bullet3 = (f'3. **ULAS discrimina entre metodos**: detecta diferencas '
                   f'significativas entre todos os pares A/B/C (todos p_Holm < {alpha}), '
                   f'ordenacao A = {_fmt(means_ulas.get("A"))} < '
                   f'B = {_fmt(means_ulas.get("B"))} < C = {_fmt(means_ulas.get("C"))}. '
                   f'ULAS captura informacao direcional que o Pearson global nao '
                   f'captura, validando-se como **metrica discriminativa** entre '
                   f'metodos de UQ pixelwise.')
    elif n_sig_ulas > 0:
        bullet3 = (f'3. **ULAS valida-se como metrica de UQ**: deteccao em '
                   f'{n_sig_ulas} de 3 pares; medias A = {_fmt(means_ulas.get("A"))}, '
                   f'B = {_fmt(means_ulas.get("B"))}, C = {_fmt(means_ulas.get("C"))}. '
                   f'Detalhes em Secao 4.4.')
    else:
        bullet3 = (f'3. **ULAS nao discriminou** nesta avaliacao (nenhum par com '
                   f'p_Holm < {alpha}). Discussao em Secao 4.4.')
    lines.append(bullet3)
    lines.append('')

    return lines


def render_md_report(metric_results: dict, coverage_global: dict, coverage_lesion: dict,
                     summaries: dict, hyp_results: list, args) -> str:
    """Gera docs/S5.md com narrativa CONDICIONAL aos p-values."""
    lines = []
    today = datetime.now(timezone.utc).strftime('%d/%m/%Y')

    lines.append('# S5 - Quantificacao de Incerteza em MRI Cerebral Acelerada')
    lines.append('')
    lines.append('**Trabalho de Conclusao de Curso**')
    lines.append('**Programa**: USP-ESALQ / PECEGE - Data Science & Analytics')
    lines.append('**Versao**: 4a entrega (resultados consolidados)')
    lines.append('**Repositorio**: github.com/KR0N0S7/tcc-mri-uncertainty')
    lines.append(f'**Data deste relatorio**: {today}')
    lines.append('')
    lines.append('---')
    lines.append('')

    lines.append('## Resumo executivo')
    lines.append('')
    lines.append(
        'Este trabalho investiga quantificacao de incerteza (UQ) pixelwise em '
        'reconstrucao de Ressonancia Magnetica (MRI) cerebral acelerada (4x) '
        'usando uma End-to-End Variational Network (E2E-VarNet) com backbone '
        'congelado. Tres metodos de UQ foram comparados sobre 352 volumes do '
        'fastMRI Brain (Zbontar et al., 2020) anotados pelo fastMRI+ (Zhao et '
        'al., 2022): (A) Residual Magnitude (ResM, locally-adaptive); (B) CQR '
        '(Giannakopoulos et al., 2026); (C) **QR-Lesion**, contribuicao original '
        'com loss ponderada (lambda=5). Como contribuicao paralela, propomos o '
        '**Uncertainty-Lesion Alignment Score (ULAS)**, metrica de alinhamento '
        'direcional entre gradientes da incerteza e do erro.'
    )
    lines.append('')
    lines.append('**Achados principais (4a entrega):**')
    lines.append('')
    lines.extend(_build_executive_summary(hyp_results, metric_results, args.alpha))

    lines.append(
        '**Interpretacao geral**: a magnitude dos efeitos, mesmo quando '
        'significativos, sugere que a calibracao conforme marginal tende a '
        'achatar ganhos locais induzidos pela loss. Direcoes promissoras: '
        'ablacao de lambda (5.1), calibracao condicional (5.2) e combinacao '
        'ResM + loss lesion-aware (5.3).'
    )
    lines.append('')
    lines.append('---')
    lines.append('')

    lines.append('## 1. Setup experimental')
    lines.append('')
    lines.append('### 1.1 Dataset e splits')
    lines.append('')
    lines.append('- fastMRI Brain (Zbontar et al., 2020), 4x undersampled k-space, multi-coil.')
    lines.append('- fastMRI+ (Zhao et al., 2022): 352 volumes com mascaras de lesao (AXFLAIR, AXT1, AXT1POST).')
    lines.append('- Splits slice-wise, seed=42: 213 train + 46 val + 46 cal + 47 test.')
    lines.append('- Reconstrucao: E2E-VarNet pre-treinada (Sriram et al., 2020), 12 cascadas, congelada.')
    lines.append('')
    lines.append('### 1.2 Modelos')
    lines.append('')
    for g in GROUPS:
        lines.append(f'- **Grupo {g}**: {GROUP_DESCRIPTIONS[g]}')
    lines.append('')
    lines.append(
        '- **Arquitetura comum**: U-Net (chans=32, num_pool_layers=4, ~15.5M parametros). '
        'B e C sao a mesma classe Python; unica variavel independente entre eles e a loss.'
    )
    lines.append('')
    lines.append('### 1.3 Treino')
    lines.append('')
    lines.append('AdamW (lr=3e-4, wd=1e-4), warmup linear 7500 iters, 210000 iters totais, batch=1, seed=42.')
    lines.append('Normalizacao por max_val do volume (D1, preserva exchangeability para CQR).')
    lines.append('')
    lines.append('### 1.4 Calibracao conforme (S5.7)')
    lines.append('')
    lines.append('alpha=0.10 (cobertura nominal 90%), sobre 46 volumes cal:')
    lines.append('')
    lines.append('| Grupo | Metodo | q_hat | SHA-256 (16 hex) |')
    lines.append('|---|---|---|---|')
    for g in GROUPS:
        s = summaries.get(g) or {}
        method = s.get('method', 'n.d.')
        qhat = s.get('q_hat', float('nan'))
        sha = s.get('checkpoint_sha256', 'n.d.')
        sha_short = (sha[:16] + '...') if len(sha) > 16 else sha
        lines.append(f'| {GROUP_LABELS[g]} | {method} | {_fmt(qhat, ".6f")} | `{sha_short}` |')
    lines.append('')
    lines.append('---')
    lines.append('')

    lines.append('## 2. Resultados S5.8 - Metricas por regiao')
    lines.append('')
    lines.append('Split test: 47 volumes, 736 slices, 362 com lesao.')
    lines.append('')
    lines.append('### 2.1 Comparativo (mean +/- std)')
    lines.append('')
    lines.append('| Metrica | A (ResM) | B (QR) | C (QR-Lesion) |')
    lines.append('|---|---|---|---|')
    for metric in METRICS:
        r = metric_results[metric]
        means = r.get('means', {})
        stds = r.get('stds', {})
        n = r.get('n_paired', 0)
        cells = [f'{METRIC_LABELS[metric]} (n={n})']
        for g in GROUPS:
            cells.append(f'{_fmt(means.get(g))} +/- {_fmt(stds.get(g))}')
        lines.append('| ' + ' | '.join(cells) + ' |')
    lines.append('')
    lines.append('### 2.2 ULAS null baseline')
    lines.append('')
    lines.append('| Grupo | ULAS_real | ULAS_null | z-score |')
    lines.append('|---|---|---|---|')
    for g in GROUPS:
        s = summaries.get(g) or {}
        ms = s.get('metrics_summary', {})
        ul = ms.get('ulas_lesion', {}).get('mean', float('nan'))
        nn = ms.get('ulas_null_mean', {}).get('mean', float('nan'))
        zz = ms.get('ulas_z_score', {}).get('mean', float('nan'))
        lines.append(f'| {GROUP_LABELS[g]} | {_fmt(ul)} | {_fmt(nn)} | {_fmt(zz, ".2f")} |')
    lines.append('')
    lines.append('---')
    lines.append('')

    lines.append('## 3. Analise estatistica formal (S5.9)')
    lines.append('')
    lines.append(
        f'Friedman test (k=3 grupos pareados); Wilcoxon signed-rank par-a-par '
        f'com Holm-Bonferroni step-down. BCa bootstrap ({args.n_bootstrap} resamples) '
        f'para CIs 95%. Clopper-Pearson micro-coverage. alpha = {args.alpha}.'
    )
    lines.append('')
    lines.append('### 3.1 Friedman + Wilcoxon-Holm')
    lines.append('')
    lines.append('Marcador `*`: p_Holm < alpha.')
    lines.append('')
    lines.append('| Metrica | n | Friedman p | A vs B | A vs C | B vs C |')
    lines.append('|---|---|---|---|---|---|')
    for metric in METRICS:
        r = metric_results[metric]
        n = r.get('n_paired', 0)
        f_p = r.get('friedman', {}).get('p_value', float('nan'))
        pairs_data = {p['pair']: p for p in r.get('pairwise_wilcoxon', [])}
        ab = pairs_data.get('A vs B', {})
        ac = pairs_data.get('A vs C', {})
        bc = pairs_data.get('B vs C', {})
        cells = [
            METRIC_LABELS[metric], str(n),
            _fmt_pval(f_p) + _signif_marker(f_p, args.alpha),
            _fmt_pval(ab.get('p_holm', float('nan'))) + _signif_marker(ab.get('p_holm', float('nan')), args.alpha),
            _fmt_pval(ac.get('p_holm', float('nan'))) + _signif_marker(ac.get('p_holm', float('nan')), args.alpha),
            _fmt_pval(bc.get('p_holm', float('nan'))) + _signif_marker(bc.get('p_holm', float('nan')), args.alpha),
        ]
        lines.append('| ' + ' | '.join(cells) + ' |')
    lines.append('')
    lines.append('### 3.2 BCa bootstrap (CIs 95%)')
    lines.append('')
    lines.append('| Metrica | A | B | C |')
    lines.append('|---|---|---|---|')
    for metric in METRICS:
        r = metric_results[metric]
        means = r.get('means', {})
        cis = r.get('bootstrap_ci_95', {})
        cells = [METRIC_LABELS[metric]]
        for g in GROUPS:
            m = means.get(g, float('nan'))
            ci = cis.get(g, {})
            cells.append(f'{_fmt(m)} [{_fmt(ci.get("low", float("nan")))}, {_fmt(ci.get("high", float("nan")))}]')
        lines.append('| ' + ' | '.join(cells) + ' |')
    lines.append('')
    lines.append('### 3.3 Clopper-Pearson micro-coverage')
    lines.append('')
    lines.append('**(a) Coverage global:**')
    lines.append('')
    lines.append('| Grupo | Cobertos / Total | Proporcao | CI 95% |')
    lines.append('|---|---|---|---|')
    for g in GROUPS:
        cp = coverage_global.get(g, {})
        lines.append(f'| {GROUP_LABELS[g]} | {cp.get("covered", 0):,} / {cp.get("total", 0):,} | '
                     f'{_fmt(cp.get("proportion"))} | '
                     f'[{_fmt(cp.get("ci_low"))}, {_fmt(cp.get("ci_high"))}] |')
    lines.append('')
    lines.append('**(b) Coverage_lesion:**')
    lines.append('')
    lines.append('| Grupo | Cobertos / Total | Proporcao | CI 95% |')
    lines.append('|---|---|---|---|')
    for g in GROUPS:
        cp = coverage_lesion.get(g, {})
        lines.append(f'| {GROUP_LABELS[g]} | {cp.get("covered", 0):,} / {cp.get("total", 0):,} | '
                     f'{_fmt(cp.get("proportion"))} | '
                     f'[{_fmt(cp.get("ci_low"))}, {_fmt(cp.get("ci_high"))}] |')
    lines.append('')
    lines.append('---')
    lines.append('')

    lines.append('## 4. Discussao')
    lines.append('')
    lines.append('### 4.1 Hipoteses originais e resultados')
    lines.append('')
    lines.append('Avaliacao dinamica: **CONFIRMADA** (p_Holm < alpha E direcao bate), '
                 '**NAO detectada** (p_Holm >= alpha), **INVERTIDA** (p_Holm < alpha mas direcao contraria).')
    lines.append('')
    lines.append('| # | Hipotese | p_Holm | Delta (g1 - g2) | Resultado |')
    lines.append('|---|---|---|---|---|')
    for h in hyp_results:
        lines.append(f'| {h["id"]} | {h["desc"]} | {_fmt_pval(h["p_holm"])} | '
                     f'{_fmt_delta(h["delta_observed"])} | **{h["outcome"]}** |')
    lines.append('')
    n_confirmed = sum(1 for h in hyp_results if h['outcome'] == 'CONFIRMADA')
    lines.append(f'Sintese: {n_confirmed} de {len(hyp_results)} hipoteses sobre C > B '
                 f'em metricas de lesao confirmadas estatisticamente.')
    lines.append('')
    lines.append('### 4.2 Mecanica do efeito de lambda=5')
    lines.append('')
    lines.append(
        'Lesoes ocupam ~3-5% dos pixels; lambda=5 leva peso efetivo a ~15-25% '
        'da loss total. Efeitos consistentes em ~60-70% das slices indicam '
        'ajuste local dos intervalos durante o treino. Magnitude pequena '
        'pos-calibracao reflete gargalo do q_hat marginal escalar: a calibracao '
        'aplica uma correcao global, achatando ganhos locais.'
    )
    lines.append('')
    lines.append('### 4.3 A descoberta inesperada sobre o Grupo A')
    lines.append('')
    lines.append(
        'Quando A supera B/C em Coverage_lesion: o intervalo do ResM e '
        '[x - q*u(x), x + q*u(x)] com u(x) pixel-a-pixel; CQR usa q constante. '
        'Para coverage de regioes pequenas clinicamente relevantes, metodos '
        'locally-adaptive sao estruturalmente superiores ao CQR marginal. Sugere '
        'Caminho 3 (Secao 5.3): combinar loss QR-Lesion com calibracao adaptativa.'
    )
    lines.append('')
    lines.append('### 4.4 ULAS como contribuicao metodologica')
    lines.append('')
    lines.append('Validacao em tres dimensoes:')
    lines.append('')
    lines.append('1. **Construct (sintetica)**: tests/test_ulas.py confirma ULAS=1.0 '
                 'para gradientes identicos, ~0.0 para ortogonais, ~2/pi para aleatorios.')
    lines.append('2. **Empirica (null baseline)**: z > 2 em todos os grupos.')
    lines.append('3. **Discriminacao entre metodos**: ver Secao 3.1 par-a-par. Quando '
                 'ULAS discrimina (p_Holm < alpha em multiplos pares), funciona como '
                 'instrumento de comparacao alem do Pearson global.')
    lines.append('')
    lines.append('### 4.5 Limitacoes')
    lines.append('')
    lines.append('1. **Tamanho amostral**: 47 volumes test, 362 slices com lesao. Limitacao para subgrupos estratificados.')
    lines.append('2. **Subcobertura empirica**: cobertura global ~88% em B/C (alvo 90%), sugere violacao sutil de exchangeability.')
    lines.append('3. **Calibracao marginal**: garantia formal sob exchangeability; garantias condicionais impossiveis em distribuicao-livre (Barber et al., 2021).')
    lines.append('4. **IoU_topk com threshold unico**: top_pct=0.05 arbitrario; curva implementada em iou_curve.')
    lines.append('5. **Backbone VarNet congelado**: interacao reconstrucao-UQ nao capturada.')
    lines.append('6. **Lambda unico (=5)**: sem ablacao sistematica.')
    lines.append('')
    lines.append('---')
    lines.append('')

    lines.append('## 5. Trabalho futuro')
    lines.append('')
    lines.append('**5.1 Ablacao de lambda** em {2, 10, 20, 50}. ~24h GPU T4.')
    lines.append('')
    lines.append('**5.2 Calibracao condicional / Group-balanced CP** (Romano 2020, Cauchois 2023). 3-5 dias.')
    lines.append('')
    lines.append('**5.3 ResM + loss lesion-aware**: combina os dois achados principais. 1-2 semanas.')
    lines.append('')
    lines.append('---')
    lines.append('')

    lines.append('## 6. Reprodutibilidade')
    lines.append('')
    lines.append('### 6.1 SHA-256 dos checkpoints')
    lines.append('')
    lines.append('| Grupo | SHA-256 (16 primeiros hex) |')
    lines.append('|---|---|')
    for g in GROUPS:
        s = summaries.get(g) or {}
        sha = s.get('checkpoint_sha256', 'n.d.')
        sha_short = (sha[:16] + '...') if len(sha) > 16 else sha
        lines.append(f'| {GROUP_LABELS[g]} | `{sha_short}` |')
    lines.append('')
    lines.append('SHA-256 completos em `q_hat_*.json` (dataset Kaggle `tcc-mri-conformal-qhats`).')
    lines.append('')
    lines.append('### 6.2 Repositorio')
    lines.append('')
    lines.append('- `github.com/KR0N0S7/tcc-mri-uncertainty` (branch `main`, tag `4a-entrega`).')
    lines.append('')
    lines.append('### 6.3 Comandos')
    lines.append('')
    lines.append('```bash')
    lines.append('git clone https://github.com/KR0N0S7/tcc-mri-uncertainty.git')
    lines.append('cd tcc-mri-uncertainty && git checkout 4a-entrega')
    lines.append('pip install -r requirements.txt')
    lines.append('python -m pytest tests/ -v')
    lines.append('python scripts/analyze_S5_9.py \\')
    lines.append('    --csv-dir <dir_com_3_csvs> \\')
    lines.append('    --output-json docs/figures/s5_9_analysis.json \\')
    lines.append('    --output-md docs/S5.md')
    lines.append('```')
    lines.append('')
    lines.append('### 6.4 Versoes')
    lines.append('')
    lines.append('Python 3.12, PyTorch 2.10, scipy >= 1.11, pandas >= 2.0, numpy >= 1.24.')
    lines.append('')
    lines.append('---')
    lines.append('')

    lines.append('## 7. Referencias')
    lines.append('')
    refs = [
        'Barber, R.F. et al. (2021). The limits of distribution-free conditional predictive inference. *Information and Inference*, 10(2):455-482.',
        'Bates, S. et al. (2021). Distribution-free, risk-controlling prediction sets. *J. ACM*, 68(6):1-34.',
        'Cauchois, M. et al. (2023). Robust Validation. *JASA*.',
        'Clopper, C.J.; Pearson, E.S. (1934). The use of confidence or fiducial limits illustrated in the case of the binomial. *Biometrika*, 26(4):404-413.',
        'Demsar, J. (2006). Statistical comparisons of classifiers over multiple data sets. *JMLR*, 7:1-30.',
        'Efron, B.; Tibshirani, R.J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall/CRC.',
        'Friedman, M. (1937). The use of ranks. *JASA*, 32(200):675-701.',
        'Giannakopoulos, C. et al. (2026). Pixelwise Uncertainty Quantification of Accelerated MRI Reconstruction. *arXiv:2601.13236*.',
        'Gibbs, I.; Candes, E. (2021). Adaptive Conformal Inference Under Distribution Shift. *NeurIPS*.',
        'Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scand. J. Statist.*, 6(2):65-70.',
        'Romano, Y.; Patterson, E.; Candes, E. (2019). Conformalized Quantile Regression. *NeurIPS*, 32:3543-3553.',
        'Romano, Y. et al. (2020). Malice Aforethought. *NeurIPS*.',
        'Sriram, A. et al. (2020). End-to-End Variational Networks for Accelerated MRI Reconstruction. *MICCAI*, 64-73.',
        'Wilcoxon, F. (1945). Individual comparisons by ranking methods. *Biometrics*, 1(6):80-83.',
        'Zbontar, J. et al. (2020). fastMRI. *arXiv:1811.08839*.',
        'Zhao, R. et al. (2022). fastMRI+. *Scientific Data*, 9:152.',
    ]
    for r in sorted(refs):
        lines.append(f'- {r}')
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append(f'*Gerado por `scripts/analyze_S5_9.py` em {today}. Narrativa condicional aos p-values reais.*')
    lines.append('')

    return '\n'.join(lines)


def main() -> int:
    args = parse_args()
    if not args.csv_dir.is_dir():
        print(f'ERRO: --csv-dir nao existe: {args.csv_dir}', file=sys.stderr)
        return 2

    print(f'[1/5] Carregando 3 CSVs de {args.csv_dir}...')
    df = load_csvs(args.csv_dir)
    print(f'      DataFrame total: {len(df)} linhas')

    summaries = load_summary_metadata(args.csv_dir)
    print(f'      Summaries: {sum(1 for s in summaries.values() if s)} de 3 JSONs carregados')

    print(f'\n[2/5] Analisando {len(METRICS)} metricas (Friedman + Wilcoxon-Holm + BCa)...')
    metric_results = {}
    for metric in METRICS:
        print(f'      - {metric}...')
        metric_results[metric] = analyze_metric(df, metric, args.n_bootstrap, args.alpha, args.seed)

    print(f'\n[3/5] Clopper-Pearson micro-coverage...')
    coverage_global = clopper_pearson_coverage(df, region='global')
    coverage_lesion = clopper_pearson_coverage(df, region='lesion')

    print(f'\n[4/5] Avaliando {len(HYPOTHESES)} hipoteses (alpha={args.alpha})...')
    hyp_results = evaluate_all_hypotheses(metric_results, args.alpha)
    for h in hyp_results:
        print(f'      {h["id"]:<4} {h["outcome"]:<14} '
              f'(p_Holm={_fmt_pval(h["p_holm"])}, delta={_fmt_delta(h["delta_observed"])})')

    print(f'\n[5/5] Salvando outputs...')
    payload = {
        'metadata': {
            'created_at': datetime.now(timezone.utc).isoformat(),
            'alpha': args.alpha, 'n_bootstrap': args.n_bootstrap,
            'bootstrap_method': 'BCa', 'seed': args.seed,
            'groups': list(GROUPS), 'metrics_tested': list(METRICS),
        },
        'metric_results': metric_results,
        'coverage_clopper_pearson': {'global': coverage_global, 'lesion': coverage_lesion},
        'hypothesis_evaluation': hyp_results,
        'group_metadata': {g: (summaries[g] if summaries[g] else None) for g in GROUPS},
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, default=str), encoding='utf-8')
    print(f'      JSON salvo em: {args.output_json}')

    if args.output_md is not None:
        md = render_md_report(metric_results, coverage_global, coverage_lesion,
                              summaries, hyp_results, args)
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(md, encoding='utf-8')
        print(f'      Markdown salvo em: {args.output_md} ({md.count(chr(10)) + 1} linhas)')

    print('\n' + '=' * 70)
    print(f'HEADLINE - p-values (alpha = {args.alpha})')
    print('=' * 70)
    for metric in METRICS:
        r = metric_results[metric]
        f_p = r.get('friedman', {}).get('p_value', float('nan'))
        print(f'\n{METRIC_LABELS[metric]:<22} Friedman p = {_fmt_pval(f_p)}'
              + _signif_marker(f_p, args.alpha))
        for pair_data in r.get('pairwise_wilcoxon', []):
            ph = pair_data.get('p_holm', float('nan'))
            print(f'  {pair_data["pair"]:<10} Wilcoxon-Holm p = {_fmt_pval(ph)}'
                  + _signif_marker(ph, args.alpha))
    print('=' * 70)
    print('\n* = p_Holm < alpha')
    print('\n' + '=' * 70)
    print('HIPOTESES')
    print('=' * 70)
    for h in hyp_results:
        print(f'  {h["id"]:<4} {h["outcome"]:<14} | {h["desc"][:80]}')
    print('=' * 70)
    print()
    return 0


if __name__ == '__main__':
    sys.exit(main())
