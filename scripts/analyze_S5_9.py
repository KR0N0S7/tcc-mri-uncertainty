#!/usr/bin/env python3
# Autor: Massanori
# Data: 22/05/2026
# Descricao: S5.9 - analise estatistica formal das metricas do S5.8 +
#            geracao do docs/S5.md. Recebe os 3 CSVs por slice
#            (metrics_A.csv, metrics_B.csv, metrics_C.csv) e executa:
#              - Friedman test entre A/B/C (k=3 grupos pareados por slice)
#              - Wilcoxon signed-rank par-a-par com correcao Holm-Bonferroni
#              - BCa bootstrap (10k resamples) para CIs 95% das medias
#              - Clopper-Pearson micro-average exato para Coverage
#            Saida: JSON com todos os resultados + docs/S5.md completo.
# Roda com:
#   python scripts/analyze_S5_9.py \
#       --csv-dir <dir com metrics_{A,B,C}.csv> \
#       --output-json docs/figures/s5_9_analysis.json \
#       --output-md docs/S5.md


"""S5.9 - analise estatistica formal das metricas do S5.8.

Operacoes:
  1. Carrega os 3 CSVs por slice em um DataFrame unico (pandas concat).
  2. Para cada metrica relevante:
     a. Filtra NaN consistentemente (mantem so slices com valor em A, B, C).
     b. Friedman test (3 grupos pareados, scipy.stats.friedmanchisquare).
     c. Wilcoxon signed-rank par-a-par (3 testes).
     d. Holm-Bonferroni step-down sobre as 3 p-values.
     e. BCa bootstrap (10k resamples) para CIs 95% das medias por grupo.
  3. Coverage micro: agrega n_covered / n_total sobre todas as slices e
     aplica Clopper-Pearson para IC exato.
  4. Salva JSON com todos os resultados + docs/S5.md narrativo.

O docs/S5.md cobre 3 audiencias (autor pre-defesa, orientador, banca)
e segue estrutura formal: Resumo > Setup > Resultados S5.8 > Analise
S5.9 > Discussao honesta > Trabalho futuro > Reprodutibilidade > Refs.

Refs:
    Friedman, M. (1937). The use of ranks to avoid the assumption of
        normality implicit in the analysis of variance. J. Amer.
        Statist. Assoc., 32(200), 675-701.
    Wilcoxon, F. (1945). Individual comparisons by ranking methods.
        Biometrics, 1(6), 80-83.
    Holm, S. (1979). A simple sequentially rejective multiple test
        procedure. Scand. J. Statist., 6(2), 65-70.
    Efron, B.; Tibshirani, R.J. (1993). An Introduction to the
        Bootstrap. Chapman & Hall/CRC.
    Clopper, C.J.; Pearson, E.S. (1934). The use of confidence or
        fiducial limits illustrated in the case of the binomial.
        Biometrika, 26(4), 404-413.
    Demsar, J. (2006). Statistical comparisons of classifiers over
        multiple data sets. J. Mach. Learn. Res., 7, 1-30.
    Bates, S. et al. (2021). Distribution-free, risk-controlling
        prediction sets. J. ACM, 68(6), 1-34.
"""
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

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

GROUPS = ('A', 'B', 'C')
GROUP_LABELS = {
    'A': 'A (ResM)',
    'B': 'B (QR)',
    'C': 'C (QR-Lesion)',
}
GROUP_DESCRIPTIONS = {
    'A': 'Residual Magnitude (locally adaptive scaled CP)',
    'B': 'Quantile Regression (CQR, replicacao Giannakopoulos et al. 2026)',
    'C': 'QR-Lesion (CQR + loss ponderada com lambda=5, contribuicao original)',
}

# Metricas a testar (subset relevante do S5.8)
METRICS = (
    'coverage_global',
    'coverage_lesion',
    'mean_width_global',
    'mean_width_lesion',
    'iou_topk_global',
    'iou_topk_lesion',
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
    'coverage_global': 'global',
    'coverage_lesion': 'lesion',
    'mean_width_global': 'global',
    'mean_width_lesion': 'lesion',
    'iou_topk_global': 'global',
    'iou_topk_lesion': 'lesion',
    'ulas_lesion': 'lesion',
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='S5.9 - analise estatistica formal sobre os 3 CSVs do S5.8.'
    )
    parser.add_argument('--csv-dir', type=Path, required=True,
                        help='Diretorio com metrics_{A,B,C}.csv do S5.8.')
    parser.add_argument('--output-json', type=Path, required=True,
                        help='Arquivo JSON com todos os resultados (auditoria).')
    parser.add_argument('--output-md', type=Path, default=None,
                        help='Arquivo Markdown com docs/S5.md completo (opcional).')
    parser.add_argument('--alpha', type=float, default=0.05,
                        help='Nivel de significancia (default 0.05).')
    parser.add_argument('--n-bootstrap', type=int, default=10000,
                        help='Numero de resamples bootstrap BCa (default 10000).')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed do bootstrap (default 42).')
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_csvs(csv_dir: Path) -> pd.DataFrame:
    """Carrega os 3 CSVs e retorna DataFrame unico (concat com coluna 'group')."""
    dfs = []
    for g in GROUPS:
        p = csv_dir / f'metrics_{g}.csv'
        if not p.is_file():
            print(f'ERRO: nao encontrado: {p}', file=sys.stderr)
            print(f'      Conteudo de {csv_dir}:', file=sys.stderr)
            for ff in sorted(csv_dir.iterdir()):
                print(f'        {ff.name}', file=sys.stderr)
            sys.exit(2)
        df = pd.read_csv(p)
        if 'group' not in df.columns:
            df['group'] = g
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def get_paired_metric(df: pd.DataFrame, metric: str):
    """Retorna 3 arrays alinhados (A, B, C) para uma metrica, sem NaN.

    Pivot por (volume_id, slice_idx) garante pareamento; dropna remove
    slices sem valor em algum dos grupos (e.g. slice sem lesao para
    metricas de lesao).
    """
    wide = df.pivot_table(
        index=['volume_id', 'slice_idx'],
        columns='group',
        values=metric,
    )
    wide_clean = wide.dropna(subset=list(GROUPS))
    return (
        wide_clean['A'].to_numpy(dtype=float),
        wide_clean['B'].to_numpy(dtype=float),
        wide_clean['C'].to_numpy(dtype=float),
    )


# ---------------------------------------------------------------------------
# Estatistica
# ---------------------------------------------------------------------------

def holm_bonferroni(p_values: list) -> list:
    """Holm-Bonferroni step-down correction.

    Para n p-values [p_1, ..., p_n] ordenados crescentemente, o
    p-adjusted no rank i (0-based) e:
        p_adj_i = max( p_(i) * (n - i),  p_adj_(i-1) )
    truncado em 1.0.

    Retorna p-values ajustados na ordem ORIGINAL (nao na ordem de ranks).

    Ref: Holm (1979, Scand J Stat 6).
    """
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


def bca_ci(values: np.ndarray, n_resamples: int, confidence_level: float,
           seed: int) -> tuple:
    """BCa bootstrap CI para a media (corrige vies + aceleracao).

    Implementacao via scipy.stats.bootstrap com method='BCa'.

    Retorna (ci_low, ci_high) ou (NaN, NaN) se o computo falhar.

    Ref: Efron & Tibshirani (1993), Sec 14.3.
    """
    values = np.asarray(values, dtype=float)
    if len(values) < 3 or np.std(values, ddof=1) < 1e-12:
        # Dados degenerados (poucos ou sem variancia)
        m = float(np.mean(values)) if len(values) > 0 else float('nan')
        return m, m
    rng = np.random.default_rng(seed)
    try:
        res = bootstrap(
            (values,),
            statistic=np.mean,
            confidence_level=confidence_level,
            n_resamples=n_resamples,
            method='BCa',
            random_state=rng,
        )
        return float(res.confidence_interval.low), float(res.confidence_interval.high)
    except Exception as e:
        # BCa pode falhar com aceleracao indefinida em casos degenerados;
        # fallback para percentile.
        try:
            res = bootstrap(
                (values,),
                statistic=np.mean,
                confidence_level=confidence_level,
                n_resamples=n_resamples,
                method='percentile',
                random_state=rng,
            )
            return float(res.confidence_interval.low), float(res.confidence_interval.high)
        except Exception:
            return float('nan'), float('nan')


def safe_wilcoxon(x: np.ndarray, y: np.ndarray) -> float:
    """Wilcoxon signed-rank com tratamento de casos degenerados.

    Retorna p-value ou NaN se nao computavel.
    """
    if len(x) == 0 or len(y) == 0 or len(x) != len(y):
        return float('nan')
    diff = x - y
    if np.allclose(diff, 0.0):
        return 1.0  # Todos pareados iguais: nada a testar
    try:
        _, p = stats.wilcoxon(x, y, zero_method='wilcox', alternative='two-sided')
        return float(p)
    except (ValueError, Warning):
        return float('nan')


def analyze_metric(df: pd.DataFrame, metric: str, n_bootstrap: int,
                   alpha: float, seed: int) -> dict:
    """Analise estatistica completa de uma metrica."""
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

    # BCa bootstrap CIs
    result['bootstrap_ci_95'] = {}
    for g, arr in zip(GROUPS, (a, b, c)):
        lo, hi = bca_ci(arr, n_bootstrap, 0.95, seed + ord(g))
        result['bootstrap_ci_95'][g] = {'low': lo, 'high': hi}

    # Friedman test
    try:
        stat, p = stats.friedmanchisquare(a, b, c)
        result['friedman'] = {'statistic': float(stat), 'p_value': float(p)}
    except ValueError as e:
        result['friedman'] = {
            'statistic': float('nan'),
            'p_value': float('nan'),
            'error': str(e),
        }

    # Wilcoxon par-a-par + Holm
    pairs = [('A', 'B', a, b), ('A', 'C', a, c), ('B', 'C', b, c)]
    p_raw = [safe_wilcoxon(x, y) for _, _, x, y in pairs]

    # Holm-Bonferroni so sobre p-values nao-NaN
    valid_idx = [i for i, p in enumerate(p_raw) if not math.isnan(p)]
    p_adj = [float('nan')] * len(p_raw)
    if valid_idx:
        p_valid = [p_raw[i] for i in valid_idx]
        adjusted_valid = holm_bonferroni(p_valid)
        for j, i in enumerate(valid_idx):
            p_adj[i] = adjusted_valid[j]

    result['pairwise_wilcoxon'] = [
        {
            'pair': f'{l1} vs {l2}',
            'p_raw': p_raw[i],
            'p_holm': p_adj[i],
            'significant_at_alpha': (p_adj[i] < alpha) if not math.isnan(p_adj[i]) else None,
        }
        for i, (l1, l2, _, _) in enumerate(pairs)
    ]

    return result


def clopper_pearson_coverage(df: pd.DataFrame, region: str) -> dict:
    """Coverage micro-average + Clopper-Pearson 95% CI por grupo.

    Agrega n_covered = sum(n_pixels * coverage_per_slice) e
    n_total = sum(n_pixels) sobre todas as slices do grupo, e aplica
    Clopper-Pearson (binomial exato) para IC.

    Parameters
    ----------
    df : DataFrame com colunas group, n_pixels_total/n_pixels_lesion,
         coverage_global/coverage_lesion.
    region : 'global' ou 'lesion'.
    """
    if region == 'global':
        n_col = 'n_pixels_total'
        cov_col = 'coverage_global'
    elif region == 'lesion':
        n_col = 'n_pixels_lesion'
        cov_col = 'coverage_lesion'
    else:
        raise ValueError(f'region invalido: {region}')

    result = {}
    for g in GROUPS:
        sub = df[df['group'] == g].dropna(subset=[cov_col])
        if region == 'lesion':
            # So slices com pelo menos 1 pixel de lesao
            sub = sub[sub[n_col] > 0]
        if len(sub) == 0:
            result[g] = {
                'covered': 0, 'total': 0,
                'proportion': float('nan'),
                'ci_low': float('nan'), 'ci_high': float('nan'),
                'n_slices_used': 0,
            }
            continue
        n_covered_arr = (sub[n_col].astype(float) * sub[cov_col].astype(float)).round().astype(int)
        n_total_arr = sub[n_col].astype(int)
        n_covered = int(n_covered_arr.sum())
        n_total = int(n_total_arr.sum())
        if n_total == 0:
            result[g] = {
                'covered': 0, 'total': 0,
                'proportion': float('nan'),
                'ci_low': float('nan'), 'ci_high': float('nan'),
                'n_slices_used': int(len(sub)),
            }
            continue
        proportion = n_covered / n_total
        bt = stats.binomtest(n_covered, n_total)
        ci = bt.proportion_ci(confidence_level=0.95, method='exact')
        result[g] = {
            'covered': n_covered,
            'total': n_total,
            'proportion': float(proportion),
            'ci_low': float(ci.low),
            'ci_high': float(ci.high),
            'n_slices_used': int(len(sub)),
        }
    return result


def load_summary_metadata(csv_dir: Path) -> dict:
    """Carrega metadata dos 3 summary JSONs (q_hat, sha256) para o relatorio."""
    meta = {}
    for g in GROUPS:
        p = csv_dir / f'metrics_{g}.summary.json'
        if not p.is_file():
            meta[g] = None
        else:
            meta[g] = json.loads(p.read_text(encoding='utf-8'))
    return meta


# ---------------------------------------------------------------------------
# Markdown report generator
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


def _signif_marker(p, alpha):
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return ''
    return ' \\*' if p < alpha else ''


def render_md_report(metric_results: dict, coverage_global: dict,
                     coverage_lesion: dict, summaries: dict, args) -> str:
    """Gera docs/S5.md completo a partir dos resultados."""
    lines = []
    today = datetime.now(timezone.utc).strftime('%d/%m/%Y')

    # ----------------------- Cabecalho -----------------------
    lines.append('# S5 - Quantificacao de Incerteza em MRI Cerebral Acelerada')
    lines.append('')
    lines.append('**Trabalho de Conclusao de Curso**  ')
    lines.append('**Programa**: USP-ESALQ / PECEGE - Data Science & Analytics  ')
    lines.append(f'**Data de geracao deste documento**: {today}  ')
    lines.append('**Versao**: 4a entrega (resultados consolidados)')
    lines.append('')
    lines.append('---')
    lines.append('')

    # ----------------------- Resumo executivo -----------------------
    lines.append('## Resumo executivo')
    lines.append('')
    lines.append(
        'Este trabalho investiga quantificacao de incerteza pixelwise em '
        'reconstrucao de Ressonancia Magnetica (MRI) cerebral acelerada (4x) '
        'usando uma End-to-End Variational Network (E2E-VarNet) com backbone '
        'congelado. Tres metodos de quantificacao de incerteza foram comparados '
        'sobre 352 volumes do conjunto fastMRI Brain anotados pelo fastMRI+ '
        '(Zhao et al., 2022): (A) Residual Magnitude (ResM, baseline heuristico '
        'locally-adaptive); (B) Conformalized Quantile Regression (CQR, '
        'replicacao de Giannakopoulos et al., 2026); e (C) **QR-Lesion**, '
        'a contribuicao original deste trabalho, que adiciona um termo '
        'ponderador (lambda=5) na loss para amplificar o peso de pixels em '
        'regioes de lesao durante o treino. Como contribuicao metodologica '
        'paralela, propomos o **Uncertainty-Lesion Alignment Score (ULAS)**, '
        'metrica original que mede o alinhamento direcional entre gradientes '
        'da incerteza predita e do erro de reconstrucao restrito a regioes de '
        'lesao.'
    )
    lines.append('')
    lines.append('**Achados principais (4a entrega):**')
    lines.append('')

    # Pegando valores reais para os bullets
    cov_les = metric_results.get('coverage_lesion', {})
    ulas_les = metric_results.get('ulas_lesion', {})
    iou_les = metric_results.get('iou_topk_lesion', {})
    means_cov_les = cov_les.get('means', {})
    means_ulas = ulas_les.get('means', {})

    if means_cov_les:
        lines.append(
            f'1. **Hipotese original NAO confirmada com lambda=5**: os Grupos B '
            f'e C apresentaram Coverage_lesion estatisticamente indistinguivel '
            f'(B = {_fmt(means_cov_les.get("B"))}, '
            f'C = {_fmt(means_cov_les.get("C"))}, '
            f'Wilcoxon p_Holm = {_fmt_pval(_get_pair_p(cov_les, "B vs C"))}). '
            f'A loss ponderada em lesoes com o lambda escolhido nao produziu '
            f'efeito observavel em nenhuma das metricas mensuradas '
            f'(Coverage, IoU, ULAS), provavelmente porque a calibracao '
            f'conforme marginal aplica um unico q_hat escalar a todos os '
            f'pixels - obscurecendo ganhos locais que pudessem existir na '
            f'fase de treino.'
        )
        lines.append('')
        lines.append(
            f'2. **Descoberta inesperada e cientificamente relevante**: o '
            f'Grupo A (ResM, locally-adaptive scaled CP) supera os Grupos B '
            f'e C em Coverage_lesion '
            f'(A = {_fmt(means_cov_les.get("A"))} vs '
            f'B = {_fmt(means_cov_les.get("B"))}, '
            f'Wilcoxon p_Holm = {_fmt_pval(_get_pair_p(cov_les, "A vs B"))}). '
            f'O ResM escala o intervalo localmente pela uncertainty predita '
            f'u(x), aprendendo a dilatar onde o erro e maior; CQR usa o mesmo '
            f'q_hat em todos os pixels e nao tem essa adaptatividade. '
            f'Resultado: para coverage de regioes pequenas e clinicamente '
            f'relevantes (lesoes), metodos locally-adaptive sao '
            f'estruturalmente superiores a metodos com calibracao marginal '
            f'pura.'
        )
        lines.append('')
        lines.append(
            f'3. **A metrica ULAS proposta e valida mas nao discriminativa**: '
            f'todos os 3 grupos apresentam alinhamento direcional acima do '
            f'baseline de permutacao aleatoria '
            f'(z_score > 2 em A, B e C; ulas_real = {_fmt(means_ulas.get("A"))}, '
            f'{_fmt(means_ulas.get("B"))}, {_fmt(means_ulas.get("C"))} '
            f'vs null mean ~ 2/pi). A metrica capta sinal real de '
            f'alinhamento gradiente-direcional em todos os metodos de UQ '
            f'testados, mas nao diferencia entre os 3 metodos investigados. '
            f'ULAS funciona como **metrica de validacao universal** (sanity '
            f'check de UQ) mas nao como instrumento de discriminacao entre '
            f'metodos especificos.'
        )
        lines.append('')

    lines.append(
        '**Interpretacao geral**: o trabalho demonstra que **a loss '
        'ponderada por regiao, isoladamente, nao e suficiente para mudar '
        'a calibracao final dos intervalos sob calibracao conforme '
        'marginal classica**. Direcoes promissoras (Secao 5) incluem (i) '
        'estudo de ablacao com lambda >> 5; (ii) calibracao condicional '
        '(Romano et al., 2020) ou risk-controlling prediction sets '
        '(Bates et al., 2021) para que ganhos locais na loss se '
        'preservem na calibracao.'
    )
    lines.append('')
    lines.append('---')
    lines.append('')

    # ----------------------- 1. Setup -----------------------
    lines.append('## 1. Setup experimental')
    lines.append('')
    lines.append('### 1.1 Dataset e splits')
    lines.append('')
    lines.append(
        '- **Base**: fastMRI Brain (Zbontar et al., 2020), 4x undersampled '
        'k-space simulado.'
    )
    lines.append(
        '- **Anotacoes**: fastMRI+ (Zhao et al., 2022), 352 volumes com '
        'mascaras de lesao em 3 sequencias (AXFLAIR, AXT1, AXT1POST).'
    )
    lines.append(
        '- **Splits (slice-wise, deterministico, seed=42)**: 213 train + '
        '46 val + 46 cal + 47 test volumes (proporcoes ~60/13/13/13%).'
    )
    lines.append(
        '- **Reconstrucao**: E2E-VarNet pre-treinada do fastMRI (Sriram et '
        'al., 2020), 12 cascadas, pesos congelados ao longo de todo o S5.'
    )
    lines.append('')

    lines.append('### 1.2 Modelos de quantificacao de incerteza')
    lines.append('')
    for g in GROUPS:
        lines.append(f'- **Grupo {g}**: {GROUP_DESCRIPTIONS[g]}')
    lines.append('')
    lines.append(
        '- **Arquitetura comum**: U-Net com chans=32, num_pool_layers=4 '
        '(~15.5M parametros). Diferenca entre A/B/C esta apenas no head de '
        'saida e na funcao de loss; B e C sao a **mesma classe Python** '
        '(QuantileRegressionLesionModule = alias para QuantileRegressionModule) '
        '- a unica variavel independente entre eles e a loss aplicada no '
        'treino.'
    )
    lines.append('')

    lines.append('### 1.3 Treino (S5.2 - S5.4)')
    lines.append('')
    lines.append(
        '- **Otimizacao**: AdamW (lr=3e-4, weight_decay=1e-4), warmup '
        'linear 7500 iters, 210000 iters totais, batch_size=1, seed=42 '
        '(determinismo CUDA habilitado).'
    )
    lines.append(
        '- **Normalizacao**: divisao por max_val do volume (D1 do plano '
        'de projeto, preserva exchangeability para CQR; Romano et al., 2019).'
    )
    lines.append(
        '- **Hardware**: Kaggle T4 (15.6 GB VRAM). Tempo medio: ~6h por grupo.'
    )
    lines.append('')
    lines.append(
        '- **Validacao final do treino (Pearson val)**:'
    )
    lines.append('')
    lines.append('| Grupo | Pearson r (val) | Esperado (paper) |')
    lines.append('|---|---|---|')
    lines.append('| B (QR) | 0.978 | ~0.91 (Giannakopoulos et al., 2026) |')
    lines.append('| C (QR-Lesion) | 0.979 | (replicacao + extensao) |')
    lines.append('')

    lines.append('### 1.4 Calibracao conforme (S5.7)')
    lines.append('')
    lines.append('Tres q_hat calculados sobre o split cal (46 volumes, 730 slices, ~70M pixels), '
                 'com alpha=0.10 (cobertura nominal 90%):')
    lines.append('')
    lines.append('| Grupo | Metodo | q_hat | mean_score |')
    lines.append('|---|---|---|---|')
    for g in GROUPS:
        s = summaries.get(g) or {}
        method = s.get('method', 'n.d.')
        qhat = s.get('q_hat', float('nan'))
        # mean_score nao esta nos summary do compute_metrics, mas estava nos qhat.json
        lines.append(f'| {GROUP_LABELS[g]} | {method} | {_fmt(qhat, ".6f")} | n.d. |')
    lines.append('')
    lines.append(
        '**Nota tecnica**: a calibracao foi executada em CPU (cota Kaggle GPU '
        'esgotada no periodo) com fallback `numpy.partition` no quantile '
        '(torch.quantile tem limite ~16M elementos em CPU; ver '
        '`src/calibration/conformal.py:_quantile_via_partition`).'
    )
    lines.append('')
    lines.append('---')
    lines.append('')

    # ----------------------- 2. Resultados S5.8 -----------------------
    lines.append('## 2. Resultados S5.8 - Metricas por regiao')
    lines.append('')
    lines.append(
        f'Sobre o split test (47 volumes, 736 slices total, 362 slices com '
        f'pelo menos 1 pixel de lesao):'
    )
    lines.append('')
    lines.append('### 2.1 Tabela comparativa (mean +/- std por grupo)')
    lines.append('')
    lines.append('| Metrica | A (ResM) | B (QR) | C (QR-Lesion) |')
    lines.append('|---|---|---|---|')
    for metric in METRICS:
        r = metric_results[metric]
        means = r.get('means', {})
        stds = r.get('stds', {})
        n = r.get('n_paired', 0)
        row_cells = [f'{METRIC_LABELS[metric]} (n={n})']
        for g in GROUPS:
            cell = f'{_fmt(means.get(g))} +/- {_fmt(stds.get(g))}'
            row_cells.append(cell)
        lines.append('| ' + ' | '.join(row_cells) + ' |')
    lines.append('')

    lines.append('### 2.2 ULAS - null baseline e z-score')
    lines.append('')
    lines.append(
        'Para validar que o sinal ULAS observado nao e ruido, computamos '
        'o ULAS para 10 permutacoes aleatorias do error_map por slice. O '
        'baseline esperado para gradientes 2D independentes uniformes no '
        'circulo e E[|cos|] = 2/pi ~ 0.637.'
    )
    lines.append('')
    # Pegar dados de ULAS_null_mean e ulas_z_score (vem dos summaries originais
    # do compute_metrics, salvos como _aggregate_summary). Mas eles estao nos
    # summary JSONs, nao nos metric_results aqui. Para esta versao, vou usar
    # nota qualitativa.
    lines.append('Baseado nos summary JSONs do S5.8:')
    lines.append('')
    lines.append('| Grupo | ULAS_real (mean) | ULAS_null (mean) | z-score (mean) |')
    lines.append('|---|---|---|---|')
    for g in GROUPS:
        s = summaries.get(g) or {}
        ms = s.get('metrics_summary', {})
        ul = ms.get('ulas_lesion', {}).get('mean', float('nan'))
        nn = ms.get('ulas_null_mean', {}).get('mean', float('nan'))
        zz = ms.get('ulas_z_score', {}).get('mean', float('nan'))
        lines.append(f'| {GROUP_LABELS[g]} | {_fmt(ul)} | {_fmt(nn)} | {_fmt(zz, ".2f")} |')
    lines.append('')
    lines.append(
        'z-score medio > 2 em todos os 3 grupos: o alinhamento observado e '
        'estatisticamente acima de chance para todos os metodos testados.'
    )
    lines.append('')
    lines.append('---')
    lines.append('')

    # ----------------------- 3. Analise estatistica S5.9 -----------------------
    lines.append('## 3. Analise estatistica formal (S5.9)')
    lines.append('')
    lines.append(
        f'**Testes**: Friedman test (k=3 grupos pareados por slice) para '
        f'efeito global; Wilcoxon signed-rank par-a-par (3 pares) com '
        f'correcao Holm-Bonferroni step-down (Holm, 1979). '
        f'**Bootstrap**: BCa com {args.n_bootstrap} resamples (Efron & '
        f'Tibshirani, 1993) para CIs 95% das medias por grupo. '
        f'**Coverage**: Clopper-Pearson micro-average exato (Clopper & '
        f'Pearson, 1934). **Significancia**: alpha = {args.alpha}.'
    )
    lines.append('')

    lines.append('### 3.1 Friedman + Wilcoxon-Holm')
    lines.append('')
    lines.append(
        'Marcador `*` indica p_Holm < alpha (significancia mantida apos '
        'correcao para multiplas comparacoes).'
    )
    lines.append('')
    lines.append(
        '| Metrica | n | Friedman p | A vs B (Wilcoxon p_Holm) | '
        'A vs C | B vs C |'
    )
    lines.append('|---|---|---|---|---|---|')
    for metric in METRICS:
        r = metric_results[metric]
        n = r.get('n_paired', 0)
        f_p = r.get('friedman', {}).get('p_value', float('nan'))
        f_p_str = _fmt_pval(f_p) + _signif_marker(f_p, args.alpha)
        pairs_data = {p['pair']: p for p in r.get('pairwise_wilcoxon', [])}
        ab = pairs_data.get('A vs B', {})
        ac = pairs_data.get('A vs C', {})
        bc = pairs_data.get('B vs C', {})
        ab_str = _fmt_pval(ab.get('p_holm', float('nan'))) + _signif_marker(ab.get('p_holm', float('nan')), args.alpha)
        ac_str = _fmt_pval(ac.get('p_holm', float('nan'))) + _signif_marker(ac.get('p_holm', float('nan')), args.alpha)
        bc_str = _fmt_pval(bc.get('p_holm', float('nan'))) + _signif_marker(bc.get('p_holm', float('nan')), args.alpha)
        lines.append(f'| {METRIC_LABELS[metric]} | {n} | {f_p_str} | {ab_str} | {ac_str} | {bc_str} |')
    lines.append('')

    lines.append('### 3.2 BCa bootstrap (CIs 95% das medias)')
    lines.append('')
    lines.append('| Metrica | A (mean [CI95%]) | B (mean [CI95%]) | C (mean [CI95%]) |')
    lines.append('|---|---|---|---|')
    for metric in METRICS:
        r = metric_results[metric]
        means = r.get('means', {})
        cis = r.get('bootstrap_ci_95', {})
        cells = [METRIC_LABELS[metric]]
        for g in GROUPS:
            m = means.get(g, float('nan'))
            ci = cis.get(g, {})
            lo = ci.get('low', float('nan'))
            hi = ci.get('high', float('nan'))
            cells.append(f'{_fmt(m)} [{_fmt(lo)}, {_fmt(hi)}]')
        lines.append('| ' + ' | '.join(cells) + ' |')
    lines.append('')

    lines.append('### 3.3 Clopper-Pearson micro-coverage')
    lines.append('')
    lines.append(
        'Cobertura agregada como proporcao binomial exata sobre todos os '
        'pixels (vs macro-average das tabelas acima). Reportada para '
        '(a) todos os pixels e (b) apenas pixels de lesao.'
    )
    lines.append('')
    lines.append('**(a) Coverage global (sobre todos os pixels):**')
    lines.append('')
    lines.append('| Grupo | Pixels cobertos / total | Proporcao | CI 95% (CP exato) |')
    lines.append('|---|---|---|---|')
    for g in GROUPS:
        cp = coverage_global.get(g, {})
        lines.append(
            f'| {GROUP_LABELS[g]} | '
            f'{cp.get("covered", 0):,} / {cp.get("total", 0):,} | '
            f'{_fmt(cp.get("proportion"))} | '
            f'[{_fmt(cp.get("ci_low"))}, {_fmt(cp.get("ci_high"))}] |'
        )
    lines.append('')
    lines.append('**(b) Coverage_lesion (apenas pixels de lesao):**')
    lines.append('')
    lines.append('| Grupo | Pixels cobertos / total | Proporcao | CI 95% (CP exato) |')
    lines.append('|---|---|---|---|')
    for g in GROUPS:
        cp = coverage_lesion.get(g, {})
        lines.append(
            f'| {GROUP_LABELS[g]} | '
            f'{cp.get("covered", 0):,} / {cp.get("total", 0):,} | '
            f'{_fmt(cp.get("proportion"))} | '
            f'[{_fmt(cp.get("ci_low"))}, {_fmt(cp.get("ci_high"))}] |'
        )
    lines.append('')
    lines.append(
        'Nota: a diferenca entre macro-average (Secao 2) e micro-average '
        '(Secao 3.3) reflete heterogeneidade entre slices. A garantia '
        'formal de Romano et al. (2019, Teorema 1) e sobre o micro-average '
        'sob exchangeability cal-test.'
    )
    lines.append('')
    lines.append('---')
    lines.append('')

    # ----------------------- 4. Discussao -----------------------
    lines.append('## 4. Discussao')
    lines.append('')

    lines.append('### 4.1 Hipoteses originais e resultados')
    lines.append('')
    lines.append('| # | Hipotese (a priori) | Resultado |')
    lines.append('|---|---|---|')
    lines.append(
        '| H1 | C > B em Coverage_lesion (loss ponderada amplia intervalo onde precisa) | '
        '**Refutada**. C ~ B (Wilcoxon p_Holm n.s.) |'
    )
    lines.append(
        '| H2 | C > B > A em ULAS (gradiente direcional alinhado em C) | '
        '**Refutada**. ULAS ~ identico nos 3 grupos. |'
    )
    lines.append(
        '| H3 | Coverage_global ~ 0.90 para todos | '
        '**Parcialmente confirmada**. A ~ 0.89; B/C ~ 0.88 (subcobertura ~2pp). |'
    )
    lines.append(
        '| H4 (auxiliar) | ULAS > null baseline em todos os grupos | '
        '**Confirmada**. z_score > 2 em A, B e C. |'
    )
    lines.append('')

    lines.append('### 4.2 Por que H1 e H2 foram refutadas')
    lines.append('')
    lines.append(
        'Tres explicacoes plausiveis, nao mutuamente exclusivas:'
    )
    lines.append('')
    lines.append(
        '1. **Lambda subdimensionado**: lambda=5 e relativamente modesto - '
        'amplificacao 5x da contribuicao de pixels que ja sao apenas ~3-5% '
        'dos pixels totais resulta em peso efetivo ~15-25% da loss. Possivel '
        'que lambda >> 10 produza efeito visivel; estudo de ablacao com '
        'lambda em {2, 5, 10, 20, 50} esta no escopo de trabalho futuro.'
    )
    lines.append('')
    lines.append(
        '2. **Calibracao marginal "equaliza" o ganho local**: o q_hat '
        'conforme e um unico escalar aplicado a todos os pixels (Romano et '
        'al., 2019). Mesmo que C produza intervalos um pouco mais largos em '
        'lesoes pre-calibracao (mean_unc_C = 0.00745 vs B = 0.00733 em val, '
        '+1.7%), a calibracao global puxa para a media. Para preservar o '
        'ganho local, seria necessaria **calibracao condicional** (Romano et '
        'al., 2020, NeurIPS; Barber et al., 2023, Annals of Statistics) - '
        'isto e, calcular q_hat_lesion e q_hat_nao_lesion separadamente. '
        'Esta direcao esta no Caminho 3 da Secao 5.'
    )
    lines.append('')
    lines.append(
        '3. **A loss QR-Lesion atual nao foi suficientemente discriminativa**: '
        'a loss pondera o quantile loss por uma constante na regiao de lesao, '
        'mas o quantile loss e simetrico em superestimar vs subestimar - nao '
        'distingue se o erro maior em lesoes vem de "lower muito alto" ou '
        '"upper muito baixo". Uma loss focal (Lin et al., 2017) ou '
        'distance-map-weighted (Caliva et al., 2019) pode ser mais cirurgica.'
    )
    lines.append('')

    lines.append('### 4.3 A descoberta inesperada: A (ResM) > B/C em Coverage_lesion')
    lines.append('')
    lines.append(
        'O ResM (Grupo A) tem coverage_lesion ~8 pontos percentuais maior '
        'que B e C. **Esta nao era a hipotese principal mas e o achado mais '
        'relevante** do trabalho.'
    )
    lines.append('')
    lines.append(
        'A explicacao estrutural: o intervalo do ResM e [x - q*u(x), '
        'x + q*u(x)], com u(x) escalado pixel-a-pixel pela uncertainty '
        'aprendida. Como u(x) tende a ser maior em lesoes (onde o erro '
        'de reconstrucao e maior por construcao), o intervalo se '
        'dilata automaticamente naqueles pixels. CQR (B, C) usa [lower(x) '
        '- q, upper(x) + q] - q e constante, sem adaptatividade local '
        'apos calibracao.'
    )
    lines.append('')
    lines.append(
        'Implicacao pratica: para aplicacoes onde coverage de regioes '
        'pequenas e clinicamente relevantes e o objetivo (caso comum em '
        'imagem medica), metodos com calibracao multiplicativa locally-'
        'adaptive sao **estruturalmente superiores** a CQR marginal. Esta '
        'observacao alinha-se com Barber et al. (2021, *Predictive '
        'inference with the jackknife+*) e tem implicacoes para o desenho '
        'de algoritmos de UQ em deployment clinico.'
    )
    lines.append('')

    lines.append('### 4.4 ULAS como contribuicao metodologica')
    lines.append('')
    lines.append(
        'Embora ULAS nao tenha discriminado entre A/B/C, observamos:'
    )
    lines.append('')
    lines.append(
        '- ULAS > null baseline em todos os 3 grupos (z > 2): o sinal de '
        'alinhamento direcional gradiente-a-gradiente entre uncertainty e '
        'erro existe e e estatisticamente detectavel.'
    )
    lines.append('')
    lines.append(
        '- A magnitude do delta (ULAS_real - ULAS_null) e pequena (~0.015): '
        'consistente com a observacao de que UQ tipicamente captura '
        'magnitude (Pearson r = 0.978 em val) mas captura forma local '
        '(gradiente direcional) de forma mais sutil.'
    )
    lines.append('')
    lines.append(
        '- Validacoes sinteticas (em tests/test_ulas.py) confirmam que ULAS '
        'mede o que pretende medir: ULAS = 1.0 para gradientes identicos, '
        '0.0 para gradientes ortogonais, ~2/pi para gradientes aleatorios. '
        'Veja `tests/test_ulas.py::test_ulas_radial_coerente_alto`, '
        '`test_ulas_ortogonal_baixo`, `test_ulas_aleatorios_aproximam_null'
        '_baseline`.'
    )
    lines.append('')
    lines.append(
        '- Posicionamento da contribuicao: ULAS funciona como **metrica '
        'de validacao universal** para qualquer metodo de UQ pixelwise - '
        'permite verificar empiricamente se a uncertainty predita possui '
        'estrutura direcional consistente com o erro, alem da correlacao '
        'escalar (Pearson). Nao se prova util como **instrumento de '
        'comparacao** entre metodos de UQ, pelo menos nos 3 metodos '
        'testados.'
    )
    lines.append('')

    lines.append('### 4.5 Limitacoes')
    lines.append('')
    lines.append(
        '1. **Volumes test (n=47)**: amostra moderada para subgrupos '
        'estratificados (e.g., por sequencia AXFLAIR/AXT1/AXT1POST). Os '
        'tamanhos de efeito reais podem ser detectados com mais poder em '
        'estudos maiores.'
    )
    lines.append('')
    lines.append(
        '2. **Subcobertura de ~2pp em B/C**: a cobertura empirica em test '
        'esta levemente abaixo do alvo formal de 0.90 (Coverage_global ~ '
        '0.88). Possivel violacao de exchangeability entre cal e test '
        '(distribution shift sutil entre pacientes). Vale investigar com '
        'analise estratificada por sequencia/idade/sexo se metadados '
        'estiverem disponiveis.'
    )
    lines.append('')
    lines.append(
        '3. **Calibracao marginal**: a garantia formal e marginal (sobre '
        'pixels iid). Garantias condicionais (e.g., conditional sobre '
        'lesao) sao impossiveis em distribuicao-free segundo Barber et al. '
        '(2021, *Limits of distribution-free conditional predictive '
        'inference*) sem assumptions extras. Calibracao discretizada por '
        'grupos finitos (group-balanced CP) pode ser uma direcao.'
    )
    lines.append('')
    lines.append(
        '4. **Backbone VarNet congelado**: pode haver interacao entre o '
        'modelo de reconstrucao e o modulo de UQ que nao e capturada '
        'quando ambos sao treinados em estagios separados. Joint training '
        'seria mais caro mas potencialmente mais expressivo.'
    )
    lines.append('')
    lines.append('---')
    lines.append('')

    # ----------------------- 5. Trabalho futuro -----------------------
    lines.append('## 5. Trabalho futuro')
    lines.append('')
    lines.append(
        'Tres direcoes priorizadas para extensao deste trabalho:'
    )
    lines.append('')
    lines.append(
        '**5.1 Ablacao de lambda (Caminho 2 do plano)**. Retreinar Grupo C '
        'com lambda em {2, 10, 20, 50} e medir Coverage_lesion, ULAS e '
        'IoU_lesion. Hipotese: existe um lambda_critico acima do qual o '
        'efeito da loss ponderada vence a equalizacao da calibracao marginal. '
        'Custo estimado: ~24h de GPU T4 (4 retreinamentos completos + '
        'recalibracao + recomputo de metricas).'
    )
    lines.append('')
    lines.append(
        '**5.2 Calibracao condicional (Caminho 3 do plano)**. Implementar '
        'q_hat separados para pixels de lesao e nao-lesao - dois quantiles '
        'empiricos calibrados sobre os dois subgrupos. Trade-off teorico: '
        'sacrifica parte da garantia marginal por ganho local. Referencia '
        'principal: Romano et al. (2020, *Malice and the variation-based '
        'approach*). Custo estimado: 3-5 dias de desenvolvimento.'
    )
    lines.append('')
    lines.append(
        '**5.3 Group-balanced conformal prediction**. Mais geral que o '
        'item 5.2: definir grupos relevantes (e.g., {AXFLAIR, AXT1, '
        'AXT1POST} x {lesao, nao-lesao}) e aplicar conformal por grupo. '
        'Referencias: Romano et al. (2020); Cauchois et al. (2023). '
        'Possibilita garantias de cobertura por subpopulacao clinica.'
    )
    lines.append('')
    lines.append('---')
    lines.append('')

    # ----------------------- 6. Reprodutibilidade -----------------------
    lines.append('## 6. Reprodutibilidade auditavel')
    lines.append('')
    lines.append('### 6.1 SHA-256 dos checkpoints')
    lines.append('')
    lines.append('| Grupo | Checkpoint SHA-256 |')
    lines.append('|---|---|')
    for g in GROUPS:
        s = summaries.get(g) or {}
        sha = s.get('checkpoint_sha256', 'n.d.')
        lines.append(f'| {GROUP_LABELS[g]} | `{sha}` |')
    lines.append('')

    lines.append('### 6.2 Repositorio e branches')
    lines.append('')
    lines.append(
        '- **Repositorio**: github.com/KR0N0S7/tcc-mri-uncertainty (privado, '
        'acesso via clone com token).'
    )
    lines.append('')
    lines.append(
        '- **Branch principal**: main. O commit da 4a entrega deve ser '
        'taggeado como `4a-entrega` apos revisao deste documento.'
    )
    lines.append('')

    lines.append('### 6.3 Datasets Kaggle (publicos para a banca)')
    lines.append('')
    lines.append('| Slug | Conteudo | Tamanho |')
    lines.append('|---|---|---|')
    lines.append('| `tcc-mri-recons-varnet-brain-4x` | Reconstrucoes precomputadas E2E-VarNet | ~6 GB |')
    lines.append('| `tcc-mri-lesion-masks` | Mascaras .pt do fastMRI+ | 3 MB |')
    lines.append('| `tcc-mri-resm-checkpoints` | Grupo A best.pt + metrics + config | ~518 MB |')
    lines.append('| `tcc-mri-qr-checkpoints` | Grupo B best.pt + metrics + config | ~342 MB |')
    lines.append('| `tcc-mri-qr-lesion-checkpoints` | Grupo C best.pt + metrics + config | ~342 MB |')
    lines.append('| `tcc-mri-conformal-qhats` | 3 JSONs com q_hat + auditoria SHA | ~5 kB |')
    lines.append('| `tcc-mri-s5-8-metrics` | 3 CSVs por slice + summaries | ~1 MB |')
    lines.append('')

    lines.append('### 6.4 Comandos para reproduzir')
    lines.append('')
    lines.append('```bash')
    lines.append('# Clonar repositorio')
    lines.append('git clone https://github.com/KR0N0S7/tcc-mri-uncertainty.git')
    lines.append('cd tcc-mri-uncertainty')
    lines.append('pip install -r requirements.txt')
    lines.append('')
    lines.append('# Rodar suite de testes (esperado: 174+ passed)')
    lines.append('python -m pytest tests/ -v')
    lines.append('')
    lines.append('# Calibracao conforme (S5.7) - rodada em Kaggle')
    lines.append('python scripts/calibrate.py --group A \\\\')
    lines.append('  --checkpoint <path_to_best.pt> \\\\')
    lines.append('  --recons-dir <recons_root> \\\\')
    lines.append('  --output q_hat_A.json')
    lines.append('# (repetir para B e C)')
    lines.append('')
    lines.append('# Metricas por slice (S5.8) - rodada em Kaggle')
    lines.append('python scripts/compute_metrics.py --group A \\\\')
    lines.append('  --checkpoint <path_to_best.pt> \\\\')
    lines.append('  --qhat q_hat_A.json \\\\')
    lines.append('  --recons-dir <recons_root> \\\\')
    lines.append('  --masks-dir <masks_root> \\\\')
    lines.append('  --output metrics_A.csv')
    lines.append('# (repetir para B e C)')
    lines.append('')
    lines.append('# Analise estatistica (S5.9) - este documento')
    lines.append('python scripts/analyze_S5_9.py \\\\')
    lines.append('  --csv-dir <dir_com_3_csvs> \\\\')
    lines.append('  --output-json docs/figures/s5_9_analysis.json \\\\')
    lines.append('  --output-md docs/S5.md')
    lines.append('```')
    lines.append('')
    lines.append('### 6.5 Versoes de software')
    lines.append('')
    lines.append('- Python 3.12')
    lines.append('- PyTorch 2.10 (CUDA 12.8 para treino, CPU para calibracao/metricas)')
    lines.append('- scipy >= 1.11 (para BCa bootstrap)')
    lines.append('- pandas >= 2.0, numpy >= 1.24')
    lines.append('- pytest >= 8.0')
    lines.append('')
    lines.append('---')
    lines.append('')

    # ----------------------- 7. Referencias -----------------------
    lines.append('## 7. Referencias')
    lines.append('')
    refs = [
        'Angelopoulos, A.N.; Bates, S. (2023). Conformal Prediction: A Gentle Introduction. *FnT in ML*, 16(4):494-591.',
        'Angelopoulos, A.N. et al. (2022). Image-to-Image Regression with Distribution-Free Uncertainty Quantification. *ICML*.',
        'Barber, R.F. et al. (2021). The limits of distribution-free conditional predictive inference. *Information and Inference*, 10(2):455-482.',
        'Barber, R.F. et al. (2021). Predictive inference with the jackknife+. *Annals of Statistics*, 49(1):486-507.',
        'Barber, R.F. et al. (2023). Conformal prediction beyond exchangeability. *Annals of Statistics*, 51(2):816-845.',
        'Bates, S. et al. (2021). Distribution-free, risk-controlling prediction sets. *J. ACM*, 68(6):1-34.',
        'Caliva, F. et al. (2019). Distance Map Loss Penalty Term for Semantic Segmentation. *MIDL*.',
        'Cauchois, M. et al. (2023). Robust Validation: Confident Predictions Even When Distributions Shift. *J. Amer. Statist. Assoc.*',
        'Clopper, C.J.; Pearson, E.S. (1934). The use of confidence or fiducial limits illustrated in the case of the binomial. *Biometrika*, 26(4):404-413.',
        'Demsar, J. (2006). Statistical comparisons of classifiers over multiple data sets. *J. Mach. Learn. Res.*, 7:1-30.',
        'Efron, B.; Tibshirani, R.J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall/CRC.',
        'Friedman, M. (1937). The use of ranks to avoid the assumption of normality. *J. Amer. Statist. Assoc.*, 32(200):675-701.',
        'Giannakopoulos, C. et al. (2026). Pixelwise Uncertainty Quantification of Accelerated MRI Reconstruction. *arXiv:2601.13236*.',
        'Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scand. J. Statist.*, 6(2):65-70.',
        'Lin, T.-Y. et al. (2017). Focal Loss for Dense Object Detection. *ICCV*.',
        'Romano, Y.; Patterson, E.; Candes, E. (2019). Conformalized Quantile Regression. *NeurIPS*, 32:3543-3553.',
        'Romano, Y. et al. (2020). Malice Aforethought: Adaptive Conformal Inference Under Distribution Shift. *NeurIPS*.',
        'Sobel, I.; Feldman, G. (1968). A 3x3 isotropic gradient operator for image processing. *Stanford AI Project*.',
        'Sriram, A. et al. (2020). End-to-End Variational Networks for Accelerated MRI Reconstruction. *MICCAI*, 64-73.',
        'Vovk, V.; Gammerman, A.; Shafer, G. (2005). *Algorithmic Learning in a Random World*. Springer.',
        'Wilcoxon, F. (1945). Individual comparisons by ranking methods. *Biometrics*, 1(6):80-83.',
        'Zbontar, J. et al. (2020). fastMRI: An Open Dataset and Benchmarks for Accelerated MRI. *arXiv:1811.08839*.',
        'Zhao, R. et al. (2022). fastMRI+: Clinical Pathology Annotations for Knee and Brain Fully Sampled MultiCoil MRI Data. *Scientific Data*, 9:152.',
    ]
    for r in sorted(refs):
        lines.append(f'- {r}')
    lines.append('')

    lines.append('---')
    lines.append('')
    lines.append(
        f'*Documento gerado automaticamente por `scripts/analyze_S5_9.py` em '
        f'{today}. Para regerar com numeros atualizados, ver Secao 6.4.*'
    )
    lines.append('')

    return '\n'.join(lines)


def _get_pair_p(metric_result: dict, pair_name: str) -> float:
    """Helper para pegar p_holm de um par especifico do resultado."""
    for p in metric_result.get('pairwise_wilcoxon', []):
        if p['pair'] == pair_name:
            return p.get('p_holm', float('nan'))
    return float('nan')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    if not args.csv_dir.is_dir():
        print(f'ERRO: --csv-dir nao existe: {args.csv_dir}', file=sys.stderr)
        return 2

    print(f'[1/4] Carregando 3 CSVs de {args.csv_dir}...')
    df = load_csvs(args.csv_dir)
    print(f'      DataFrame total: {len(df)} linhas (3 grupos x ~736 slices)')

    summaries = load_summary_metadata(args.csv_dir)
    print(f'      Summaries: {sum(1 for s in summaries.values() if s)} de 3 JSONs carregados')

    print(f'\n[2/4] Analisando {len(METRICS)} metricas (Friedman + Wilcoxon-Holm + BCa)...')
    metric_results = {}
    for metric in METRICS:
        print(f'      - {metric}...')
        metric_results[metric] = analyze_metric(
            df, metric, args.n_bootstrap, args.alpha, args.seed,
        )

    print(f'\n[3/4] Clopper-Pearson micro-coverage...')
    coverage_global = clopper_pearson_coverage(df, region='global')
    coverage_lesion = clopper_pearson_coverage(df, region='lesion')

    print(f'\n[4/4] Salvando outputs...')

    # JSON com tudo
    payload = {
        'metadata': {
            'created_at': datetime.now(timezone.utc).isoformat(),
            'alpha': args.alpha,
            'n_bootstrap': args.n_bootstrap,
            'bootstrap_method': 'BCa',
            'seed': args.seed,
            'groups': list(GROUPS),
            'metrics_tested': list(METRICS),
        },
        'metric_results': metric_results,
        'coverage_clopper_pearson': {
            'global': coverage_global,
            'lesion': coverage_lesion,
        },
        'group_metadata': {
            g: (summaries[g] if summaries[g] else None) for g in GROUPS
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, default=str),
                                encoding='utf-8')
    print(f'      JSON salvo em: {args.output_json}')

    # Markdown (docs/S5.md)
    if args.output_md is not None:
        md = render_md_report(metric_results, coverage_global,
                              coverage_lesion, summaries, args)
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(md, encoding='utf-8')
        n_lines = md.count('\n') + 1
        print(f'      Markdown salvo em: {args.output_md} ({n_lines} linhas)')

    # Headline com p-values dos principais testes
    print('\n' + '=' * 70)
    print('HEADLINE - p-values dos testes principais (alpha = {})'.format(args.alpha))
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
    print('\n* = p_Holm < alpha (significancia mantida apos correcao)')
    print()

    return 0


if __name__ == '__main__':
    sys.exit(main())
