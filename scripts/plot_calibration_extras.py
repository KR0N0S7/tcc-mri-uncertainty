# Autor: Massanori
# Data: 02/06/2026
# Descrição: Figuras das analises de calibracao, destino:
#            scripts/plot_calibration_extras.py. Le todos os CSVs gerados pelo
#            analyze_calibration_sweep.py (glob sweep_*.csv) e produz:
#              - Item 4: curva de confiabilidade (cobertura empirica vs nivel
#                        nominal), global e em lesao, uma linha por calibrador.
#                        A diagonal y=x e' a calibracao perfeita.
#              - Item 2: fronteira de eficiencia (cobertura_lesion vs
#                        largura_lesion), permitindo ler "cobertura a largura
#                        igualada" — comparacao justa entre ResM e CQR.
#              - Item 1: tabela-resumo no nivel 0.90 (CQR-Norm vs CQR vs ScaledCP)
#                        para isolar o mecanismo do achado de Coverage_lesion.
#            Recebe via CLI: --sweep-dir (com os sweep_*.csv), --out-dir.
#            Gera: reliability_curve.png, efficiency_frontier.png (300 DPI,
#            colormap perceptualmente uniforme — NUNCA jet), e
#            summary_level090.csv. Sem dados pessoais/sensiveis (apenas IDs de
#            grupo/calibrador). Fundamentos: Angelopoulos & Bates (2023);
#            Lei et al. (2018, Fig. 7: locally-weighted pode ser mais largo).

"""Figuras e tabela das analises de calibracao (itens 1, 2, 4)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # backend headless (Kaggle/CI)
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# Estilo consistente: fontes legiveis, sem jet.
plt.rcParams.update({
    'figure.dpi': 120, 'savefig.dpi': 300, 'font.size': 12,
    'axes.grid': True, 'grid.alpha': 0.3, 'axes.spines.top': False,
    'axes.spines.right': False,
})

# Rotulos legiveis por (grupo, calibrador).
LABELS = {
    ('A', 'scaled'): 'A · ResM (scaled CP)',
    ('B', 'cqr'): 'B · CQR aditivo',
    ('B', 'cqr_norm'): 'B · CQR normalizado',
    ('C', 'cqr'): 'C · QR-Lesion (CQR aditivo)',
    ('C', 'cqr_norm'): 'C · QR-Lesion (CQR normalizado)',
}
# Cores qualitativas (tab10), estaveis por serie.
COLORS = {
    ('A', 'scaled'): '#1f77b4',
    ('B', 'cqr'): '#ff7f0e',
    ('B', 'cqr_norm'): '#2ca02c',
    ('C', 'cqr'): '#d62728',
    ('C', 'cqr_norm'): '#9467bd',
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Figuras das analises de calibracao.')
    p.add_argument('--sweep-dir', type=Path, required=True,
                   help='Diretorio com os arquivos sweep_*.csv.')
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--pattern', default='sweep_*.csv')
    return p.parse_args()


def load_all(sweep_dir: Path, pattern: str) -> pd.DataFrame:
    files = sorted(sweep_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f'Nenhum {pattern} em {sweep_dir}')
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df['key'] = list(zip(df['group'], df['calibrator']))
    return df


def _series(df: pd.DataFrame):
    """Itera series (group, calibrator) na ordem canonica de LABELS."""
    present = list(dict.fromkeys(df['key']))
    ordered = [k for k in LABELS if k in present] + \\
              [k for k in present if k not in LABELS]
    for k in ordered:
        yield k, df[df['key'] == k].sort_values('nominal_coverage')


def plot_reliability(df: pd.DataFrame, out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, col, title in [
        (axes[0], 'coverage_global', 'Cobertura global'),
        (axes[1], 'coverage_lesion', 'Cobertura em lesao'),
    ]:
        ax.plot([0.5, 1.0], [0.5, 1.0], ls='--', color='gray', lw=1,
                label='calibracao perfeita')
        for k, g in _series(df):
            ax.plot(g['nominal_coverage'], g[col], marker='o', ms=4,
                    color=COLORS.get(k), label=LABELS.get(k, str(k)))
        ax.axvline(0.90, color='k', lw=0.8, alpha=0.4)
        ax.set_xlabel('Cobertura nominal (1 - alpha)')
        ax.set_ylabel('Cobertura empirica (test)')
        ax.set_title(title)
        ax.set_xlim(0.5, 1.0); ax.set_ylim(0.5, 1.0)
    axes[0].legend(fontsize=9, loc='upper left')
    fig.suptitle('Confiabilidade da calibracao — empirico vs nominal', y=1.02)
    fig.tight_layout()
    out = out_dir / 'reliability_curve.png'
    fig.savefig(out, bbox_inches='tight'); plt.close(fig)
    return out


def plot_efficiency(df: pd.DataFrame, out_dir: Path) -> Path:
    """Fronteira de eficiencia em lesao: cobertura_lesion vs largura_lesion.

    Permite leitura "cobertura a largura igualada": uma vertical em uma
    largura fixa cruza cada curva na cobertura atingida por aquele metodo.
    """
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for k, g in _series(df):
        gg = g.dropna(subset=['width_lesion', 'coverage_lesion'])
        ax.plot(gg['width_lesion'], gg['coverage_lesion'], marker='o', ms=4,
                color=COLORS.get(k), label=LABELS.get(k, str(k)))
        # marca o ponto de nivel 0.90
        p90 = gg[abs(gg['nominal_coverage'] - 0.90) < 1e-9]
        if len(p90):
            ax.scatter(p90['width_lesion'], p90['coverage_lesion'],
                       s=90, facecolors='none', edgecolors=COLORS.get(k), lw=2,
                       zorder=5)
    ax.axhline(0.90, color='k', lw=0.8, alpha=0.4, label='alvo 0.90')
    ax.set_xlabel('Largura media do intervalo em lesao (escala max_val)')
    ax.set_ylabel('Cobertura empirica em lesao')
    ax.set_title('Fronteira de eficiencia em lesao\n'
                 '(circulos vazios = nivel nominal 0.90)')
    ax.legend(fontsize=9, loc='lower right')
    fig.tight_layout()
    out = out_dir / 'efficiency_frontier.png'
    fig.savefig(out, bbox_inches='tight'); plt.close(fig)
    return out


def table_level090(df: pd.DataFrame, out_dir: Path) -> Path:
    sub = df[abs(df['nominal_coverage'] - 0.90) < 1e-9].copy()
    sub['serie'] = sub['key'].map(lambda k: LABELS.get(k, str(k)))
    cols = ['serie', 'group', 'calibrator', 'q_hat',
            'coverage_global', 'coverage_lesion', 'width_global', 'width_lesion']
    sub = sub[cols].sort_values(['group', 'calibrator'])
    out = out_dir / 'summary_level090.csv'
    sub.to_csv(out, index=False)
    # Print legivel
    print('\n=== Nivel nominal 0.90 ===')
    with pd.option_context('display.float_format', lambda x: f'{x:.4f}'):
        print(sub.to_string(index=False))
    print('\nLeitura item 1: se "CQR normalizado" recupera coverage_lesion '
          'proximo do "ResM (scaled CP)", o ganho do ResM vem da '
          'adaptatividade local da calibracao, nao da arquitetura.')
    return out


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = load_all(args.sweep_dir, args.pattern)
    f1 = plot_reliability(df, args.out_dir)
    f2 = plot_efficiency(df, args.out_dir)
    t1 = table_level090(df, args.out_dir)
    print(f'\nFiguras: {f1}\n         {f2}\nTabela : {t1}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
