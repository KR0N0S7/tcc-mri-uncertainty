#!/usr/bin/env python3
# Autor: Massanori
# Data: 22/05/2026
# Descricao: Regera a tabela comparativa A vs B vs C do S5.8 a partir dos 3
#            arquivos summary.json gerados por scripts/compute_metrics.py.
#            Util para:
#              (1) Conferir a tabela offline sem rerodar o notebook Kaggle.
#              (2) Exportar a tabela em formato Markdown para colar direto
#                  no docs/S5.md (--output-md <path>).
#              (3) Corrigir o bug cosmetico de truncamento que existia na
#                  Celula 7 do notebook anterior (largura insuficiente).
#            Recebe via CLI: --summary-dir (com os 3 JSONs).
#            Saida: tabela formatada no stdout + (opcional) arquivo .md.


"""Imprime tabela comparativa A vs B vs C do S5.8 a partir dos summary JSONs.

Roda com:
    # Tabela no terminal (formato ASCII)
    python scripts/print_metrics_table.py \\
        --summary-dir /caminho/com/metrics_A.summary.json/etc

    # Tambem exporta como Markdown para o docs/S5.md
    python scripts/print_metrics_table.py \\
        --summary-dir <dir> \\
        --output-md docs/figures/s5_8_table.md

Utilidade extra: se o usuario baixou os JSONs do Kaggle para o laptop
(para fazer parte da analise localmente), este script regenera a tabela
sem precisar rerodar nada.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

GROUPS = ('A', 'B', 'C')
GROUP_LABELS = {
    'A': 'A (ResM)',
    'B': 'B (QR)',
    'C': 'C (QR-Lesion)',
}

# Metricas a reportar, em ordem de relevancia para a banca.
METRIC_KEYS = (
    'coverage_global',
    'coverage_lesion',
    'mean_width_global',
    'mean_width_lesion',
    'iou_topk_global',
    'iou_topk_lesion',
    'ulas_lesion',
    'ulas_null_mean',
    'ulas_z_score',
)

METRIC_LABELS = {
    'coverage_global': 'Coverage_global',
    'coverage_lesion': 'Coverage_lesion',
    'mean_width_global': 'Width_global',
    'mean_width_lesion': 'Width_lesion',
    'iou_topk_global': 'IoU_topk_global',
    'iou_topk_lesion': 'IoU_topk_lesion',
    'ulas_lesion': 'ULAS_lesion',
    'ulas_null_mean': 'ULAS_null',
    'ulas_z_score': 'ULAS_z_score',
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Imprime tabela comparativa S5.8 a partir dos summary JSONs.'
    )
    parser.add_argument(
        '--summary-dir', type=Path, required=True,
        help='Diretorio com metrics_A.summary.json, metrics_B.summary.json, '
             'metrics_C.summary.json.'
    )
    parser.add_argument(
        '--output-md', type=Path, default=None,
        help='Se fornecido, salva tabela no formato Markdown neste arquivo.'
    )
    parser.add_argument(
        '--no-ascii', action='store_true',
        help='Suprime tabela ASCII no stdout (uso silencioso quando so '
             'queremos gerar o Markdown).'
    )
    return parser.parse_args()


def load_summaries(summary_dir: Path) -> dict:
    """Carrega os 3 summary JSONs do diretorio. Erra com mensagem clara."""
    summaries = {}
    for g in GROUPS:
        path = summary_dir / f'metrics_{g}.summary.json'
        if not path.is_file():
            print(f'ERRO: nao encontrado: {path}', file=sys.stderr)
            print(f'      Conteudo de {summary_dir}:', file=sys.stderr)
            for p in sorted(summary_dir.iterdir()):
                print(f'        {p.name}', file=sys.stderr)
            sys.exit(2)
        summaries[g] = json.loads(path.read_text(encoding='utf-8'))
    return summaries


def fmt_cell(s: dict, key: str) -> str:
    m = s['metrics_summary'][key]
    return f'{m["mean"]:.4f} \u00b1 {m["std"]:.4f} (n={m["n_valid"]})'


def print_ascii_table(summaries: dict) -> None:
    """Imprime tabela ASCII larga o suficiente para nao truncar nada."""
    # Larguras calibradas: '0.0000 \u00b1 0.0000 (n=99999)' = 24 chars; folga 5
    col_width = 29
    key_width = 18  # 'Coverage_global' + spacing
    total_width = key_width + 3 * col_width

    print('=' * total_width)
    print(f'METRICAS S5.8 \u2014 Comparativo A vs B vs C (split test, alpha=0.10)')
    print('=' * total_width)

    # Headline por grupo: n + qhat + sha
    for g in GROUPS:
        s = summaries[g]
        print(
            f'Grupo {g}: {s["n_slices_total"]} slices '
            f'({s["n_slices_with_lesion"]} com lesao), '
            f'q_hat={s["q_hat"]:.6f}, '
            f'sha256={s["checkpoint_sha256"][:8]}...'
        )
    print()

    # Header
    header = f'{"Metrica":<{key_width}}'
    for g in GROUPS:
        header += f'{GROUP_LABELS[g]:<{col_width}}'
    print(header)
    print('-' * total_width)

    # Linhas
    for key in METRIC_KEYS:
        row = f'{METRIC_LABELS[key]:<{key_width}}'
        for g in GROUPS:
            row += f'{fmt_cell(summaries[g], key):<{col_width}}'
        print(row)

    print('=' * total_width)


def render_markdown_table(summaries: dict) -> str:
    """Renderiza tabela em Markdown com headline e rodape de leitura."""
    lines = []
    lines.append('# S5.8 \u2014 Metricas por Regiao + ULAS (split test, alpha=0.10)')
    lines.append('')
    lines.append('## Metadata por grupo')
    lines.append('')
    lines.append('| Grupo | Slices total | Slices com lesao | q_hat | Checkpoint SHA-256 |')
    lines.append('|---|---|---|---|---|')
    for g in GROUPS:
        s = summaries[g]
        lines.append(
            f'| {GROUP_LABELS[g]} | {s["n_slices_total"]} | '
            f'{s["n_slices_with_lesion"]} | {s["q_hat"]:.6f} | '
            f'`{s["checkpoint_sha256"][:16]}...` |'
        )
    lines.append('')

    lines.append('## Comparativo de metricas (mean \u00b1 std (n_valid))')
    lines.append('')
    lines.append('| Metrica | A (ResM) | B (QR) | C (QR-Lesion) |')
    lines.append('|---|---|---|---|')
    for key in METRIC_KEYS:
        row_cells = [METRIC_LABELS[key]]
        for g in GROUPS:
            row_cells.append(fmt_cell(summaries[g], key))
        lines.append('| ' + ' | '.join(row_cells) + ' |')
    lines.append('')

    lines.append('## Notas')
    lines.append('')
    lines.append('- `Coverage` reportado como macro-average por slice. '
                 'Cobertura nominal alvo: 1 - alpha = 0.90.')
    lines.append('- `Width` em escala normalizada por `max_val` do volume (D1).')
    lines.append('- `IoU_topk` com X = 5% (top-X% mais incertos vs top-X% com maior erro).')
    lines.append('- `ULAS_null` = media de 10 permutacoes do error_map; '
                 '`ULAS_z_score = (ULAS_real - ULAS_null) / std_null`.')
    lines.append('- Para n_valid das metricas globais: total de slices test. '
                 'Para metricas de lesao: total de slices com pelo menos 1 '
                 'pixel de lesao.')
    lines.append('')
    lines.append('## Pr\u00f3ximo passo (S5.9)')
    lines.append('')
    lines.append('Analise estatistica formal sobre os CSVs por slice:')
    lines.append('- **Friedman test + Nemenyi post-hoc** (Dem\u0161ar, 2006) para '
                 'compara\u00e7\u00f5es entre os 3 grupos.')
    lines.append('- **Holm-Bonferroni** para correcao em multiplas comparacoes par-a-par.')
    lines.append('- **BCa bootstrap** para CIs 95% das metricas agregadas.')
    lines.append('- **Clopper-Pearson** para IC exato de Coverage como proporcao.')
    lines.append('')
    return '\n'.join(lines)


def main() -> int:
    args = parse_args()

    if not args.summary_dir.is_dir():
        print(f'ERRO: --summary-dir nao existe: {args.summary_dir}',
              file=sys.stderr)
        return 2

    summaries = load_summaries(args.summary_dir)

    if not args.no_ascii:
        print_ascii_table(summaries)

    if args.output_md is not None:
        md_content = render_markdown_table(summaries)
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(md_content, encoding='utf-8')
        print(f'\nMarkdown salvo em: {args.output_md}', file=sys.stderr)

    return 0


if __name__ == '__main__':
    sys.exit(main())
