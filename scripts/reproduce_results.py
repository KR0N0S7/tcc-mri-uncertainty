# Autor: Massanori
# Data: 10/06/2026
# Descricao: Orquestrador de reproducao dos resultados (S5.7 -> S5.8 -> S5.9)
#            num unico comando. A partir dos checkpoints ja treinados dos 3
#            grupos (A/ResM, B/QR, C/QR-Lesion), encadeia:
#            (1) calibrate.py    -> q_hat_{A,B,C}.json   (calibracao conforme)
#            (2) compute_metrics -> metrics_{A,B,C}.csv   (metricas por slice)
#            (3) analyze_S5_9.py -> docs/figures/s5_9_analysis.json + docs/S5.md
#            Recebe os paths dos checkpoints, recons-dir, masks-dir e um
#            diretorio de trabalho; gera os artefatos de resultado. NAO executa
#            as etapas pesadas de GPU (precompute do S4 e treino do S5.2-5.4),
#            que sao upstream e documentadas no README. Cada etapa e resumivel:
#            pula se a saida ja existe (use --force para recomputar). Chama os
#            scripts existentes via subprocess para nao duplicar logica.
r"""Reproduz os resultados (calibracao -> metricas -> analise) com um comando.

Exemplo (Kaggle ou local), com checkpoints e dados ja montados:

    python scripts/reproduce_results.py \
        --ckpt-a /path/group_a/best.pt \
        --ckpt-b /path/group_b/best.pt \
        --ckpt-c /path/group_c/best.pt \
        --recons-dir /path/recons \
        --masks-dir  /path/masks \
        --work-dir   results/repro

Saidas geradas:
    <work-dir>/q_hat_A.json, q_hat_B.json, q_hat_C.json
    <work-dir>/metrics_A.csv, metrics_B.csv, metrics_C.csv  (+ .summary.json)
    docs/figures/s5_9_analysis.json
    docs/S5.md

Pre-requisitos (upstream, nao executados aqui):
    - S4  : scripts/precompute_reconstructions.py  (reconstrucoes E2E-VarNet)
    - S5.2-5.4 : scripts/train_{resm,qr,qr_lesion}.py  (best.pt por grupo)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / 'scripts'
GROUPS = ('A', 'B', 'C')


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Orquestra S5.7->S5.8->S5.9 a partir dos checkpoints treinados.'
    )
    p.add_argument('--ckpt-a', type=Path, required=True,
                   help='best.pt do Grupo A (ResM).')
    p.add_argument('--ckpt-b', type=Path, required=True,
                   help='best.pt do Grupo B (QR).')
    p.add_argument('--ckpt-c', type=Path, required=True,
                   help='best.pt do Grupo C (QR-Lesion).')
    p.add_argument('--recons-dir', type=Path, required=True,
                   help='Raiz das reconstrucoes com subdirs cal/ e test/.')
    p.add_argument('--masks-dir', type=Path, required=True,
                   help='Diretorio das mascaras .pt do fastMRI+.')
    p.add_argument('--work-dir', type=Path, default=ROOT / 'results' / 'repro',
                   help='Onde salvar q_hat_*.json e metrics_*.csv.')
    p.add_argument('--analysis-json', type=Path,
                   default=ROOT / 'docs' / 'figures' / 's5_9_analysis.json',
                   help='Saida JSON do S5.9.')
    p.add_argument('--analysis-md', type=Path,
                   default=ROOT / 'docs' / 'S5.md',
                   help='Saida Markdown do S5.9 (docs/S5.md).')
    p.add_argument('--alpha', type=float, default=0.10,
                   help='Alpha da calibracao/metricas (cobertura nominal 1-alpha).')
    p.add_argument('--analysis-alpha', type=float, default=0.05,
                   help='Alpha dos testes de hipotese no S5.9.')
    p.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    p.add_argument('--force', action='store_true',
                   help='Recomputa etapas mesmo que a saida ja exista.')
    p.add_argument('--skip-calibrate', action='store_true',
                   help='Pula a etapa 1 (usa q_hat_*.json existentes).')
    p.add_argument('--skip-metrics', action='store_true',
                   help='Pula a etapa 2 (usa metrics_*.csv existentes).')
    p.add_argument('--skip-analyze', action='store_true',
                   help='Pula a etapa 3 (nao gera S5.md).')
    return p.parse_args()


def run(cmd: list) -> None:
    """Roda um subcomando, abortando o pipeline em caso de erro."""
    printable = ' '.join(str(c) for c in cmd)
    print('\n>>> ' + printable + '\n', flush=True)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(
            'ERRO: etapa falhou (returncode='
            + str(result.returncode) + '): ' + printable,
            file=sys.stderr,
        )
        sys.exit(result.returncode)


def main() -> int:
    args = parse_args()
    ckpts = {'A': args.ckpt_a, 'B': args.ckpt_b, 'C': args.ckpt_c}
    args.work_dir.mkdir(parents=True, exist_ok=True)
    py = sys.executable

    # Etapa 1 - Calibracao conforme (S5.7) -> q_hat_{A,B,C}.json
    qhats = {g: args.work_dir / ('q_hat_' + g + '.json') for g in GROUPS}
    if not args.skip_calibrate:
        for g in GROUPS:
            if qhats[g].exists() and not args.force:
                print(
                    '[calibrate ' + g + '] ja existe (' + str(qhats[g])
                    + '), pulando. Use --force para refazer.'
                )
                continue
            cmd = [
                py, str(SCRIPTS / 'calibrate.py'),
                '--group', g,
                '--checkpoint', str(ckpts[g]),
                '--recons-dir', str(args.recons_dir),
                '--output', str(qhats[g]),
                '--alpha', str(args.alpha),
                '--device', args.device,
            ]
            if g == 'C':
                # masks-dir e opcional na calibracao (nao afeta q_hat), mas
                # mantem consistencia com o Dataset usado no Grupo C.
                cmd += ['--masks-dir', str(args.masks_dir)]
            run(cmd)

    # Etapa 2 - Metricas por slice (S5.8) -> metrics_{A,B,C}.csv
    csvs = {g: args.work_dir / ('metrics_' + g + '.csv') for g in GROUPS}
    if not args.skip_metrics:
        for g in GROUPS:
            if csvs[g].exists() and not args.force:
                print(
                    '[metrics ' + g + '] ja existe (' + str(csvs[g])
                    + '), pulando. Use --force para refazer.'
                )
                continue
            cmd = [
                py, str(SCRIPTS / 'compute_metrics.py'),
                '--group', g,
                '--checkpoint', str(ckpts[g]),
                '--qhat', str(qhats[g]),
                '--recons-dir', str(args.recons_dir),
                '--masks-dir', str(args.masks_dir),
                '--output', str(csvs[g]),
                '--alpha', str(args.alpha),
                '--device', args.device,
            ]
            run(cmd)

    # Etapa 3 - Analise estatistica (S5.9) -> JSON + docs/S5.md
    if not args.skip_analyze:
        cmd = [
            py, str(SCRIPTS / 'analyze_S5_9.py'),
            '--csv-dir', str(args.work_dir),
            '--output-json', str(args.analysis_json),
            '--output-md', str(args.analysis_md),
            '--alpha', str(args.analysis_alpha),
        ]
        run(cmd)

    print('\n' + '=' * 60)
    print('REPRODUCAO CONCLUIDA')
    print('  q_hats:    ' + str(args.work_dir) + '/q_hat_[A,B,C].json')
    print('  metricas:  ' + str(args.work_dir) + '/metrics_[A,B,C].csv')
    print('  analise:   ' + str(args.analysis_json))
    print('  relatorio: ' + str(args.analysis_md))
    print('=' * 60)
    return 0


if __name__ == '__main__':
    sys.exit(main())
