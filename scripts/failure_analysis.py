#!/usr/bin/env python3
# Autor: Massanori
# Data: 12/06/2026
# Descricao: Analise de falha estruturada (item 2.3 do Bloco 2 / pre-entrega
#            da 5a entrega). Ranqueia os piores slices do split test para um
#            grupo (default C = QR-Lesion) por gap de cobertura em lesao
#            (nominal - Coverage_lesion) ou por ULAS_lesion ascendente,
#            exporta os N piores com atributos (sequencia, area da lesao em
#            px^2, cobertura, largura, ULAS) num CSV auditavel e renderiza
#            painel de figuras (GT | recon | |erro| | largura | falha de
#            cobertura) para os K piores, com contorno da lesao sobreposto.
#            Recebe via CLI: --metrics-csv (metrics_C.csv do S5.8, usado para
#            ranquear sem GPU), --checkpoint + --qhat + --recons-dir +
#            --masks-dir (para reabrir os K piores e regerar intervalos).
#            Gera: results/failure_top{N}_group{G}.csv e
#            figures/failures/*.png (300 DPI). Pos-treino, custo de GPU nulo:
#            ranqueia do CSV e so faz forward nos K slices selecionados.
#            Sem narrativa hardcoded: o CSV e as figuras refletem os dados.

"""Analise de falha estruturada do Grupo C sobre o split test.

Pipeline em duas etapas para nao desperdicar GPU/CPU:

    Etapa 1 (ranqueamento, sem modelo): le o metrics_C.csv ja computado no
    S5.8, restringe a slices com lesao e ordena pelos piores casos.

    Etapa 2 (renderizacao, so nos K piores): reabre apenas os slices
    selecionados via ReconsSliceDataset, roda o forward do grupo, aplica o
    intervalo calibrado (q_hat do S5.7) e desenha o painel de diagnostico.

Roda com:
    python scripts/failure_analysis.py \\
        --metrics-csv /caminho/metrics_C.csv \\
        --checkpoint  /caminho/group_c_full/best.pt \\
        --qhat        /caminho/q_hat_C.json \\
        --recons-dir  /caminho/recons (com test/) \\
        --masks-dir   /caminho/masks \\
        --output-dir-csv results --output-dir-fig figures/failures \\
        --group C --n-worst 10 --n-figures 3 --rank-by coverage_gap

Refs:
    Romano et al. (2019), Conformalized Quantile Regression, NeurIPS.
    Angelopoulos & Bates (2023), A Gentle Introduction to Conformal
    Prediction (interpretacao de under-coverage como falha de garantia).
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402
from src.calibration import apply_cqr_interval, apply_resm_interval  # noqa: E402
from src.data import ReconsSliceDataset  # noqa: E402
from src.losses import DEFAULT_ALPHA  # noqa: E402
from src.models import QuantileRegressionModule, ResidualMagnitudeModule  # noqa: E402
from src.training import load_checkpoint  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
)
logger = logging.getLogger('failure_analysis')

# Colunas do CSV de saida (uma linha por slice ruim).
CSV_FIELDS = [
    'rank', 'volume_id', 'slice_idx', 'sequence',
    'n_pixels_lesion', 'lesion_area_bbox_px2',
    'nominal_coverage', 'coverage_lesion', 'coverage_gap',
    'mean_width_lesion', 'ulas_lesion', 'ulas_z_score',
    'artifact_type',
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Analise de falha estruturada (10 piores casos) do S5.'
    )
    p.add_argument('--metrics-csv', type=Path, required=True,
                   help='CSV por slice do grupo (metrics_C.csv do S5.8).')
    p.add_argument('--group', default='C', choices=['A', 'B', 'C'],
                   help='Grupo a auditar (default C = QR-Lesion).')
    p.add_argument('--rank-by', default='coverage_gap',
                   choices=['coverage_gap', 'ulas'],
                   help='Criterio de pior caso. coverage_gap = nominal menos '
                        'Coverage_lesion (desc). ulas = ULAS_lesion (asc).')
    p.add_argument('--n-worst', type=int, default=10,
                   help='Quantos piores casos exportar no CSV.')
    p.add_argument('--n-figures', type=int, default=3,
                   help='Quantos dos piores renderizar como figura.')
    p.add_argument('--alpha', type=float, default=None,
                   help='Override do alpha. Default: le do --qhat, senao '
                        f'{DEFAULT_ALPHA}.')
    p.add_argument('--brain-csv', type=Path, default=None,
                   help='brain.csv do fastMRI+ (opcional) para anexar a area '
                        'da maior bbox por slice. Sem isso, fica NaN.')
    # Etapa 2 (renderizacao) - opcional: se ausente, so gera o CSV.
    p.add_argument('--checkpoint', type=Path, default=None)
    p.add_argument('--qhat', type=Path, default=None,
                   help='JSON do S5.7 com q_hat (necessario p/ figuras).')
    p.add_argument('--recons-dir', type=Path, default=None,
                   help='Dir com split test/ (.npz). Necessario p/ figuras.')
    p.add_argument('--masks-dir', type=Path, default=None,
                   help='Dir com mascaras .pt. Necessario p/ figuras.')
    p.add_argument('--output-dir-csv', type=Path, default=ROOT / 'results')
    p.add_argument('--output-dir-fig', type=Path,
                   default=ROOT / 'figures' / 'failures')
    p.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    p.add_argument('--chans', type=int, default=32)
    p.add_argument('--num-pool-layers', type=int, default=4)
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Etapa 1: ranqueamento (puro pandas, sem modelo)
# --------------------------------------------------------------------------- #
def load_ranked(metrics_csv: Path, rank_by: str, alpha: float):
    """Le metrics CSV, restringe a slices com lesao e ordena pelos piores.

    Retorna (df_worst_ordenado, df_lesion_completo). coverage_gap = nominal
    menos Coverage_lesion (positivo = sub-cobertura, que e a falha de
    interesse). NaN em coverage_lesion sao slices sem lesao -> descartados.
    """
    import pandas as pd

    df = pd.read_csv(metrics_csv)
    nominal = 1.0 - alpha

    # Slices com lesao = onde coverage_lesion e finita e ha pixels de lesao.
    mask = df['n_pixels_lesion'].fillna(0) > 0
    if 'coverage_lesion' in df.columns:
        mask &= df['coverage_lesion'].notna()
    df_les = df[mask].copy()

    df_les['nominal_coverage'] = nominal
    df_les['coverage_gap'] = nominal - df_les['coverage_lesion']

    if rank_by == 'coverage_gap':
        # Pior = maior sub-cobertura (gap mais positivo).
        df_sorted = df_les.sort_values('coverage_gap', ascending=False)
    else:  # ulas: pior = menor alinhamento
        df_sorted = df_les.sort_values('ulas_lesion', ascending=True)

    return df_sorted, df_les


def attach_bbox_area(df_worst, brain_csv: Optional[Path]):
    """Anexa a area da maior bbox por slice a partir do brain.csv.

    Match best-effort: o volume_id do CSV de metricas costuma conter o stem
    do arquivo .h5; cruzamos por substring com a coluna 'file' do brain.csv
    e usamos a coluna 'slice'. Se nada casar, a coluna fica NaN (degrada
    sem quebrar).
    """
    import pandas as pd

    df_worst = df_worst.copy()
    df_worst['lesion_area_bbox_px2'] = np.nan
    if brain_csv is None or not Path(brain_csv).is_file():
        return df_worst

    bdf = pd.read_csv(brain_csv).dropna(subset=['x', 'y', 'width', 'height'])
    bdf['area'] = bdf['width'] * bdf['height']

    for idx, row in df_worst.iterrows():
        vid = str(row['volume_id'])
        sidx = int(row['slice_idx'])
        # 'file' do brain.csv pode ter extensao/sufixo; casa por substring.
        cand = bdf[bdf['file'].apply(lambda f: str(f) in vid or vid in str(f))]
        cand = cand[cand['slice'] == sidx]
        if len(cand) > 0:
            df_worst.at[idx, 'lesion_area_bbox_px2'] = float(cand['area'].max())
    return df_worst


def write_worst_csv(df_worst, out_csv: Path, n_worst: int):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for rank, (_, r) in enumerate(df_worst.head(n_worst).iterrows(), start=1):
        rows.append({
            'rank': rank,
            'volume_id': r['volume_id'],
            'slice_idx': int(r['slice_idx']),
            'sequence': r.get('sequence', ''),
            'n_pixels_lesion': int(r['n_pixels_lesion']),
            'lesion_area_bbox_px2': (
                '' if (isinstance(r.get('lesion_area_bbox_px2'), float)
                       and math.isnan(r.get('lesion_area_bbox_px2')))
                else r.get('lesion_area_bbox_px2', '')
            ),
            'nominal_coverage': round(float(r['nominal_coverage']), 4),
            'coverage_lesion': round(float(r['coverage_lesion']), 4),
            'coverage_gap': round(float(r['coverage_gap']), 4),
            'mean_width_lesion': round(float(r.get('mean_width_lesion', float('nan'))), 6),
            'ulas_lesion': round(float(r.get('ulas_lesion', float('nan'))), 4),
            'ulas_z_score': round(float(r.get('ulas_z_score', float('nan'))), 3),
            # Atributo qualitativo: deixado em branco para anotacao MANUAL
            # pelo especialista (tipo de artefato nao e inferivel do CSV).
            'artifact_type': '',
        })
    with out_csv.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)
    logger.info(f'CSV dos piores casos: {out_csv} ({len(rows)} linhas)')
    return rows


# --------------------------------------------------------------------------- #
# Etapa 2: renderizacao (so nos K piores)
# --------------------------------------------------------------------------- #
def resolve_device(arg: str) -> str:
    if arg == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    if arg == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA solicitado mas indisponivel.')
    return arg


def compute_interval(group: str, forward_out, recon: torch.Tensor, q_hat: float):
    """Espelha scripts/compute_metrics.py::_compute_interval."""
    if group == 'A':
        u = forward_out
        lo, hi = apply_resm_interval(recon, u, q_hat)
        return lo, hi
    lower, upper = forward_out['lower'], forward_out['upper']
    return apply_cqr_interval(lower, upper, q_hat)


def _to_2d(t: torch.Tensor) -> np.ndarray:
    a = t.detach().cpu().numpy()
    return np.squeeze(a)


def render_panel(slice_data: dict, lower, upper, q_hat: float, alpha: float,
                 out_png: Path, meta: dict):
    """Painel 1x5: GT | recon | |erro| | largura do intervalo | falha.

    A ultima coluna marca em vermelho os pixels DENTRO da lesao que ficaram
    FORA do intervalo (a falha de cobertura concreta). Contorno da lesao em
    ciano em todos os paineis. 300 DPI, inferno para mapas, fonte 12pt.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    gt = _to_2d(slice_data['target'])
    rec = _to_2d(slice_data['recon'])
    lo = _to_2d(lower)
    hi = _to_2d(upper)
    mask = _to_2d(slice_data['lesion_mask']) > 0.5
    err = np.abs(gt - rec)
    width = hi - lo
    inside = (gt >= lo) & (gt <= hi)
    miss_in_lesion = mask & (~inside)

    fig, axes = plt.subplots(1, 5, figsize=(20, 4.4))
    panels = [
        ('Ground truth', gt, 'gray', None),
        ('Reconstrucao (VarNet)', rec, 'gray', None),
        ('|Erro| = |GT - recon|', err, 'inferno', None),
        ('Largura do intervalo', width, 'inferno', None),
        ('Falha de cobertura\n(lesao, fora do intervalo)', err, 'gray', 'miss'),
    ]
    for ax, (title, img, cmap, overlay) in zip(axes, panels):
        im = ax.imshow(img, cmap=cmap)
        if overlay == 'miss':
            red = np.zeros((*miss_in_lesion.shape, 4))
            red[miss_in_lesion] = [1.0, 0.0, 0.0, 0.9]
            ax.imshow(red)
        # Contorno da lesao
        if mask.any():
            ax.contour(mask.astype(float), levels=[0.5],
                       colors='cyan', linewidths=1.0)
        ax.set_title(title, fontsize=12)
        ax.set_xticks([]); ax.set_yticks([])
        if cmap == 'inferno':
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    cov = meta['coverage_lesion']
    nominal = 1.0 - alpha
    fig.suptitle(
        f'Falha #{meta["rank"]} | grupo {meta["group"]} | {meta["sequence"]} | '
        f'{meta["volume_id"]} slice {meta["slice_idx"]} | '
        f'Cov_lesao={cov:.3f} (nominal {nominal:.2f}, '
        f'gap={nominal - cov:+.3f}) | ULAS={meta["ulas_lesion"]:.3f} | '
        f'q_hat={q_hat:.4f}',
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f'Figura: {out_png}')


def render_worst(worst_rows: list, args, alpha: float):
    """Reabre apenas os slices em worst_rows[:n_figures] e renderiza."""
    if args.checkpoint is None or args.qhat is None:
        logger.warning(
            'Sem --checkpoint/--qhat: pulando figuras (CSV gerado). '
            'Para as figuras, reexecute apontando o best.pt e q_hat_C.json.'
        )
        return

    qpayload = json.loads(Path(args.qhat).read_text())
    q_hat = float(qpayload['q_hat'])

    device = resolve_device(args.device)
    logger.info(f'Device (renderizacao): {device}')

    if args.group == 'A':
        module = ResidualMagnitudeModule(chans=args.chans,
                                         num_pool_layers=args.num_pool_layers)
    else:
        module = QuantileRegressionModule(chans=args.chans,
                                          num_pool_layers=args.num_pool_layers)
    load_checkpoint(args.checkpoint, module, device=device)
    module = module.to(device).eval()

    recons_root = (args.recons_dir or cfg.recons_dir()).expanduser().resolve()
    test_dir = recons_root / 'test'
    masks_dir = (args.masks_dir or cfg.masks_dir()).expanduser().resolve()
    ds = ReconsSliceDataset(test_dir, masks_dir=masks_dir)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    # Conjunto-alvo (volume_id, slice_idx) -> metadados do rank.
    targets = {(r['volume_id'], int(r['slice_idx'])): r
               for r in worst_rows[:args.n_figures]}
    found = 0
    with torch.no_grad():
        for batch in loader:
            vid = batch['volume_id'][0]
            sidx = (int(batch['slice_idx'][0].item())
                    if torch.is_tensor(batch['slice_idx'][0])
                    else int(batch['slice_idx'][0]))
            key = (vid, sidx)
            if key not in targets:
                continue
            recon = batch['recon'].to(device)
            forward_out = module(recon)
            lower, upper = compute_interval(args.group, forward_out, recon, q_hat)
            meta = targets[key]
            out_png = (args.output_dir_fig /
                       f'fail_rank{meta["rank"]:02d}_{args.group}_'
                       f'{sidx}.png')
            render_panel(
                slice_data={
                    'target': batch['target'],
                    'recon': batch['recon'],
                    'lesion_mask': batch['lesion_mask'],
                },
                lower=lower, upper=upper, q_hat=q_hat, alpha=alpha,
                out_png=out_png,
                meta={
                    'rank': meta['rank'], 'group': args.group,
                    'sequence': meta['sequence'], 'volume_id': vid,
                    'slice_idx': sidx,
                    'coverage_lesion': float(meta['coverage_lesion']),
                    'ulas_lesion': float(meta['ulas_lesion'])
                    if not (isinstance(meta['ulas_lesion'], float)
                            and math.isnan(meta['ulas_lesion'])) else float('nan'),
                },
            )
            found += 1
            if found >= len(targets):
                break
    logger.info(f'Renderizadas {found}/{len(targets)} figuras.')


def main() -> int:
    args = parse_args()

    # Resolve alpha: prioridade CLI > qhat json > DEFAULT_ALPHA.
    alpha = args.alpha
    if alpha is None and args.qhat is not None and Path(args.qhat).is_file():
        try:
            alpha = float(json.loads(Path(args.qhat).read_text()).get('alpha'))
        except (ValueError, TypeError):
            alpha = None
    if alpha is None:
        alpha = DEFAULT_ALPHA
    logger.info(f'alpha = {alpha} (cobertura nominal = {1 - alpha:.2f}); '
                f'criterio = {args.rank_by}; grupo = {args.group}')

    if not args.metrics_csv.is_file():
        logger.error(f'metrics CSV nao encontrado: {args.metrics_csv}')
        return 2

    df_sorted, df_les = load_ranked(args.metrics_csv, args.rank_by, alpha)
    logger.info(f'Slices com lesao: {len(df_les)}; '
                f'piores exportados: {min(args.n_worst, len(df_sorted))}')

    df_worst = attach_bbox_area(df_sorted.head(args.n_worst), args.brain_csv)
    out_csv = args.output_dir_csv / f'failure_top{args.n_worst}_group{args.group}.csv'
    worst_rows = write_worst_csv(df_worst, out_csv, args.n_worst)

    render_worst(worst_rows, args, alpha)
    logger.info('Analise de falha concluida.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
