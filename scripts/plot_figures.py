#!/usr/bin/env python3
# Autor: Massanori
# Data: 12/06/2026
# Descricao: Orquestrador de figuras de publicacao (item 2.4 do Bloco 2).
#            Regenera todas as figuras do TCC por um comando, comecando pela
#            hero figure: painel GT | recon | |erro| | incerteza A | B | C,
#            com bounding box por componente de lesao (scipy.ndimage.label) e
#            ESCALA DE INCERTEZA COMPARTILHADA entre os 3 grupos (comparacao
#            visual justa). A incerteza plotada e a halfwidth POS-calibracao
#            (upper_cal - lower_cal)/2 — a mesma definicao do notebooks/
#            demo.ipynb, para a figura do script e a do notebook coincidirem.
#            Recebe via CLI: --which {hero,stratified,failures,histogram,all},
#            checkpoints + q_hats dos 3 grupos (hero), e os args repassados
#            aos scripts ja commitados (stratified_analysis, failure_analysis,
#            lesion_area_histogram) quando --which != hero. Gera PNGs 300 DPI
#            em figures/. Sem narrativa hardcoded; o slice default e o de
#            MAIOR area de lesao no test (mais ilustrativo), com override por
#            --volume/--slice. Imports de torch/src sao lazy (a renderizacao
#            pura e testavel sem GPU).

"""Regenera as figuras do TCC. Comeca pela hero figure (item 2.4).

Hero (precisa dos 3 checkpoints + q_hats):
    python scripts/plot_figures.py --which hero \\
        --ckpt-a A/best.pt --ckpt-b B/best.pt --ckpt-c C/best.pt \\
        --qhat-a q_hat_A.json --qhat-b q_hat_B.json --qhat-c q_hat_C.json \\
        --recons-dir recons --masks-dir masks --figures-dir figures

Demais (repassa aos scripts ja commitados):
    python scripts/plot_figures.py --which stratified --csv-dir <dir>
    python scripts/plot_figures.py --which failures   --csv-dir <dir> \\
        --ckpt-c C/best.pt --qhat-c q_hat_C.json --recons-dir r --masks-dir m
    python scripts/plot_figures.py --which histogram
    python scripts/plot_figures.py --which all ...   (faz o que for possivel)

Refs:
    Romano et al. (2019), NeurIPS (intervalo CQR pos-calibracao).
    Tufte (2001), The Visual Display of Quantitative Information (escala
    compartilhada para comparacao honesta entre paineis).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

GROUPS = ('A', 'B', 'C')
GROUP_TITLE = {'A': 'Incerteza A (ResM)', 'B': 'Incerteza B (QR)',
               'C': 'Incerteza C (QR-Lesion)'}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Orquestrador de figuras do TCC (item 2.4).'
    )
    p.add_argument('--which', default='hero',
                   choices=['hero', 'stratified', 'failures', 'histogram', 'all'])
    # Hero: checkpoints + q_hats por grupo.
    p.add_argument('--ckpt-a', type=Path, default=None)
    p.add_argument('--ckpt-b', type=Path, default=None)
    p.add_argument('--ckpt-c', type=Path, default=None)
    p.add_argument('--qhat-a', type=Path, default=None)
    p.add_argument('--qhat-b', type=Path, default=None)
    p.add_argument('--qhat-c', type=Path, default=None)
    p.add_argument('--recons-dir', type=Path, default=None)
    p.add_argument('--masks-dir', type=Path, default=None)
    p.add_argument('--volume', default=None, help='volume_id p/ fixar o slice.')
    p.add_argument('--slice', type=int, default=None, help='slice_idx p/ fixar.')
    p.add_argument('--figures-dir', type=Path, default=ROOT / 'figures')
    p.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    p.add_argument('--chans', type=int, default=32)
    p.add_argument('--num-pool-layers', type=int, default=4)
    # Repasse aos scripts orquestrados.
    p.add_argument('--csv-dir', type=Path, default=None,
                   help='Dir com metrics_*.csv (stratified/failures).')
    p.add_argument('--brain-csv', type=Path, default=None)
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Helpers de figura (puros; testaveis sem torch)
# --------------------------------------------------------------------------- #
def bboxes_from_mask(mask2d: np.ndarray):
    """Lista de (x, y, w, h) por componente conexa. Espelha o demo.ipynb."""
    m = mask2d > 0.5
    if not m.any():
        return []
    try:
        from scipy import ndimage
        lab, n = ndimage.label(m)
        boxes = []
        for k in range(1, n + 1):
            ys, xs = np.where(lab == k)
            boxes.append((int(xs.min()), int(ys.min()),
                          int(xs.max() - xs.min() + 1),
                          int(ys.max() - ys.min() + 1)))
        return boxes
    except Exception:
        ys, xs = np.where(m)
        return [(int(xs.min()), int(ys.min()),
                 int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))]


def render_hero(gt: np.ndarray, recon: np.ndarray, error: np.ndarray,
                unc_maps: dict, mask2d: np.ndarray, out_png: Path,
                meta: dict):
    """Painel 1x6: GT | recon | |erro| | incerteza A | B | C.

    unc_maps: {'A': hw_A, 'B': hw_B, 'C': hw_C} (halfwidth pos-calibracao).
    A escala de cor das 3 colunas de incerteza e compartilhada (vmax = 99o
    percentil sobre os 3 mapas) para comparacao visual honesta. Bbox por
    componente de lesao em todos os paineis.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    boxes = bboxes_from_mask(mask2d)
    stacked = np.concatenate([unc_maps[g].ravel() for g in GROUPS])
    vmax_u = float(np.percentile(stacked, 99)) if stacked.size else 1.0
    if vmax_u <= 0:
        vmax_u = float(stacked.max()) if stacked.size else 1.0

    fig, axes = plt.subplots(1, 6, figsize=(24, 4.4))
    spec = [
        ('Ground truth', gt, 'gray', None, None),
        ('Reconstrucao (VarNet)', recon, 'gray', None, None),
        ('|Erro| = |GT - recon|', error, 'inferno', None, None),
        (GROUP_TITLE['A'], unc_maps['A'], 'inferno', 0.0, vmax_u),
        (GROUP_TITLE['B'], unc_maps['B'], 'inferno', 0.0, vmax_u),
        (GROUP_TITLE['C'], unc_maps['C'], 'inferno', 0.0, vmax_u),
    ]
    for ax, (title, img, cmap, vmin, vmax) in zip(axes, spec):
        im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        for (x, y, w, h) in boxes:
            ax.add_patch(Rectangle((x, y), w, h, fill=False,
                                   edgecolor='cyan', linewidth=1.2))
        ax.set_title(title, fontsize=12)
        ax.set_xticks([]); ax.set_yticks([])
        if cmap == 'inferno':
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f'Hero figure | {meta.get("sequence", "?")} | '
        f'{meta.get("volume_id", "?")} slice {meta.get("slice_idx", "?")} | '
        f'area lesao = {meta.get("lesion_area", "?")} px | '
        f'incerteza = halfwidth pos-calibracao (escala compartilhada A/B/C)',
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Hero figure: {out_png}')


# --------------------------------------------------------------------------- #
# Hero: carregamento de modelos e forward (lazy torch/src)
# --------------------------------------------------------------------------- #
def _resolve_device(arg: str) -> str:
    import torch
    if arg == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    if arg == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA solicitado mas indisponivel.')
    return arg


def _to2d(t) -> np.ndarray:
    return np.squeeze(t.detach().cpu().numpy())


def run_hero(args) -> int:
    import json
    import torch
    from torch.utils.data import DataLoader
    from src import config as cfg
    from src.calibration import apply_cqr_interval, apply_resm_interval
    from src.data import ReconsSliceDataset
    from src.models import QuantileRegressionModule, ResidualMagnitudeModule
    from src.training import load_checkpoint

    ckpts = {'A': args.ckpt_a, 'B': args.ckpt_b, 'C': args.ckpt_c}
    qhats_p = {'A': args.qhat_a, 'B': args.qhat_b, 'C': args.qhat_c}
    missing = [g for g in GROUPS if ckpts[g] is None or qhats_p[g] is None]
    if missing:
        print(f'ERRO: hero precisa de ckpt+qhat dos grupos {missing}.',
              file=sys.stderr)
        return 2

    device = _resolve_device(args.device)
    print(f'Device: {device}')

    qhats = {g: json.loads(Path(qhats_p[g]).read_text()) for g in GROUPS}
    modules = {}
    for g in GROUPS:
        mod = (ResidualMagnitudeModule(chans=args.chans,
                                       num_pool_layers=args.num_pool_layers)
               if g == 'A'
               else QuantileRegressionModule(chans=args.chans,
                                             num_pool_layers=args.num_pool_layers))
        load_checkpoint(ckpts[g], mod, device=device)
        modules[g] = mod.to(device).eval()

    recons_root = (args.recons_dir or cfg.recons_dir()).expanduser().resolve()
    test_dir = recons_root / 'test'
    masks_dir = (args.masks_dir or cfg.masks_dir()).expanduser().resolve()
    ds = ReconsSliceDataset(test_dir, masks_dir=masks_dir)

    # Seleciona o slice: override (--volume/--slice) ou maior area de lesao.
    sel = None
    if args.volume is not None and args.slice is not None:
        for i in range(len(ds)):
            s = ds[i]
            sid = (int(s['slice_idx'].item()) if torch.is_tensor(s['slice_idx'])
                   else int(s['slice_idx']))
            if s['volume_id'] == args.volume and sid == args.slice:
                sel = s
                break
        if sel is None:
            print(f'ERRO: ({args.volume}, {args.slice}) nao encontrado.',
                  file=sys.stderr)
            return 2
    else:
        best_area = -1
        for i in range(len(ds)):
            s = ds[i]
            area = int((s['lesion_mask'] > 0.5).sum().item())
            if area > best_area:
                best_area, sel = area, s
        if best_area <= 0:
            print('ERRO: nenhum slice com lesao no test.', file=sys.stderr)
            return 2

    recon = sel['recon'].unsqueeze(0).to(device) if sel['recon'].dim() == 3 \
        else sel['recon'].to(device)
    target = sel['target']
    lesion = sel['lesion_mask']

    def predict(g):
        q = float(qhats[g]['q_hat'])
        with torch.no_grad():
            out = modules[g](recon)
        if g == 'A':
            lo, hi = apply_resm_interval(recon, out, q)
        else:
            lo, hi = apply_cqr_interval(out['lower'], out['upper'], q)
        return (hi - lo) / 2.0  # halfwidth pos-calibracao

    unc_maps = {g: _to2d(predict(g)) for g in GROUPS}
    gt = _to2d(target)
    rc = _to2d(recon)
    err = np.abs(gt - rc)
    mask2d = _to2d(lesion)

    sid = (int(sel['slice_idx'].item()) if torch.is_tensor(sel['slice_idx'])
           else int(sel['slice_idx']))
    out_png = args.figures_dir / 'hero_figure.png'
    render_hero(gt, rc, err, unc_maps, mask2d, out_png, meta={
        'sequence': sel['sequence'], 'volume_id': sel['volume_id'],
        'slice_idx': sid, 'lesion_area': int((mask2d > 0.5).sum()),
    })
    return 0


# --------------------------------------------------------------------------- #
# Orquestracao dos scripts ja commitados
# --------------------------------------------------------------------------- #
def _run(cmd: list) -> int:
    print('>>> ' + ' '.join(str(c) for c in cmd))
    return subprocess.call([str(c) for c in cmd])


def run_stratified(args) -> int:
    if args.csv_dir is None:
        print('AVISO: stratified pulado (sem --csv-dir).', file=sys.stderr)
        return 0
    return _run([sys.executable, ROOT / 'scripts' / 'stratified_analysis.py',
                 '--csv-dir', args.csv_dir])


def run_failures(args) -> int:
    if args.csv_dir is None:
        print('AVISO: failures pulado (sem --csv-dir).', file=sys.stderr)
        return 0
    cmd = [sys.executable, ROOT / 'scripts' / 'failure_analysis.py',
           '--metrics-csv', args.csv_dir / 'metrics_C.csv', '--group', 'C']
    if args.ckpt_c and args.qhat_c:
        cmd += ['--checkpoint', args.ckpt_c, '--qhat', args.qhat_c]
        if args.recons_dir:
            cmd += ['--recons-dir', args.recons_dir]
        if args.masks_dir:
            cmd += ['--masks-dir', args.masks_dir]
    if args.brain_csv:
        cmd += ['--brain-csv', args.brain_csv]
    return _run(cmd)


def run_histogram(args) -> int:
    return _run([sys.executable, ROOT / 'scripts' / 'lesion_area_histogram.py'])


def main() -> int:
    args = parse_args()
    if args.which == 'hero':
        return run_hero(args)
    if args.which == 'stratified':
        return run_stratified(args)
    if args.which == 'failures':
        return run_failures(args)
    if args.which == 'histogram':
        return run_histogram(args)
    # all: faz o que for possivel, sem abortar no primeiro que faltar arg.
    rc = 0
    if all([args.ckpt_a, args.ckpt_b, args.ckpt_c,
            args.qhat_a, args.qhat_b, args.qhat_c]):
        rc |= run_hero(args)
    else:
        print('AVISO: hero pulado (faltam checkpoints/q_hats).', file=sys.stderr)
    rc |= run_stratified(args)
    rc |= run_failures(args)
    rc |= run_histogram(args)
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
