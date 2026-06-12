# Autor: Massanori
# Data: 11/06/2026
# Descricao: Bloco de melhoria 3.3 — gerador do VISUALIZADOR HTML ESTATICO.
#            Le os 3 modulos calibrados (S5.7) + volumes pre-computados do split
#            test e, para alguns volumes representativos (por padrao 1 por
#            sequencia), renderiza por slice os paineis GT/reconstrucao/erro e
#            os 3 mapas de incerteza calibrados (ResM/QR/QR-Lesion), alem das
#            metricas do slice (Coverage, largura, IoU, ULAS), reusando
#            EXATAMENTE src/calibration, src/metrics e a mesma logica do
#            notebooks/demo.ipynb. Emite um unico index.html AUTOCONTIDO
#            (PNGs embutidos em base64 + dados em JSON inline): abre em qualquer
#            navegador, offline, sem servidor nem dependencias — pensado para a
#            banca clicar e navegar (slider de slice + toggle A/B/C).
#            Nao roda inferencia no cliente; e um "resultado congelado",
#            auditavel via este gerador.
#            AVISO: o HTML gerado embute imagens DERIVADAS do fastMRI/fastMRI+.
#            Trate-o como o volume de amostra do 3.2 (Data Use Agreement);
#            so torne publico apos confirmar os termos. Este script (codigo)
#            e seguro para o repo publico; o index.html renderizado nao e
#            commitado por ele.

"""Gera um visualizador HTML estatico autocontido dos 3 grupos calibrados.

Exemplo (Kaggle, com os datasets anexados — mesma config do demo):

    python scripts/build_static_viewer.py \\
        --recons-test /kaggle/input/tcc-mri-recons-varnet-brain-4x/test \\
        --masks       /kaggle/input/tcc-mri-lesion-masks \\
        --resm        /kaggle/input/tcc-mri-resm-checkpoints/best.pt \\
        --qr          /kaggle/input/tcc-mri-qr-checkpoints/best.pt \\
        --qr-lesion   /kaggle/input/tcc-mri-qr-lesion-checkpoints/best.pt \\
        --qhats-dir   /kaggle/input/tcc-mri-conformal-qhats \\
        --out         /kaggle/working/viewer/index.html

Refs:
    Romano, Y.; Patterson, E.; Candes, E. (2019). Conformalized Quantile Regression. NeurIPS 32.
    Giannakopoulos, C. et al. (2026). arXiv:2601.13236.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # backend sem display (servidor/Kaggle)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.calibration import apply_cqr_interval, apply_resm_interval  # noqa: E402
from src.data import ReconsSliceDataset  # noqa: E402
from src.metrics.coverage import empirical_coverage, mean_interval_width  # noqa: E402
from src.metrics.iou import iou_topk  # noqa: E402
from src.metrics.ulas import ulas_with_null  # noqa: E402
from src.models import QuantileRegressionModule, ResidualMagnitudeModule  # noqa: E402

GROUPS = ("A", "B", "C")
GROUP_LABEL = {"A": "ResM (A)", "B": "QR (B)", "C": "QR-Lesion (C)"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gera visualizador HTML estatico (3.3).")
    p.add_argument("--recons-test", type=Path, required=True)
    p.add_argument("--masks", type=Path, required=True)
    p.add_argument("--resm", type=Path, required=True)
    p.add_argument("--qr", type=Path, required=True)
    p.add_argument("--qr-lesion", type=Path, required=True)
    p.add_argument("--qhats-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("viewer/index.html"))
    p.add_argument("--chans", type=int, default=32)
    p.add_argument("--num-pool-layers", type=int, default=4)
    p.add_argument("--alpha", type=float, default=0.10)
    p.add_argument("--top-pct", type=float, default=0.05)
    p.add_argument("--n-perms-ulas", type=int, default=10)
    p.add_argument("--one-per-sequence", action="store_true", default=True,
                   help="Seleciona 1 volume por sequencia (padrao).")
    p.add_argument("--all-volumes", dest="one_per_sequence", action="store_false",
                   help="Usa todos os volumes (HTML pode ficar grande).")
    p.add_argument("--max-volumes", type=int, default=3)
    p.add_argument("--max-slices", type=int, default=10,
                   help="Maximo de slices por volume (prioriza os com lesao).")
    p.add_argument("--panel-px", type=int, default=240, help="Lado do painel em px.")
    return p.parse_args()


def load_modules(args):
    def flexible(path, module):
        obj = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(obj, dict) and "model_state_dict" in obj:
            sd = obj["model_state_dict"]
        elif isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
            sd = obj["model"]
        else:
            sd = obj
        module.load_state_dict(sd)
        return module.eval()

    paths = {"A": args.resm, "B": args.qr, "C": args.qr_lesion}
    modules, qhats = {}, {}
    for g in GROUPS:
        mod = (ResidualMagnitudeModule(chans=args.chans, num_pool_layers=args.num_pool_layers)
               if g == "A"
               else QuantileRegressionModule(chans=args.chans, num_pool_layers=args.num_pool_layers))
        modules[g] = flexible(paths[g], mod)
        qhats[g] = json.loads((args.qhats_dir / f"q_hat_{g}.json").read_text())
    return modules, qhats


def predict(modules, qhats, group, recon):
    """Mesma logica do demo/compute_metrics: u_log p/ metricas, halfwidth cal p/ figura."""
    q = float(qhats[group]["q_hat"])
    with torch.no_grad():
        out = modules[group](recon)
    if group == "A":
        u = out
        lo, hi = apply_resm_interval(recon, u, q)
    else:
        lo, hi = apply_cqr_interval(out["lower"], out["upper"], q)
        u = (out["upper"] - out["lower"]) / 2.0
    hw = (hi - lo) / 2.0
    return lo.cpu(), hi.cpu(), u.cpu(), hw.cpu()


def bboxes_from_mask(mask2d):
    m = mask2d > 0.5
    if not m.any():
        return []
    try:
        from scipy import ndimage
        lab, n = ndimage.label(m)
        out = []
        for k in range(1, n + 1):
            ys, xs = np.where(lab == k)
            out.append([int(xs.min()), int(ys.min()),
                        int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)])
        return out
    except Exception:
        ys, xs = np.where(m)
        return [[int(xs.min()), int(ys.min()),
                 int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]]


def panel_png(img2d, cmap, boxes, panel_px, vmax=None):
    """Renderiza um painel quadrado sem eixos e devolve data-URI base64."""
    dpi = 80
    side = panel_px / dpi
    fig = plt.figure(figsize=(side, side), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img2d, cmap=cmap, vmax=vmax)
    ax.axis("off")
    for (x, y, w, h) in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, linewidth=1.4,
                                   edgecolor="lime", facecolor="none"))
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def select_volumes(ds, args):
    """Agrupa o index por volume e seleciona os representativos."""
    by_vol = defaultdict(list)  # stem -> [pos no dataset]
    for pos, (npz, _s) in enumerate(ds.index):
        by_vol[npz.stem].append(pos)

    # sequencia + area total de lesao por volume (para escolher os ilustrativos)
    info = {}
    for stem, positions in by_vol.items():
        seq = ds[positions[0]]["sequence"]
        area = sum(int((ds[p]["lesion_mask"] > 0.5).sum().item()) for p in positions)
        info[stem] = {"sequence": seq, "lesion_area": area, "positions": positions}

    if args.one_per_sequence:
        best_by_seq = {}
        for stem, meta in info.items():
            cur = best_by_seq.get(meta["sequence"])
            if cur is None or meta["lesion_area"] > info[cur]["lesion_area"]:
                best_by_seq[meta["sequence"]] = stem
        chosen = list(best_by_seq.values())
    else:
        chosen = sorted(info, key=lambda s: -info[s]["lesion_area"])

    chosen = sorted(chosen, key=lambda s: -info[s]["lesion_area"])[:args.max_volumes]
    return chosen, info


def to2d(t):
    return t.squeeze().detach().cpu().numpy()


def build_data(ds, modules, qhats, args):
    chosen, info = select_volumes(ds, args)
    volumes = []
    for stem in chosen:
        positions = info[stem]["positions"]
        # prioriza slices com lesao, mantem ordem por slice_idx, limita a max_slices
        with_area = [(p, int((ds[p]["lesion_mask"] > 0.5).sum().item())) for p in positions]
        lesion_first = sorted(with_area, key=lambda pa: (-pa[1], ds[pa[0]]["slice_idx"]))
        keep = sorted([p for p, _ in lesion_first[:args.max_slices]],
                      key=lambda p: ds[p]["slice_idx"])

        slices = []
        for p in keep:
            sample = ds[p]
            recon = sample["recon"].unsqueeze(0)
            target = sample["target"].unsqueeze(0)
            error = sample["error_map"].unsqueeze(0)
            lesion = sample["lesion_mask"].unsqueeze(0)
            mask2d = to2d(lesion)
            boxes = bboxes_from_mask(mask2d)

            preds = {g: predict(modules, qhats, g, recon) for g in GROUPS}
            hw_maps = {g: to2d(preds[g][3]) for g in GROUPS}
            umax = max(float(np.percentile(hw_maps[g], 99.5)) for g in GROUPS)
            umax = umax if umax > 0 else None

            metrics = {}
            for g in GROUPS:
                lo, hi, u, _hw = preds[g]
                ur = ulas_with_null(u, error, lesion,
                                    n_permutations=args.n_perms_ulas, seed=42 + p)
                metrics[g] = {
                    "cov_g": empirical_coverage(lo, hi, target)["coverage"],
                    "cov_l": empirical_coverage(lo, hi, target, mask=lesion)["coverage"],
                    "w_g": mean_interval_width(lo, hi),
                    "w_l": mean_interval_width(lo, hi, mask=lesion),
                    "iou_l": iou_topk(u, error, top_pct=args.top_pct, restrict_mask=lesion),
                    "ulas": ur["ulas"], "ulas_z": ur["z_score"],
                }
                metrics[g] = {k: (None if (isinstance(v, float) and v != v) else round(float(v), 4))
                              for k, v in metrics[g].items()}

            imgs = {
                "gt": panel_png(to2d(target), "gray", boxes, args.panel_px),
                "recon": panel_png(to2d(recon), "gray", boxes, args.panel_px),
                "error": panel_png(to2d(error), "inferno", boxes, args.panel_px),
            }
            for g in GROUPS:
                imgs[g] = panel_png(hw_maps[g], "inferno", boxes, args.panel_px, vmax=umax)

            slices.append({
                "slice_idx": int(sample["slice_idx"]),
                "lesion_px": int((mask2d > 0.5).sum()),
                "metrics": metrics, "img": imgs,
            })

        volumes.append({"id": stem, "sequence": ds[positions[0]]["sequence"], "slices": slices})
        print(f"  volume {stem} ({volumes[-1]['sequence']}): {len(slices)} slices")

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "alpha": args.alpha, "top_pct": args.top_pct,
        "group_label": GROUP_LABEL,
        "qhats": {g: round(float(qhats[g]["q_hat"]), 6) for g in GROUPS},
        "volumes": volumes,
    }


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="pt-br">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Visualizador — Incerteza por Pixel em MRI Acelerada</title>
<style>
  :root { --bg:#0f1419; --fg:#e6edf3; --muted:#8b949e; --card:#161b22; --acc:#2f81f7; --line:#30363d; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--fg);
         font-family:system-ui,Segoe UI,Roboto,Arial,sans-serif; }
  header { padding:18px 22px; border-bottom:1px solid var(--line); }
  h1 { font-size:18px; margin:0 0 4px; }
  .sub { color:var(--muted); font-size:13px; }
  main { padding:18px 22px; max-width:1100px; margin:0 auto; }
  .controls { display:flex; flex-wrap:wrap; gap:18px; align-items:center;
              background:var(--card); border:1px solid var(--line);
              border-radius:10px; padding:14px 16px; margin-bottom:16px; }
  .controls label { font-size:13px; color:var(--muted); display:block; margin-bottom:6px; }
  select, input[type=range] { width:240px; }
  .toggle button { background:var(--card); color:var(--fg); border:1px solid var(--line);
                   padding:7px 12px; border-radius:8px; cursor:pointer; font-size:13px; }
  .toggle button.active { background:var(--acc); border-color:var(--acc); color:#fff; }
  .panels { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; }
  .panel { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:8px; }
  .panel h3 { margin:0 0 6px; font-size:12px; color:var(--muted); font-weight:600; text-align:center; }
  .panel img { width:100%; display:block; border-radius:6px; image-rendering:pixelated; background:#000; }
  table { width:100%; border-collapse:collapse; margin-top:16px; font-size:13px; }
  th, td { border-bottom:1px solid var(--line); padding:7px 10px; text-align:right; }
  th:first-child, td:first-child { text-align:left; color:var(--muted); }
  caption { text-align:left; color:var(--muted); font-size:12px; margin-bottom:6px; }
  .meta { font-size:13px; color:var(--muted); margin:10px 0 0; }
  footer { color:var(--muted); font-size:12px; padding:18px 22px; border-top:1px solid var(--line); }
  code { background:#0b0f14; padding:1px 5px; border-radius:4px; }
</style>
</head>
<body>
<header>
  <h1>Quantificacao de Incerteza por Pixel — Reconstrucao Acelerada de MRI (4x)</h1>
  <div class="sub">Resultado congelado dos 3 grupos calibrados (A=ResM, B=QR, C=QR-Lesion). Navegue por volume/slice e alterne o metodo.</div>
</header>
<main>
  <div class="controls">
    <div>
      <label for="vol">Volume</label>
      <select id="vol"></select>
    </div>
    <div>
      <label for="sl">Slice (<span id="slval"></span>)</label>
      <input type="range" id="sl" min="0" value="0">
    </div>
    <div>
      <label>Mapa de incerteza</label>
      <div class="toggle" id="tg">
        <button data-g="A" class="active">ResM (A)</button>
        <button data-g="B">QR (B)</button>
        <button data-g="C">QR-Lesion (C)</button>
      </div>
    </div>
  </div>

  <div class="panels">
    <div class="panel"><h3>Ground Truth</h3><img id="img-gt"></div>
    <div class="panel"><h3>Reconstrucao 4x</h3><img id="img-recon"></div>
    <div class="panel"><h3>Erro |y - x|</h3><img id="img-error"></div>
    <div class="panel"><h3 id="ttl-unc">Incerteza</h3><img id="img-unc"></div>
  </div>
  <p class="meta" id="meta"></p>

  <table>
    <caption id="cap"></caption>
    <thead><tr><th>Metrica</th><th>ResM (A)</th><th>QR (B)</th><th>QR-Lesion (C)</th></tr></thead>
    <tbody id="tb"></tbody>
  </table>
</main>
<footer id="ft"></footer>

<script>
const DATA = /*__DATA__*/;
const GROUPS = ["A","B","C"];
const METRICS = [
  ["Coverage global","cov_g"], ["Coverage lesao","cov_l"],
  ["Largura media global","w_g"], ["Largura media lesao","w_l"],
  ["IoU top-5% lesao","iou_l"], ["ULAS lesao","ulas"], ["ULAS z-score","ulas_z"],
];
let curVol = 0, curSlice = 0, curG = "A";

const $ = (id) => document.getElementById(id);

function fmt(v){ return v === null || v === undefined ? "nan" : (typeof v === "number" ? v.toFixed(3) : v); }

function initVols(){
  const sel = $("vol");
  DATA.volumes.forEach((v,i) => {
    const o = document.createElement("option");
    o.value = i; o.textContent = v.id + "  (" + v.sequence + ")";
    sel.appendChild(o);
  });
  sel.onchange = () => { curVol = +sel.value; curSlice = 0; setupSlice(); render(); };
}

function setupSlice(){
  const n = DATA.volumes[curVol].slices.length;
  const sl = $("sl");
  sl.max = n - 1; sl.value = Math.min(curSlice, n - 1);
  sl.oninput = () => { curSlice = +sl.value; render(); };
}

function initToggle(){
  $("tg").querySelectorAll("button").forEach(b => {
    b.onclick = () => {
      curG = b.dataset.g;
      $("tg").querySelectorAll("button").forEach(x => x.classList.remove("active"));
      b.classList.add("active");
      render();
    };
  });
}

function render(){
  const vol = DATA.volumes[curVol];
  const s = vol.slices[curSlice];
  $("slval").textContent = s.slice_idx;
  $("img-gt").src = s.img.gt;
  $("img-recon").src = s.img.recon;
  $("img-error").src = s.img.error;
  $("img-unc").src = s.img[curG];
  $("ttl-unc").textContent = "Incerteza — " + DATA.group_label[curG];

  $("meta").innerHTML = "Volume <code>" + vol.id + "</code> | slice " + s.slice_idx +
    " | sequencia " + vol.sequence + " | pixels de lesao " + s.lesion_px +
    " | cobertura nominal " + Math.round((1 - DATA.alpha) * 100) + "%" +
    " | q_hat A/B/C = " + DATA.qhats.A + " / " + DATA.qhats.B + " / " + DATA.qhats.C;

  $("cap").textContent = "Metricas do slice (incerteza pre-calibracao para IoU/ULAS, igual a compute_metrics.py).";
  const tb = $("tb"); tb.innerHTML = "";
  for (const [name, key] of METRICS){
    const tr = document.createElement("tr");
    let row = "<th>" + name + "</th>";
    for (const g of GROUPS){
      const v = s.metrics[g][key];
      row += "<td>" + (key === "ulas_z" ? (v===null?"nan":(+v).toFixed(2)) : fmt(v)) + "</td>";
    }
    tr.innerHTML = row; tb.appendChild(tr);
  }
}

$("ft").innerHTML = "Gerado em " + DATA.created_at +
  ". Resultado congelado; pipeline e codigo em github.com/KR0N0S7/tcc-mri-uncertainty. " +
  "Imagens derivadas do fastMRI/fastMRI+ (sujeitas ao Data Use Agreement).";

initVols(); initToggle(); setupSlice(); render();
</script>
</body>
</html>
"""


def main() -> int:
    args = parse_args()
    print("Carregando modulos + q_hats...")
    modules, qhats = load_modules(args)
    print("Montando dataset do split test...")
    ds = ReconsSliceDataset(args.recons_test, masks_dir=args.masks)
    print(f"Split test: {len(ds)} slices. Selecionando volumes...")
    data = build_data(ds, modules, qhats, args)

    if not data["volumes"]:
        print("ERRO: nenhum volume selecionado (verifique recons/masks).")
        return 2

    html = HTML_TEMPLATE.replace("/*__DATA__*/", json.dumps(data, ensure_ascii=False))
    out = args.out.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    size_mb = out.stat().st_size / (1024 * 1024)
    n_slices = sum(len(v["slices"]) for v in data["volumes"])
    print(f"\nOK -> {out}  ({size_mb:.2f} MB | {len(data['volumes'])} volumes | {n_slices} slices)")
    print("AVISO: o index.html embute imagens derivadas do fastMRI. Confirme o Data Use "
          "Agreement antes de publicar (ex.: GitHub Pages). Na duvida, compartilhe o "
          "arquivo direto com a banca.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
