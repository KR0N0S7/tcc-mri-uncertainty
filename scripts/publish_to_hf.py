# Autor: Massanori
# Data: 11/06/2026
# Descricao: Bloco de melhoria 3.2 — publicacao de ciencia aberta no HuggingFace
#            Hub. Recebe via CLI: os 3 best.pt do treino (S5), o diretorio dos
#            q_hat_*.json (S5.7) e, opcionalmente, 1 volume de amostra (.npz +
#            mascara .pt) para um demo publico. Para cada grupo, EXTRAI apenas
#            o model_state_dict (peso puro), DESCARTANDO optimizer/scheduler/
#            config_snapshot/rng — i.e. nunca sobe dados de paciente nem estado
#            de treino (politica de anonimizacao do repo + aviso do guia 6.4).
#            VALIDA por round-trip que o state_dict recarrega no modulo correto
#            com strict=True, computa o SHA-256 de cada arquivo publicado,
#            gera um MANIFEST.json + model card (README.md) auditavel pela
#            banca, e faz o upload para um repo HuggingFace (model). O volume
#            de amostra vai para um repo separado (dataset), PRIVADO por padrao,
#            com aviso explicito sobre o Data Use Agreement do fastMRI.
#            Token via env HF_TOKEN (Kaggle secret / variavel de ambiente),
#            nunca hardcoded. Use --dry-run para preparar o staging sem subir.

"""Publica os 3 modulos calibrados + q_hats no HuggingFace Hub (ciencia aberta).

So o `model_state_dict` e publicado — nunca o `best.pt` inteiro (que carrega
optimizer state e config). Isso atende ao aviso do guia (secao 6.4: "Salve
apenas model state_dict, nao batches de dados") e a politica de anonimizacao
do repositorio publico.

Exemplo (Kaggle, com os 3 datasets de checkpoint + qhats anexados):

    python scripts/publish_to_hf.py \\
        --resm       /kaggle/input/tcc-mri-resm-checkpoints/best.pt \\
        --qr         /kaggle/input/tcc-mri-qr-checkpoints/best.pt \\
        --qr-lesion  /kaggle/input/tcc-mri-qr-lesion-checkpoints/best.pt \\
        --qhats-dir  /kaggle/input/tcc-mri-conformal-qhats \\
        --repo-id    <usuario>/tcc-mri-uncertainty

    # (opcional) demo publico com 1 volume de amostra — leia o aviso de licenca:
    #   --sample-npz  /kaggle/input/tcc-mri-recons-varnet-brain-4x/test/<vol>.npz
    #   --sample-mask /kaggle/input/tcc-mri-lesion-masks/<vol>.pt
    #   --sample-repo-id <usuario>/tcc-mri-demo-sample   (PRIVADO por padrao)

Refs:
    Giannakopoulos, C. et al. (2026). arXiv:2601.13236.
    HuggingFace Hub: https://huggingface.co/docs/huggingface_hub
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models import QuantileRegressionModule, ResidualMagnitudeModule  # noqa: E402

# Grupo -> (subpasta no repo HF, classe do modulo)
GROUPS = {
    "A": ("resm", ResidualMagnitudeModule),
    "B": ("qr", QuantileRegressionModule),
    "C": ("qr_lesion", QuantileRegressionModule),
}
GROUP_LABEL = {"A": "ResM (baseline)", "B": "QR (CQR)", "C": "QR-Lesion (lambda=5)"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Publica modulos calibrados no HuggingFace.")
    p.add_argument("--resm", type=Path, required=True, help="best.pt do Grupo A.")
    p.add_argument("--qr", type=Path, required=True, help="best.pt do Grupo B.")
    p.add_argument("--qr-lesion", type=Path, required=True, help="best.pt do Grupo C.")
    p.add_argument("--qhats-dir", type=Path, required=True,
                   help="Dir com q_hat_A/B/C.json (S5.7).")
    p.add_argument("--repo-id", required=True,
                   help="Repo HF de modelo, ex.: usuario/tcc-mri-uncertainty.")
    p.add_argument("--chans", type=int, default=32)
    p.add_argument("--num-pool-layers", type=int, default=4)
    p.add_argument("--staging-dir", type=Path, default=Path("hf_staging"))
    p.add_argument("--private", action="store_true",
                   help="Cria o repo de modelo como privado.")
    # Volume de amostra (demo publico) — opcional
    p.add_argument("--sample-npz", type=Path, default=None)
    p.add_argument("--sample-mask", type=Path, default=None)
    p.add_argument("--sample-repo-id", default=None,
                   help="Repo HF (dataset) para o volume de amostra.")
    p.add_argument("--sample-public", action="store_true",
                   help="Torna o repo de amostra PUBLICO (cuidado: licenca fastMRI).")
    p.add_argument("--dry-run", action="store_true",
                   help="Prepara o staging e valida, sem subir nada.")
    return p.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_clean_state_dict(best_pt: Path, module: torch.nn.Module) -> dict:
    """Extrai SO o model_state_dict e valida que recarrega com strict=True.

    Descarta optimizer/scheduler/config/rng. Levanta erro se o checkpoint
    nao tiver pesos compativeis com o modulo (protege contra subir o grupo
    errado).
    """
    obj = torch.load(best_pt, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "model_state_dict" in obj:
        sd = obj["model_state_dict"]
    elif isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
        sd = obj["model"]
    elif isinstance(obj, dict) and all(torch.is_tensor(v) for v in obj.values()):
        sd = obj  # ja e state_dict puro
    else:
        raise ValueError(f"Formato de checkpoint nao reconhecido em {best_pt}")

    # Round-trip: strict=True garante que e o modulo certo (sem missing/unexpected).
    module.load_state_dict(sd, strict=True)
    # Re-extrai do modulo p/ garantir tensores limpos (sem refs ao optimizer).
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def build_model_card(repo_id: str, manifest: dict, sample_repo_id: str | None) -> str:
    rows = "\n".join(
        f"| {g} | {GROUP_LABEL[g]} | `{sub}/best_state_dict.pt` | "
        f"{manifest['groups'][g]['n_params']:,} | {manifest['groups'][g]['method']} | "
        f"{manifest['groups'][g]['q_hat']:.6f} | `{manifest['groups'][g]['sha256'][:12]}...` |"
        for g, (sub, _) in GROUPS.items()
    )
    sample_line = (
        f"\nVolume de amostra para o demo: [`{sample_repo_id}`](https://huggingface.co/datasets/{sample_repo_id})."
        if sample_repo_id else ""
    )
    return f"""---
license: mit
tags:
  - mri
  - uncertainty-quantification
  - conformal-prediction
  - medical-imaging
library_name: pytorch
---

# Quantificacao de Incerteza por Pixel em Reconstrucao Acelerada de MRI

Modulos de incerteza calibrados (conformal prediction) sobre reconstrucoes
E2E-VarNet de MRI cerebral acelerada 4x. Tres grupos comparados:

| Grupo | Metodo | Arquivo | Parametros | Calibracao | q_hat | SHA-256 |
|---|---|---|---|---|---|---|
{rows}

- **A (ResM)**: baseline de magnitude do residuo; calibracao Scaled-CP localmente adaptativa.
- **B (QR)**: replicacao de Giannakopoulos et al. (2026); CQR aditivo.
- **C (QR-Lesion)**: contribuicao original; pinball ponderada por lesao (lambda=5); CQR aditivo.

Cobertura nominal: **90%** (`alpha = 0.10`). Garantia de cobertura marginal
sob exchangeability (Romano, Patterson & Candes, 2019).

## Conteudo

```
resm/best_state_dict.pt        # Grupo A (state_dict puro)
qr/best_state_dict.pt          # Grupo B
qr_lesion/best_state_dict.pt   # Grupo C
qhats/q_hat_A.json             # q_hat calibrado (S5.7)
qhats/q_hat_B.json
qhats/q_hat_C.json
MANIFEST.json                  # SHAs + metadados para auditoria
```

> **Privacidade**: apenas `model_state_dict` (pesos). Nenhum dado de paciente,
> estado de optimizer ou batch foi publicado.

## Uso

```python
import torch
from huggingface_hub import hf_hub_download
from src.models import QuantileRegressionModule

sd = torch.load(hf_hub_download("{repo_id}", "qr_lesion/best_state_dict.pt"),
                map_location="cpu")
model = QuantileRegressionModule(chans=32, num_pool_layers=4)
model.load_state_dict(sd)  # strict=True
model.eval()
```

O codigo (`src/`), o pipeline e o demo clicavel estao em
[github.com/KR0N0S7/tcc-mri-uncertainty](https://github.com/KR0N0S7/tcc-mri-uncertainty)
(`notebooks/demo.ipynb`).{sample_line}

## Referencias

- Romano, Y.; Patterson, E.; Candes, E. (2019). Conformalized Quantile Regression. NeurIPS 32.
- Sriram, A. et al. (2020). End-to-End Variational Networks for Accelerated MRI Reconstruction. MICCAI 2020.
- Giannakopoulos, C. et al. (2026). Pixelwise Uncertainty Quantification of Accelerated MRI Reconstruction. arXiv:2601.13236.

Gerado em {manifest['created_at']}.
"""


def main() -> int:
    args = parse_args()
    best_paths = {"A": args.resm, "B": args.qr, "C": args.qr_lesion}

    staging = args.staging_dir.expanduser().resolve()
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    (staging / "qhats").mkdir()

    manifest = {
        "repo_id": args.repo_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "alpha": 0.10,
        "arch": {"chans": args.chans, "num_pool_layers": args.num_pool_layers},
        "groups": {},
    }

    print("=" * 64)
    print("EXTRACAO + VALIDACAO (state_dict puro, round-trip strict=True)")
    print("=" * 64)
    for g, (sub, cls) in GROUPS.items():
        best = best_paths[g].expanduser().resolve()
        if not best.is_file():
            print(f"ERRO: checkpoint do grupo {g} ausente: {best}")
            return 2
        module = cls(chans=args.chans, num_pool_layers=args.num_pool_layers)
        sd = extract_clean_state_dict(best, module)

        out_dir = staging / sub
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "best_state_dict.pt"
        torch.save(sd, out_file)

        # q_hat do grupo
        qsrc = (args.qhats_dir / f"q_hat_{g}.json").expanduser().resolve()
        if not qsrc.is_file():
            print(f"ERRO: q_hat do grupo {g} ausente: {qsrc}")
            return 2
        shutil.copy2(qsrc, staging / "qhats" / qsrc.name)
        qpayload = json.loads(qsrc.read_text())

        sha = sha256_file(out_file)
        manifest["groups"][g] = {
            "subdir": sub,
            "label": GROUP_LABEL[g],
            "n_params": sum(v.numel() for v in sd.values()),
            "method": qpayload.get("method", "?"),
            "q_hat": float(qpayload["q_hat"]),
            "sha256": sha,
            "source_best_pt_sha256": sha256_file(best),
        }
        print(f"  Grupo {g} ({sub}): {manifest['groups'][g]['n_params']:,} params | "
              f"q_hat={manifest['groups'][g]['q_hat']:.6f} | sha={sha[:12]}... OK")

    # MANIFEST + model card
    (staging / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (staging / "README.md").write_text(
        build_model_card(args.repo_id, manifest, args.sample_repo_id), encoding="utf-8"
    )
    print(f"\nStaging pronto em: {staging}")
    for p in sorted(staging.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(staging)}  ({p.stat().st_size/1024:.1f} kB)")

    # ----- Volume de amostra (opcional) -----
    sample_staging = None
    if args.sample_npz and args.sample_repo_id:
        sample_staging = staging.parent / "hf_sample"
        if sample_staging.exists():
            shutil.rmtree(sample_staging)
        (sample_staging / "test").mkdir(parents=True)
        (sample_staging / "masks").mkdir(parents=True)
        npz = args.sample_npz.expanduser().resolve()
        shutil.copy2(npz, sample_staging / "test" / npz.name)
        if args.sample_mask:
            mk = args.sample_mask.expanduser().resolve()
            shutil.copy2(mk, sample_staging / "masks" / mk.name)
        (sample_staging / "README.md").write_text(
            "---\nlicense: other\n---\n\n"
            "# Volume de amostra para o demo (TCC MRI)\n\n"
            "1 volume de teste pre-computado (recon/target/error_map) + mascara de lesao,\n"
            "para rodar `notebooks/demo.ipynb` publicamente.\n\n"
            "> AVISO DE LICENCA: estes arrays sao DERIVADOS dos dados fastMRI / fastMRI+.\n"
            "> A redistribuicao publica pode estar sujeita ao fastMRI Data Use Agreement.\n"
            "> Confirme os termos antes de tornar este repo publico; na duvida, mantenha-o\n"
            "> PRIVADO ou gated.\n",
            encoding="utf-8",
        )
        print(f"\nStaging de amostra pronto em: {sample_staging}")
        if args.sample_public:
            print("  ATENCAO: --sample-public ligado. Confirme o fastMRI Data Use "
                  "Agreement antes de expor dados derivados do fastMRI.")

    if args.dry_run:
        print("\n[DRY-RUN] Nada foi enviado. Revise o staging acima e rode sem --dry-run.")
        return 0

    # ----- Upload -----
    from huggingface_hub import HfApi
    api = HfApi()  # usa HF_TOKEN do ambiente / login previo

    print(f"\nPublicando modelo em: {args.repo_id} (private={args.private})")
    api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    api.upload_folder(folder_path=str(staging), repo_id=args.repo_id, repo_type="model",
                      commit_message="Publica state_dicts + q_hats + model card (bloco 3.2)")
    print(f"  OK: https://huggingface.co/{args.repo_id}")

    if sample_staging is not None:
        private = not args.sample_public
        print(f"\nPublicando amostra em: {args.sample_repo_id} (private={private})")
        api.create_repo(args.sample_repo_id, repo_type="dataset",
                        private=private, exist_ok=True)
        api.upload_folder(folder_path=str(sample_staging), repo_id=args.sample_repo_id,
                          repo_type="dataset",
                          commit_message="Volume de amostra para o demo (bloco 3.2)")
        print(f"  OK: https://huggingface.co/datasets/{args.sample_repo_id}")

    print("\nConcluido. Preencha HF_MODEL_REPO/HF_SAMPLE_REPO na Celula 2 do demo.ipynb.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
