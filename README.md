# tcc-mri-uncertainty

Pixel-wise uncertainty quantification for accelerated brain MRI reconstruction using conformalized quantile regression with lesion-aware loss weighting. Built on E2E-VarNet and fastMRI+. — Trabalho de Conclusão de Curso, Data Science & Analytics, USP-ESALQ/PECEGE.

## Status do projeto

| Estágio | Descrição | Status |
|---|---|---|
| **S1** | Setup e infra (paths via env vars, deps, testes base) | ✅ Concluído |
| **S2** | Download e filtragem dos 352 volumes elegíveis | ✅ Concluído |
| **S3** | Pré-processamento (máscaras de lesão, splits estratificados, normalização) | ✅ Concluído |
| **S4** | Pré-computação das reconstruções E2E-VarNet brain 4× | ✅ Concluído ([S4.md](docs/S4.md)) |
| **S5** | Treino + CQR + análise estatística (3 grupos: ResM, QR, QR-Lesion) | ✅ Concluído ([S5.md](docs/S5.md)) |
| | Tag da 4ª entrega: [`4a-entrega`](https://github.com/KR0N0S7/tcc-mri-uncertainty/releases/tag/4a-entrega) | ✅ Aplicada |
| **S6** | Escrita final do TCC e defesa (5ª entrega) | ⏳ Em andamento |

## Principais achados (S5, 4ª entrega)

Sobre o split test (47 volumes, 736 slices, 362 com lesão), comparando três métodos de UQ:

- **A — ResM** (Residual Magnitude, baseline locally-adaptive)
- **B — QR** (Conformalized Quantile Regression, replicação de Giannakopoulos et al. 2026)
- **C — QR-Lesion** (CQR + loss ponderada com λ=5, **contribuição original** deste trabalho)

Análise estatística: Friedman + Wilcoxon signed-rank par-a-par com correção Holm-Bonferroni, BCa bootstrap (10 000 resamples), Clopper-Pearson para Coverage.

### Hipóteses pré-registradas

| # | Hipótese | p_Holm | Resultado |
|---|---|---|---|
| H1 | C > B em Coverage_lesion | < 0,001 | **CONFIRMADA** |
| H2 | C > B em Width_lesion | < 0,001 | **CONFIRMADA** |
| H3 | C > B em ULAS_lesion | < 0,001 | **CONFIRMADA** |
| H4 | C > B em IoU_topk_lesion | 0,132 | NÃO detectada |

**3 de 4 hipóteses confirmadas** com tamanho de efeito modesto mas estatisticamente robusto (n=362 pares de slices com lesão).

### Achados adicionais

- **Descoberta inesperada de grande magnitude**: Grupo A (ResM, locally-adaptive) supera B e C em Coverage_lesion por ~8 pontos percentuais (0,869 vs 0,784 / 0,786), todos com Wilcoxon p_Holm < 0,001. Implicação: métodos com calibração multiplicativa adaptativa por pixel são estruturalmente superiores ao CQR marginal para cobertura de regiões pequenas e clinicamente relevantes.

- **Métrica original ULAS é discriminativa**: o Uncertainty-Lesion Alignment Score (proposto neste trabalho) detecta diferenças significativas entre todos os três pares A/B/C em ULAS_lesion (todos p_Holm < 0,001), captando informação direcional que o Pearson global não captura.

Detalhes completos, mecânica e discussão das limitações em **[docs/S5.md](docs/S5.md)**.

## Estrutura do código

```
.
├── src/                              # Bibliotecas reutilizáveis
│   ├── config.py                     # Resolução de paths via env vars
│   ├── data/
│   │   ├── normalization.py          # Max-volume (aplicada em runtime)
│   │   ├── lesion_masks.py           # Bbox slice-level + flip Y DICOM↔fastMRI
│   │   └── kspace_dataset.py         # Builder p/ inferência VarNet
│   ├── models/
│   │   ├── varnet_loader.py          # Loader determinístico c/ SHA-256
│   │   ├── resm.py                   # Grupo A — Residual Magnitude head
│   │   ├── quantile_regression.py    # Grupo B/C — Quantile Regression
│   │   └── quantile_lesion.py        # Grupo C — alias com loss ponderada
│   ├── calibration/
│   │   └── conformal.py              # CQR + scaled CP (q_hat em CPU c/ fallback)
│   └── metrics/
│       ├── iou.py                    # IoU top-X% (global e por região)
│       └── ulas.py                   # ULAS (contribuição original)
├── scripts/                          # Pipelines de ponta-a-ponta
│   ├── make_splits.py                # Splits estratificados por sequência
│   ├── generate_lesion_masks.py      # Geração das máscaras Y-flipped
│   ├── download_checkpoint.py        # Fetch do brain_leaderboard.pt
│   ├── precompute_reconstructions.py # Pipeline batch resumível (S4)
│   ├── train.py                      # Treino do quantile network (S5.2-5.4)
│   ├── calibrate.py                  # Calibração conforme (S5.7)
│   ├── compute_metrics.py            # Métricas por slice (S5.8)
│   ├── analyze_S5_9.py               # Análise estatística + docs/S5.md (S5.9)
│   └── print_metrics_table.py        # Standalone p/ regenerar tabela do S5.8
├── notebooks/                        # Runners Kaggle T4
│   ├── kaggle_precompute.ipynb       # S4
│   ├── kaggle_train_{resm,qr,qr_lesion}.ipynb  # S5.2-5.4
│   ├── kaggle_calibrate.ipynb        # S5.7
│   └── kaggle_metrics_S5_8.ipynb     # S5.8
├── splits/                           # Splits estratificados (S3)
│   ├── manifest.json
│   ├── train.txt val.txt cal.txt test.txt
├── configs/                          # JSONs de configuração
├── checkpoints/                      # MANIFEST.md (auditoria reprod.)
├── tests/                            # pytest — 174+ testes em CI
└── docs/                             # Relatórios técnicos por estágio
    ├── S4.md                         # Pré-computação VarNet
    ├── S5.md                         # 4ª entrega consolidada
    └── figures/
        └── s5_9_analysis.json        # Resultados estatísticos (auditoria)
```

## Documentação detalhada

- **[docs/S5.md](docs/S5.md)** — **documento canônico da 4ª entrega**: setup, resultados, análise estatística formal, discussão de achados, limitações, trabalho futuro, reprodutibilidade auditável.
- [docs/S4.md](docs/S4.md) — Pré-computação VarNet (decisões metodológicas, sanity check).
- [checkpoints/MANIFEST.md](checkpoints/MANIFEST.md) — SHA-256 e hiperparâmetros do backbone.

## Reprodutibilidade

### Versão de referência

Para reproduzir exatamente os resultados da 4ª entrega, use a tag:

```bash
git checkout 4a-entrega
```

### SHA-256 dos checkpoints treinados

| Grupo | Checkpoint SHA-256 (16 primeiros hex) |
|---|---|
| A (ResM) | `9ef2fa8e4e85706e...` |
| B (QR) | `fc185dedb3457b3c...` |
| C (QR-Lesion) | `fa5198f832acd58f...` |

SHA-256 completos em `q_hat_*.json` (dataset Kaggle `tcc-mri-conformal-qhats`).

### Datasets Kaggle (cadeia de dados auditável)

| Slug | Conteúdo | Tamanho |
|---|---|---|
| `tcc-mri-recons-varnet-brain-4x` | Reconstruções precomputadas E2E-VarNet | ~6 GB |
| `tcc-mri-lesion-masks` | Máscaras .pt do fastMRI+ | 3 MB |
| `tcc-mri-resm-checkpoints` | Grupo A: best.pt + metrics + config | ~518 MB |
| `tcc-mri-qr-checkpoints` | Grupo B: best.pt + metrics + config | ~342 MB |
| `tcc-mri-qr-lesion-checkpoints` | Grupo C: best.pt + metrics + config | ~342 MB |
| `tcc-mri-conformal-qhats` | 3 JSONs com q_hat + auditoria SHA | ~5 kB |
| `tcc-mri-s5-8-metrics` | 3 CSVs por slice + summaries | ~1 MB |

### Pipeline completo (5 etapas)

```bash
# 1. Clonar e instalar
git clone https://github.com/KR0N0S7/tcc-mri-uncertainty.git
cd tcc-mri-uncertainty
git checkout 4a-entrega
pip install -r requirements.txt

# 2. Validar setup
python -m pytest tests/ -v
# esperado: 174+ passed

# 3. S4 — Pré-computar reconstruções (rodar uma vez, ~6h em T4)
python scripts/precompute_reconstructions.py \
    --split val --output-dir <recons_root>

# 4. S5.2-5.4 — Treinar (uma vez por grupo, ~6h em T4 cada)
python scripts/train.py --group A --output-dir <ckpt_A>
python scripts/train.py --group B --output-dir <ckpt_B>
python scripts/train.py --group C --output-dir <ckpt_C>

# 5. S5.7-5.9 — Calibração, métricas e análise estatística
python scripts/calibrate.py --group A \
    --checkpoint <ckpt_A>/best.pt \
    --recons-dir <recons_root> \
    --output q_hat_A.json
# (repetir para B e C)

python scripts/compute_metrics.py --group A \
    --checkpoint <ckpt_A>/best.pt \
    --qhat q_hat_A.json \
    --recons-dir <recons_root> \
    --masks-dir <masks_root> \
    --output metrics_A.csv
# (repetir para B e C)

python scripts/analyze_S5_9.py \
    --csv-dir <dir_com_3_csvs> \
    --output-json docs/figures/s5_9_analysis.json \
    --output-md docs/S5.md
```

## Dependências externas (não versionadas)

Antes de rodar os scripts, clone o repositório oficial do fastMRI+ em `data/`:

```bash
cd data/
git clone https://github.com/microsoft/fastmri-plus.git
```

O `brain.csv` em `data/fastmri-plus/Annotations/brain.csv` é referenciado pelos scripts em `scripts/`. Ref: Zhao et al. (2022) *Scientific Data* 9:152.

Os dados brutos do fastMRI brain multicoil (.h5) devem ser baixados separadamente de [fastmri.med.nyu.edu](https://fastmri.med.nyu.edu/) após registro. Caminho esperado nos scripts: o que você configurar (ex.: `D:\Mri\anotados\`).

## Configuração do ambiente

O projeto usa variáveis de ambiente para caminhos de dados, permitindo o mesmo código rodar em diferentes máquinas sem alteração.

### Setup local (Windows / Linux / Mac)

1. Copie o template:

   ```bash
   cp .env.example .env
   ```

2. Edite `.env` com seus caminhos reais. As variáveis obrigatórias são:
   - `TCC_ANOTADOS_DIR`: pasta com volumes `.h5` filtrados (com bbox slice-level)
   - `TCC_BRAIN_CSV`: caminho para `brain.csv` do fastMRI+

3. Instale `python-dotenv`:

   ```bash
   pip install python-dotenv
   ```

4. Confira a configuração:

   ```bash
   python -m src.config
   ```

   Deve imprimir todos os caminhos resolvidos.

### Setup em Kaggle

No primeiro cell do notebook, antes de qualquer import dos scripts:

```python
import os
os.environ['TCC_ANOTADOS_DIR'] = '/kaggle/working/anotados'  # após criar symlinks dos datasets
os.environ['TCC_BRAIN_CSV']    = '/dev/null'                  # não usado em S4+
```

Ver `notebooks/kaggle_*.ipynb` para os setups completos por estágio.

### Override por linha de comando

Todos os scripts aceitam `--anotados` e `--brain-csv` como argumentos opcionais que sobrescrevem as variáveis de ambiente, útil para rodadas pontuais:

```bash
python scripts/generate_lesion_masks.py \
    --anotados /outro/caminho \
    --brain-csv ./brain.csv \
    --out ./masks
```

### ⚠ Nota sobre Windows + PowerShell

Ao criar o arquivo `.env` no PowerShell, **evite usar `echo ... > .env`** — o PowerShell salva em UTF-16 LE com BOM, que o python-dotenv não consegue ler (`UnicodeDecodeError: byte 0xff in position 0`).

Use uma destas alternativas:

```powershell
# Opção 1: Set-Content com encoding explícito
@"
TCC_ANOTADOS_DIR=D:\Mri\anotados
TCC_BRAIN_CSV=D:\Mri\brain.csv
"@ | Set-Content -Path .env -Encoding UTF8
```

```powershell
# Opção 2: editar no VS Code (salva como UTF-8 por padrão)
code .env
```

Para conferir o encoding: `Get-Content .env -Encoding Byte -TotalCount 5` deve mostrar bytes ASCII (84, 67, 67...), não 255, 254 (BOM UTF-16).

## Testes

```bash
python -m pytest tests/ -v
```

Esperado: 174+ testes verdes. Roda em CPU, sem precisar de GPU ou dados reais. A suite cobre:

- `tests/test_normalization.py` — max-volume e exchangeability
- `tests/test_lesion_masks.py` — Y-flip DICOM↔fastMRI, bbox slice-level
- `tests/test_kspace_dataset.py` — builder de dataloader
- `tests/test_varnet_loader.py` — SHA-256 + carregamento determinístico
- `tests/test_resm.py`, `test_quantile_regression.py`, `test_quantile_lesion.py` — heads e losses
- `tests/test_conformal.py` — calibração + quantile fallback `numpy.partition`
- `tests/test_iou.py` — IoU top-X% (12 testes)
- `tests/test_ulas.py` — ULAS construct validity (11 testes, contribuição original)

## Versões de software

- Python 3.12
- PyTorch 2.10 (CUDA 12.8 para treino, CPU para calibração/métricas)
- scipy ≥ 1.11 (para BCa bootstrap em `scipy.stats.bootstrap`)
- pandas ≥ 2.0, numpy ≥ 1.24
- pytest ≥ 8.0

Versões pinadas em `requirements.txt`.

## Hardware utilizado

- **Treino** (S5.2-5.4): Kaggle T4 (15.6 GB VRAM), ~6h por grupo.
- **Pré-computação VarNet** (S4): Kaggle T4, ~6h total.
- **Calibração e métricas** (S5.7-5.8): CPU (cota Kaggle GPU esgotada) com fallback `numpy.partition` em `src/calibration/conformal.py`.
- **Análise estatística** (S5.9): laptop local, ~3 min (bootstrap BCa domina).

## Referências principais

Lista completa em [docs/S5.md §7](docs/S5.md). Trabalhos-base:

- Romano, Y.; Patterson, E.; Candès, E. (2019). Conformalized Quantile Regression. *NeurIPS*, 32:3543-3553.
- Sriram, A. et al. (2020). End-to-End Variational Networks for Accelerated MRI Reconstruction. *MICCAI*, 64-73.
- Zbontar, J. et al. (2020). fastMRI: An Open Dataset and Benchmarks for Accelerated MRI. *arXiv:1811.08839*.
- Zhao, R. et al. (2022). fastMRI+: Clinical Pathology Annotations for Knee and Brain Fully Sampled MultiCoil MRI Data. *Scientific Data*, 9:152.
- Giannakopoulos, C. et al. (2026). Pixelwise Uncertainty Quantification of Accelerated MRI Reconstruction. *arXiv:2601.13236*.

## Licença

MIT License para o código deste repositório. Os datasets fastMRI e fastMRI+ têm suas próprias licenças — consulte os sites oficiais antes de uso comercial.
