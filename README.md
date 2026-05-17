# tcc-mri-uncertainty

Pixel-wise uncertainty quantification for accelerated brain MRI reconstruction using conformalized quantile regression with lesion-aware loss weighting. Built on E2E-VarNet and fastMRI. — Data Science & Analytics

## Status do projeto

| Estágio | Descrição | Status |
|---|---|---|
| **S1** | Setup e infra (paths via env vars, deps, testes base) | ✅ Concluído |
| **S2** | Download e filtragem dos 352 volumes elegíveis | ✅ Concluído |
| **S3** | Pré-processamento (máscaras de lesão, splits estratificados, normalização) | ✅ Concluído |
| **S4** | Pré-computação das reconstruções E2E-VarNet brain 4× | ✅ Concluído ([relatório](docs/S4.md)) |
| **S5** | Treino do quantile network + CQR (Conformal Quantile Regression) | ⏳ Pendente |
| **S6** | Avaliação de cobertura, análise por modalidade, escrita | ⏳ Pendente |

## Estrutura do código

```
.
├── src/                          # Bibliotecas reutilizáveis
│   ├── config.py                 # Resolução de paths via env vars
│   ├── data/
│   │   ├── normalization.py      # Max-volume (aplicada em runtime)
│   │   ├── lesion_masks.py       # Bbox slice-level + flip Y DICOM↔fastMRI
│   │   └── kspace_dataset.py     # Builder p/ inferência VarNet
│   └── models/
│       └── varnet_loader.py      # Loader determinístico c/ SHA-256
├── scripts/                      # Pipelines de ponta-a-ponta
│   ├── make_splits.py            # Splits estratificados por sequência
│   ├── generate_lesion_masks.py  # Geração das máscaras Y-flipped
│   ├── download_checkpoint.py    # Fetch do brain_leaderboard.pt
│   ├── sanity_check_varnet.py    # Gate dos 5 volumes do val (SSIM)
│   ├── diagnose_sanity_outlier.py# Análise slice-a-slice
│   └── precompute_reconstructions.py  # Pipeline batch resumível
├── notebooks/
│   └── kaggle_precompute.ipynb   # Runner Kaggle T4 para o S4
├── splits/                       # Splits estratificados (S3)
│   ├── manifest.json
│   ├── train.txt val.txt cal.txt test.txt
├── configs/                      # JSONs de configuração
├── checkpoints/                  # MANIFEST.md (auditoria reprod.)
├── tests/                        # pytest — 34+ testes em CI
└── docs/                         # Relatórios técnicos por estágio
    └── S4.md
```

## Documentação detalhada

- [Relatório do S4 (pré-computação VarNet)](docs/S4.md) — decisões metodológicas, resultados, como reproduzir
- [MANIFEST do checkpoint](checkpoints/MANIFEST.md) — SHA-256 e hiperparâmetros

## Dependências externas (não versionadas)

Antes de rodar os scripts, clone o repositório oficial do fastMRI+ em `data/`:

```bash
cd data/
git clone https://github.com/microsoft/fastmri-plus.git
```

O `brain.csv` em `data/fastmri-plus/Annotations/brain.csv` é referenciado pelos
scripts em `scripts/`. Ref: Zhao et al. (2022) *Scientific Data* 9:152.

Os dados brutos do fastMRI brain multicoil (.h5) devem ser baixados separadamente
de https://fastmri.med.nyu.edu/ após registro. Caminho esperado nos scripts:
o que você configurar (ex.: `D:\Mri\anotados\`).

## Configuração do ambiente

O projeto usa variáveis de ambiente para caminhos de dados, permitindo o mesmo
código rodar em diferentes máquinas sem alteração.

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
os.environ['TCC_ANOTADOS_DIR'] = '/kaggle/working/anotados'  # após criar symlinks dos 4 datasets
os.environ['TCC_BRAIN_CSV']    = '/dev/null'                  # não usado em S4+
```

Ver `notebooks/kaggle_precompute.ipynb` para o setup completo.

### Override por linha de comando

Todos os scripts aceitam `--anotados` e `--brain-csv` como argumentos opcionais
que sobrescrevem as variáveis de ambiente, útil para rodadas pontuais:

```bash
python scripts/generate_lesion_masks.py --anotados /outro/caminho --brain-csv ./brain.csv --out ./masks
```

### ⚠ Nota sobre Windows + PowerShell

Ao criar o arquivo `.env` no PowerShell, **evite usar `echo ... > .env`** —
o PowerShell salva em UTF-16 LE com BOM, que o python-dotenv não consegue ler
(`UnicodeDecodeError: byte 0xff in position 0`).

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

Para conferir o encoding: `Get-Content .env -Encoding Byte -TotalCount 5`
deve mostrar bytes ASCII (84, 67, 67...), não 255, 254 (BOM UTF-16).

## Testes

```bash
python -m pytest tests/ -v
```

Esperado: 34+ testes verdes. Roda em CPU, sem precisar de GPU ou dados reais.
