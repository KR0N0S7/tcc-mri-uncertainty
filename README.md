# tcc-mri-uncertainty

Pixel-wise uncertainty quantification for accelerated brain MRI reconstruction using conformalized quantile regression with lesion-aware loss weighting. Built on E2E-VarNet and fastMRI. — Data Science \& Analytics





\## Dependências externas (não versionadas)



Antes de rodar os scripts, clone o repositório oficial do fastMRI+ em `data/`:



\\`\\`\\`bash

cd data/

git clone https://github.com/microsoft/fastmri-plus.git

\\`\\`\\`



O `brain.csv` em `data/fastmri-plus/Annotations/brain.csv` é referenciado pelos

scripts em `scripts/`. Ref: Zhao et al. (2022) \*Scientific Data\* 9:152.



Os dados brutos do fastMRI brain multicoil (.h5) devem ser baixados separadamente

de https://fastmri.med.nyu.edu/ após registro. Caminho esperado nos scripts:

o que você configurar (ex.: `D:\\Mri\\anotados\\`).


## Configuração do ambiente

O projeto usa variáveis de ambiente para caminhos de dados, permitindo o mesmo
código rodar em diferentes máquinas sem alteração.

### Setup local (Windows / Linux / Mac)

1. Copie o template:

   \`\`\`bash
   cp .env.example .env
   \`\`\`

2. Edite `.env` com seus caminhos reais. As variáveis obrigatórias são:
   - `TCC_ANOTADOS_DIR`: pasta com volumes `.h5` filtrados (com bbox slice-level)
   - `TCC_BRAIN_CSV`: caminho para `brain.csv` do fastMRI+

3. Instale `python-dotenv`:

   \`\`\`bash
   pip install python-dotenv
   \`\`\`

4. Confira a configuração:

   \`\`\`bash
   python -m src.config
   \`\`\`

   Deve imprimir todos os caminhos resolvidos.

### Setup em Kaggle

No primeiro cell do notebook, antes de qualquer import dos scripts:

\`\`\`python
import os
os.environ['TCC_ANOTADOS_DIR'] = '/kaggle/input/anotados-352'
os.environ['TCC_BRAIN_CSV']    = '/kaggle/input/fastmri-plus/brain.csv'
\`\`\`

### Override por linha de comando

Todos os scripts aceitam `--anotados` e `--brain-csv` como argumentos opcionais
que sobrescrevem as variáveis de ambiente, útil para rodadas pontuais:

\`\`\`bash
python scripts/generate_lesion_masks.py --anotados /outro/caminho --brain-csv ./brain.csv --out ./masks
\`\`\`


### ⚠ Nota sobre Windows + PowerShell

Ao criar o arquivo `.env` no PowerShell, **evite usar `echo ... > .env`** —
o PowerShell salva em UTF-16 LE com BOM, que o python-dotenv não consegue ler
(`UnicodeDecodeError: byte 0xff in position 0`).

Use uma destas alternativas:

\`\`\`powershell
# Opção 1: Set-Content com encoding explícito
@"
TCC_ANOTADOS_DIR=D:\Mri\anotados
TCC_BRAIN_CSV=D:\Mri\brain.csv
"@ | Set-Content -Path .env -Encoding UTF8
\`\`\`

\`\`\`powershell
# Opção 2: editar no VS Code (salva como UTF-8 por padrão)
code .env
\`\`\`

Para conferir o encoding: `Get-Content .env -Encoding Byte -TotalCount 5`
deve mostrar bytes ASCII (84, 67, 67...), não 255, 254 (BOM UTF-16).

