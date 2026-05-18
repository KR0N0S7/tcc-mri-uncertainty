# Checkpoints — manifest de reprodutibilidade

Este arquivo registra a procedência e os hashes dos checkpoints de redes
pré-treinadas utilizadas no S4 (pré-computação de reconstruções E2E-VarNet),
bem como o dataset de outputs gerado a partir deles.

A integridade do `.pt` é o que sustenta a alegação de reprodutibilidade do TCC:
qualquer leitor que pretenda reproduzir os resultados deve poder verificar que
está usando o mesmo binário, e os outputs publicados são tratados como
"produto" auditável a partir desse binário.

---

## Modelo de entrada — `brain_leaderboard_state_dict.pt`

| Campo | Valor |
|---|---|
| **URL fonte** | https://dl.fbaipublicfiles.com/fastMRI/trained_models/varnet/brain_leaderboard_state_dict.pt |
| **Repositório origem** | facebookresearch/fastMRI (arquivado ago/2025, commit `91f2df47`) |
| **Tamanho** | 114,2 MB (119.747.864 bytes) |
| **SHA-256** | `2cdfdde2fe662f8995316a7ba981d8d3f405d2a5fd82a61bfa1bc4e56062262f` |
| **Parâmetros** | 29.936.966 (~30M) |
| **Data de download** | 13/05/2026 |

### Hiperparâmetros da arquitetura

Confirmados contra `fastmri_examples/varnet/run_pretrained_varnet_inference.py`
linha 63 do commit `91f2df47`. São os 5 args necessários para que
`load_state_dict(..., strict=True)` aceite o checkpoint sem missing nem
unexpected keys.

```python
VarNet(num_cascades=12, pools=4, chans=18, sens_pools=4, sens_chans=8)
```

### Configuração de inferência esperada

| Componente | Valor |
|---|---|
| Aceleração | 4× |
| Máscara | `EquispacedMaskFractionFunc(center_fractions=[0.08], accelerations=[4])` |
| Transform | `fastmri.data.transforms.VarNetDataTransform(mask_func=..., use_seed=True)` |
| Challenge | `multicoil` |
| Modalidades cobertas | AXFLAIR, AXT1, AXT1POST, AXT2 (cérebro) |

> **Sobre a classe da máscara:** a fastmri tem duas classes equispaced —
> `EquiSpacedMaskFunc` (legacy, com `round()`) e `EquispacedMaskFractionFunc`
> (atual, com `floor()`). O checkpoint brain 4× foi treinado com a versão
> Fraction; usar a legacy introduz 1-2 linhas de diferença no centro e
> degrada a SSIM em alguns pontos percentuais. Detalhes em `docs/S4.md`,
> seção 2.5.

### Referência

Sriram, A., Zbontar, J., Murrell, T., Defazio, A., Zitnick, C. L., Yakubova, N.,
Knoll, F., & Johnson, P. (2020). End-to-End Variational Networks for Accelerated
MRI Reconstruction. *MICCAI 2020*. https://arxiv.org/abs/2004.06688

### Como verificar a integridade

```powershell
python -c "from src.models import compute_sha256; print(compute_sha256('checkpoints/brain_leaderboard_state_dict.pt'))"
```

Saída esperada: `2cdfdde2fe662f8995316a7ba981d8d3f405d2a5fd82a61bfa1bc4e56062262f`

Se o hash diferir, **não use o arquivo** — refaça o download via
`python scripts/download_checkpoint.py`.

---

## Outputs gerados — `tcc-mri-recons-varnet-brain-4x`

Dataset Kaggle privado contendo as 352 reconstruções produzidas pelo S4. Cada
volume virou um `.npz` com recon, target, error_map e metadados — exatamente
o que o S5 precisa como input para treinar o quantile network.

| Campo | Valor |
|---|---|
| **Slug** | `massanorikishi/tcc-mri-recons-varnet-brain-4x` |
| **URL** | https://www.kaggle.com/datasets/massanorikishi/tcc-mri-recons-varnet-brain-4x |
| **Visibilidade** | Privado |
| **Tamanho total** | ~5,4 GB |
| **Arquivos** | 352 `.npz` + 1 `precompute_manifest.json` |
| **Data de criação** | 16/05/2026 |
| **Gerado por** | `scripts/precompute_reconstructions.py` rodado em Kaggle T4 |

### Conteúdo por split

| Pasta | Volumes | Tamanho |
|---|---|---|
| `train/` | 213 | 3,37 GB |
| `val/` | 46 | 695 MB |
| `cal/` | 46 | 737 MB |
| `test/` | 47 | 712 MB |
| `precompute_manifest.json` | — | 2,85 kB |

### Schema de cada `.npz`

| Campo | Tipo | Shape | Descrição |
|---|---|---|---|
| `recon` | float32 | (S, H, W) | Reconstrução VarNet, center-cropped |
| `target` | float32 | (S, H, W) | RSS da k-space fully-sampled |
| `error_map` | float32 | (S, H, W) | `\|target - recon\|` (magnitude pixel-wise) |
| `max_val` | float32 | () | `max(target_vol)`, para normalização no S5 |
| `volume_id` | str | () | Stem do `.h5` original |
| `split` | str | () | `train` \| `val` \| `cal` \| `test` |
| `acceleration` | int32 | () | 4 |
| `center_fraction` | float32 | () | 0.08 |
| `varnet_sha256` | str | () | Hash do checkpoint (campo acima) |

### Auditoria

Cada `.npz` carrega no campo `varnet_sha256` o hash do checkpoint que o gerou.
Para confirmar que um output foi produzido pelo binário registrado neste
manifest:

```python
import numpy as np
data = np.load('file_brain_AXFLAIR_200_6002460.npz')
assert str(data['varnet_sha256']) == '2cdfdde2fe662f8995316a7ba981d8d3f405d2a5fd82a61bfa1bc4e56062262f'
```

### Como consumir no S5 (referência rápida)

```python
from pathlib import Path
import numpy as np

# Em Kaggle: dataset montado como input
INPUT = Path('/kaggle/input/tcc-mri-recons-varnet-brain-4x')
# Em local: clone do dataset
# INPUT = Path('D:/Mri/recons')

for split in ['train', 'val', 'cal', 'test']:
    for npz_path in sorted((INPUT / split).glob('*.npz')):
        data = np.load(npz_path)
        recon, target, error_map = data['recon'], data['target'], data['error_map']
        max_val = float(data['max_val'])
        # ... treino do quantile network
```

### Referência do pipeline gerador

O pipeline ponta-a-ponta está documentado em [`docs/S4.md`](../docs/S4.md). O
manifest detalhado de cada run (timestamps, tempos por split, volumes
processados) fica embutido no próprio dataset como `precompute_manifest.json`.
