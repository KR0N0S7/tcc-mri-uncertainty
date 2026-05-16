# Checkpoints — manifest de reprodutibilidade

Este arquivo registra a procedência e os hashes dos checkpoints de redes
pré-treinadas utilizadas no S4 (pré-computação de reconstruções E2E-VarNet).
A integridade do `.pt` é o que sustenta a alegação de reprodutibilidade do TCC:
qualquer leitor que pretenda reproduzir os resultados deve poder verificar que
está usando o mesmo binário.

---

## `brain_leaderboard_state_dict.pt`

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
| Máscara | `EquispacedMaskFunc(center_fractions=[0.08], accelerations=[4])` |
| Transform | `fastmri.data.transforms.VarNetDataTransform()` |
| Challenge | `multicoil` |
| Modalidades cobertas | AXFLAIR, AXT1, AXT1POST, AXT2 (cérebro) |

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
