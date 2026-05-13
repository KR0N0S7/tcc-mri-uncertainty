# Autor: Massanori
# Data: 13/05/2026
# Descrição: Geração de máscaras binárias de lesão a partir dos bounding boxes
#            do fastMRI+ (Zhao et al., 2022). Implementa correção crítica de
#            orientação Y (y_fastmri = H - y_dicom - h) para alinhar bboxes
#            anotadas em DICOMs com reconstruction_rss armazenado no HDF5
#            (orientações diferem por flip vertical). Recebe: coordenadas de
#            bbox + shape da imagem, ou DataFrame de anotações, ou caminho do
#            HDF5 + brain_df completo. Retorna: tensor float32 binário (H, W)
#            por fatia ou (n_slices, H, W) por volume. Protegido por 7 testes
#            unitários em tests/test_lesion_masks.py.

"""
Geracao de mascaras binarias de lesoes a partir das anotacoes fastMRI+.

Converte os bounding boxes anotados por radiologistas no fastMRI+
(Zhao et al., 2022) em mascaras binarias 2D alinhadas com as
reconstrucoes RSS do fastMRI multicoil brain (Zbontar et al., 2020).

CORRECAO CRITICA DE ORIENTACAO:
O reconstruction_rss armazenado no HDF5 esta em orientacao espelhada
no eixo Y em relacao a convencao DICOM usada pelos radiologistas durante
a anotacao no MD.ai. O script fastmri-to-dicom.py do fastMRI+ aplica
essa reflexao antes da geracao dos DICOMs. Como aqui carregamos o
reconstruction_rss direto do HDF5 (sem passar pelo DICOM), aplicamos
a transformacao y_fastmri = H - y_dicom - h em bbox_to_mask para alinhar.

A imagem permanece em orientacao nativa fastMRI para preservar
compatibilidade com a VarNet pre-treinada (Sriram et al., 2020).

Validacao: figures/validacao_bbox/RESULTADO.md (10/10 OK pos-flip).

Refs:
- Zbontar, J. et al. (2020). fastMRI: An Open Dataset and Benchmarks
  for Accelerated MRI. arXiv:1811.08839
- Zhao, R. et al. (2022). fastMRI+: Clinical Pathology Annotations for
  Knee and Brain Fully Sampled Multi-Coil MRI Data. Scientific Data 9:152
- Sriram, A. et al. (2020). End-to-End Variational Networks for
  Accelerated MRI Reconstruction. MICCAI
"""
from __future__ import annotations
import logging
from pathlib import Path

import h5py
import pandas as pd
import torch

logger = logging.getLogger(__name__)


def bbox_to_mask(
    x: int,
    y: int,
    w: int,
    h: int,
    img_shape: tuple[int, int],
    apply_y_flip: bool = True,
) -> torch.Tensor:
    """
    Converte uma bbox (formato fastMRI+) em mascara binaria 2D alinhada
    com reconstruction_rss.

    Parameters
    ----------
    x, y : int
        Canto superior-esquerdo do bbox no sistema de coordenadas DICOM
        (origem no topo-esquerdo da imagem anotada).
    w, h : int
        Largura e altura do bbox em pixels.
    img_shape : tuple[int, int]
        Shape (H, W) da reconstruction_rss alvo.
    apply_y_flip : bool, default True
        Se True, transforma y_dicom em y_fastmri via H - y - h.
        Manter True para fastMRI brain. Parametro existe para permitir
        desativacao em testes unitarios e em datasets sem esse flip.

    Returns
    -------
    torch.Tensor
        Mascara float32 de shape (H, W) com valores em {0.0, 1.0}.
    """
    H, W = img_shape
    x, y, w, h = int(x), int(y), int(w), int(h)

    # Clip horizontal (sem flip no eixo x)
    x = max(0, min(x, W - 1))
    w_clipped = max(0, min(w, W - x))

    # Correcao de orientacao no eixo Y
    y_target = (H - y - h) if apply_y_flip else y
    y_target = max(0, y_target)
    h_clipped = max(0, min(h, H - y_target))

    mask = torch.zeros(img_shape, dtype=torch.float32)
    if w_clipped > 0 and h_clipped > 0:
        mask[y_target:y_target + h_clipped, x:x + w_clipped] = 1.0
    return mask


def slice_mask_from_annotations(
    annotations: pd.DataFrame,
    img_shape: tuple[int, int],
    apply_y_flip: bool = True,
) -> torch.Tensor:
    """
    Agrega TODAS as bboxes de uma fatia em uma unica mascara via uniao
    logica (OR). Fatias sem bbox retornam mascara zero.

    Parameters
    ----------
    annotations : pd.DataFrame
        Subset do brain.csv filtrado para um par (file, slice).
        Linhas com NaN em x/y/width/height (anotacoes study-level)
        sao ignoradas.
    img_shape : tuple[int, int]
    apply_y_flip : bool, default True
    """
    mask = torch.zeros(img_shape, dtype=torch.float32)
    if len(annotations) == 0:
        return mask

    valid = annotations.dropna(subset=['x', 'y', 'width', 'height'])
    for _, row in valid.iterrows():
        single = bbox_to_mask(
            x=row['x'], y=row['y'],
            w=row['width'], h=row['height'],
            img_shape=img_shape,
            apply_y_flip=apply_y_flip,
        )
        # Uniao logica via maximo (preserva binaridade da mascara)
        mask = torch.maximum(mask, single)
    return mask


def volume_masks_from_h5(
    h5_path: Path,
    brain_df: pd.DataFrame,
    apply_y_flip: bool = True,
) -> torch.Tensor:
    """
    Gera o tensor 3D de mascaras (n_slices, H, W) para um volume completo.

    Fatias sem anotacao recebem mascara zero. A imagem nao e modificada;
    apenas inspeciona-se o shape via reconstruction_rss.
    """
    volume_stem = h5_path.stem
    annotations = brain_df[brain_df['file'] == volume_stem]

    with h5py.File(h5_path, 'r') as hf:
        rss_shape = hf['reconstruction_rss'].shape  # (n_slices, H, W)
    n_slices, H, W = rss_shape

    masks = torch.zeros((n_slices, H, W), dtype=torch.float32)
    n_with_bbox = 0
    for sl in range(n_slices):
        slice_annots = annotations[annotations['slice'] == sl]
        if len(slice_annots.dropna(subset=['x', 'y', 'width', 'height'])) > 0:
            masks[sl] = slice_mask_from_annotations(
                slice_annots, img_shape=(H, W), apply_y_flip=apply_y_flip,
            )
            n_with_bbox += 1

    logger.info(f'{volume_stem}: {n_with_bbox}/{n_slices} fatias com lesao')
    return masks