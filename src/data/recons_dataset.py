# Autor: Massanori
# Data: 17/05/2026 (mod 20/05/2026: suporte a mascaras salvas como dict)
# Descrição: Dataset PyTorch slice-wise sobre as reconstruções pré-computadas
#            no S4 (arquivos .npz gerados por scripts/precompute_reconstructions.py).
#            Recebe: diretório de um split (e.g. data/recons/train/) com .npz,
#            opcionalmente um masks_dir com {volume_id}.pt do S3 para o Grupo C.
#            Retorna: dict por fatia com tensores normalizados por max_val do
#            volume (D1) e máscara de lesão de zeros quando ausente (D3, interface
#            unificada). Implementa cache do último .npz e do último .pt abertos
#            para amortizar IO sob iteração sequencial do DataLoader (DataLoader
#            com shuffle=False percorre slices do mesmo volume em sequência).
#            Extrai o tipo de sequência (AXFLAIR/AXT1/AXT1POST) do volume_id
#            para análise estratificada no S5.8/S6. Protegido por testes em
#            tests/test_recons_dataset.py e tests/test_extract_mask.py.
# Mod 20/05/2026: _extract_mask_tensor() suporta os 3 formatos comuns de .pt
#                 do S3: (a) tensor direto, (b) dict com chave conhecida
#                 ('mask', 'masks', 'lesion_mask', 'mask_volume', ...),
#                 (c) dict com unico tensor entre os valores. Fix do bug
#                 'Could not infer dtype of dict' que bloqueava o Grupo C
#                 quando S3 salvava metadados junto da mascara.


"""Dataset PyTorch slice-wise sobre os .npz pre-computados no S4.

A reconstrucao da E2E-VarNet, o target RSS e o mapa de erro absoluto ja
foram calculados e congelados em disco no S4. Este Dataset apenas serve
as fatias 2D normalizadas para o modulo de incerteza (Grupos A/B/C do S5),
sem nunca tocar em k-space nem na VarNet — eliminando o custo dominante
de inferencia e isolando o efeito da loss como variavel independente.

Normalizacao por max_val do volume preserva exchangeability entre slices
do mesmo paciente (Romano, Patterson & Candes, 2019), pre-requisito para
a garantia de cobertura marginal do conformal prediction.

Refs:
    Romano, Y.; Patterson, E.; Candes, E. (2019). Conformalized Quantile
        Regression. NeurIPS 32, 3543-3553.
    Giannakopoulos, C. et al. (2026). Pixelwise Uncertainty Quantification
        of Accelerated MRI Reconstruction. arXiv:2601.13236.
    Sriram, A. et al. (2020). End-to-End Variational Networks for
        Accelerated MRI Reconstruction. MICCAI 2020 (LNCS 12262), 64-73.
    Zhao, R. et al. (2022). fastMRI+: Clinical pathology annotations
        for knee and brain fully sampled MRI data. Scientific Data 9:152.
"""
from __future__ import annotations
import logging
import re
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# Regex para extrair o tipo de sequencia do nome do volume fastMRI brain.
_SEQ_PATTERN = re.compile(r'AX[A-Z0-9]+')

# Sequencias presentes nos 352 volumes elegiveis apos filtragem do S3.
KNOWN_SEQUENCES = ('AXFLAIR', 'AXT1', 'AXT1POST')

# Chaves comuns onde uma mascara pode estar armazenada quando o .pt foi
# salvo como dict (e.g., para preservar metadata junto). Ordem de prioridade.
_MASK_DICT_KEYS = (
    'mask',
    'masks',
    'lesion_mask',
    'lesion_masks',
    'mask_volume',
    'volume_mask',
    'data',
    'tensor',
    'volume',
)


def extract_sequence(volume_id: str) -> str:
    """Extrai o tipo de sequencia (AXFLAIR/AXT1/AXT1POST) do volume_id."""
    match = _SEQ_PATTERN.search(volume_id)
    return match.group(0) if match else 'UNKNOWN'


def _extract_mask_tensor(raw: Any, source_path: Path) -> torch.Tensor:
    """Extrai um Tensor de mascara de um objeto carregado de .pt.

    Suporta os 3 formatos comuns de save no S3:
        1. Tensor direto: torch.save(mask_tensor, path)
        2. Dict com chave conhecida (e.g. 'mask', 'lesion_mask', 'mask_volume')
        3. Dict com unico tensor entre os valores (fallback heuristico)

    Tambem aceita numpy arrays / listas via torch.as_tensor (legacy).

    Parameters
    ----------
    raw : Any
        Objeto retornado por torch.load().
    source_path : Path
        Path do .pt (para mensagens de erro com contexto).

    Returns
    -------
    torch.Tensor
        Tensor de mascara (ainda nao castado para float). Tipicamente
        shape (n_slices, H, W) ou (H, W).

    Raises
    ------
    ValueError
        Se raw for dict sem chave conhecida e sem tensor unico, ou
        tipo nao convertivel.
    """
    if torch.is_tensor(raw):
        return raw

    if isinstance(raw, dict):
        # Tenta chaves comuns na ordem de prioridade.
        for key in _MASK_DICT_KEYS:
            v = raw.get(key)
            if torch.is_tensor(v):
                return v

        # Fallback: se ha exatamente 1 tensor entre os valores, usa ele.
        tensor_items = [(k, v) for k, v in raw.items() if torch.is_tensor(v)]
        if len(tensor_items) == 1:
            k, v = tensor_items[0]
            logger.warning(
                f"Mascara em {source_path.name} e dict com 1 tensor sob chave "
                f"'{k}' (nao em _MASK_DICT_KEYS). Usando assim mesmo; "
                f"considere adicionar '{k}' a lista para silenciar este warning."
            )
            return v

        raise ValueError(
            f"Mascara em {source_path} e dict com chaves {list(raw.keys())} "
            f"e {len(tensor_items)} tensores entre os valores. Nao consegui "
            f"identificar o tensor da mascara. Adicione a chave correta a "
            f"_MASK_DICT_KEYS em src/data/recons_dataset.py, ou re-salve o "
            f".pt como tensor direto."
        )

    # Fallback final para numpy arrays / listas.
    try:
        return torch.as_tensor(raw)
    except (TypeError, RuntimeError) as e:
        raise ValueError(
            f"Mascara em {source_path} tem tipo nao suportado "
            f"{type(raw).__name__}: {e}"
        )


class ReconsSliceDataset(Dataset):
    """Dataset slice-wise sobre os .npz pre-computados no S4.

    Cada item retorna uma fatia 2D normalizada por max_val do volume e a
    mascara de lesao binaria correspondente (zeros se nao houver mascara).
    A interface e identica para os 3 Grupos A/B/C — a diferenca esta em
    quais campos cada loss consome (interface unificada, decisao D3).

    Indexacao
    ---------
    Em construcao, o Dataset varre `recons_dir` carregando o numero de
    slices de cada .npz (campo `recon.shape[0]`) e constroi uma lista de
    tuplas (npz_path, slice_idx). Acesso via __getitem__ e O(1) na lista,
    seguido de IO sob demanda.

    Cache
    -----
    O Dataset mantem o ultimo .npz e a ultima mascara .pt abertos em
    memoria. Sob iteracao do DataLoader com shuffle=False (padrao do
    treino slice-wise), todas as ~16 fatias de um volume sao servidas
    pelo mesmo arquivo em memoria, eliminando reabertura redundante.
    Para shuffle=True o cache ainda funciona, mas com hit rate menor.

    Parameters
    ----------
    recons_dir : str | Path
        Diretorio do split (e.g. data/recons/train/) com arquivos .npz
        produzidos pelo S4 (schema: recon, target, error_map, max_val,
        volume_id, split, acceleration, center_fraction, varnet_sha256).
    masks_dir : str | Path or None, default None
        Diretorio com mascaras .pt nomeadas {volume_id}.pt do S3,
        contendo tensor (n_slices, H, W) OU dict com a mascara sob uma
        das chaves em _MASK_DICT_KEYS. Obrigatorio para Grupo C
        (QR-Lesion); opcional para A (ResM) e B (QR), que ignoram
        a mascara. Quando None, lesion_mask retorna zeros.
    apply_normalization : bool, default True
        Se True, divide recon/target/error_map por max_val. Mantenha
        True para treino — D1 (Romano et al., 2019, §3.2). False existe
        para inspecao manual de magnitudes brutas.
    """

    def __init__(
        self,
        recons_dir: Union[str, Path],
        masks_dir: Optional[Union[str, Path]] = None,
        apply_normalization: bool = True,
    ) -> None:
        self.recons_dir = Path(recons_dir).expanduser().resolve()
        self.masks_dir = Path(masks_dir).expanduser().resolve() if masks_dir else None
        self.apply_normalization = apply_normalization

        if not self.recons_dir.is_dir():
            raise FileNotFoundError(
                f'recons_dir nao encontrado: {self.recons_dir}'
            )

        npz_files = sorted(self.recons_dir.glob('*.npz'))
        if not npz_files:
            raise FileNotFoundError(
                f'Nenhum .npz em {self.recons_dir}. '
                f'Rode scripts/precompute_reconstructions.py primeiro.'
            )

        # Indexa (npz_path, slice_idx). Abrir cada .npz uma vez aqui custa
        # ~50 ms x 213 volumes = ~10 s de overhead de construcao no train
        # split — aceitavel para uma vez por epoca.
        self.index: list[tuple[Path, int]] = []
        for npz_path in npz_files:
            with np.load(npz_path, allow_pickle=False) as data:
                n_slices = int(data['recon'].shape[0])
            for s in range(n_slices):
                self.index.append((npz_path, s))

        # Cache do ultimo .npz aberto (lazy, populado no primeiro __getitem__).
        self._cached_npz_path: Optional[Path] = None
        self._cached_npz_data: Optional[dict] = None
        self._cached_mask_volume: Optional[str] = None
        self._cached_mask_tensor: Optional[torch.Tensor] = None

        logger.info(
            f'ReconsSliceDataset: {len(self.index)} slices em '
            f'{len(npz_files)} volumes em {self.recons_dir.name}'
        )

    def __len__(self) -> int:
        return len(self.index)

    def _load_npz(self, npz_path: Path) -> dict:
        """Carrega o .npz inteiro, cacheando o ultimo aberto."""
        if self._cached_npz_path == npz_path and self._cached_npz_data is not None:
            return self._cached_npz_data

        with np.load(npz_path, allow_pickle=False) as raw:
            data = {
                'recon': raw['recon'][...],
                'target': raw['target'][...],
                'error_map': raw['error_map'][...],
                'max_val': float(raw['max_val']),
                'volume_id': str(raw['volume_id']),
                'split': str(raw['split']),
                'acceleration': int(raw['acceleration']),
                'center_fraction': float(raw['center_fraction']),
                'varnet_sha256': str(raw['varnet_sha256']),
            }

        self._cached_npz_path = npz_path
        self._cached_npz_data = data
        return data

    def _load_mask_volume(self, volume_id: str) -> Optional[torch.Tensor]:
        """Carrega tensor de mascaras (S, H, W) do volume, cacheando.

        Retorna None se masks_dir e None ou se o .pt nao existe.
        Aceita .pt salvo como tensor direto OU dict (ver _extract_mask_tensor).
        """
        if self.masks_dir is None:
            return None

        if self._cached_mask_volume == volume_id and self._cached_mask_tensor is not None:
            return self._cached_mask_tensor

        mask_path = self.masks_dir / f'{volume_id}.pt'
        if not mask_path.is_file():
            if self._cached_mask_volume != volume_id:
                logger.debug(f'Mascara nao encontrada para {volume_id}')
            self._cached_mask_volume = volume_id
            self._cached_mask_tensor = None
            return None

        # weights_only=False: dict de mascara nao e weights, e o default
        # de torch>=2.6 (weights_only=True) recusa carregar dicts arbitrarios.
        raw = torch.load(mask_path, map_location='cpu', weights_only=False)
        mask = _extract_mask_tensor(raw, mask_path).float()

        self._cached_mask_volume = volume_id
        self._cached_mask_tensor = mask
        return mask

    def __getitem__(self, idx: int) -> dict:
        if not 0 <= idx < len(self.index):
            raise IndexError(f'idx={idx} fora de range [0, {len(self.index)})')

        npz_path, slice_idx = self.index[idx]
        data = self._load_npz(npz_path)

        max_val = data['max_val']
        if max_val <= 0:
            raise ValueError(
                f'max_val={max_val} <= 0 em {npz_path.name}. '
                f'Volume corrompido — re-rode precompute_reconstructions.'
            )

        recon_2d = data['recon'][slice_idx].astype(np.float32)
        target_2d = data['target'][slice_idx].astype(np.float32)
        error_2d = data['error_map'][slice_idx].astype(np.float32)

        if self.apply_normalization:
            recon_2d = recon_2d / max_val
            target_2d = target_2d / max_val
            error_2d = error_2d / max_val

        sample = {
            'recon': torch.from_numpy(recon_2d).unsqueeze(0),       # (1, H, W)
            'target': torch.from_numpy(target_2d).unsqueeze(0),
            'error_map': torch.from_numpy(error_2d).unsqueeze(0),
            'max_val': torch.tensor(max_val, dtype=torch.float32),
            'volume_id': data['volume_id'],
            'slice_idx': slice_idx,
            'sequence': extract_sequence(data['volume_id']),
        }

        # Mascara de lesao: zeros se masks_dir e None ou .pt ausente.
        volume_masks = self._load_mask_volume(data['volume_id'])
        if volume_masks is not None:
            sample['lesion_mask'] = volume_masks[slice_idx].unsqueeze(0).float()
        else:
            sample['lesion_mask'] = torch.zeros_like(sample['recon'])

        return sample
