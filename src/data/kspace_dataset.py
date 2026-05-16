# Autor: Massanori
# Data: 14/05/2026
# Descrição: Builder do dataset brain multicoil para inferência da VarNet
#            pré-treinada brain 4x oficial. Compõe sobre fastmri.data.SliceDataset
#            usando o hook raw_sample_filter para filtrar volumes do split
#            estratificado do S3 (splits/{train,val,cal,test}.txt). Aplica
#            EquispacedMaskFractionFunc com center_fractions=[0.08],
#            accelerations=[4] — os hiperparâmetros que o checkpoint brain
#            oficial foi treinado para receber, conforme
#            run_pretrained_varnet_inference.py do fastMRI@91f2df47. Nota:
#            existem 2 classes equispaced no fastmri: EquiSpacedMaskFunc
#            (legado, com round()) e EquispacedMaskFractionFunc (atual, com
#            floor() garantindo a fração exata). O checkpoint oficial espera
#            a versão Fraction. Usa VarNetDataTransform com use_seed=True,
#            que deriva a seed da máscara do hash do fname: dois runs
#            produzem exatamente a mesma máscara para o mesmo volume,
#            sustentando a reprodutibilidade do TCC. Iteração é por slice
#            (compatível com SliceDataset oficial); agrupamento por volume
#            é responsabilidade do consumer (precompute_reconstructions.py
#            do passo 5). Protegido por testes em tests/test_kspace_dataset.py.


"""Builder do dataset k-space brain multicoil 4x para inferência VarNet.

Refs:
    Sriram, A. et al. (2020). End-to-End Variational Networks for
        Accelerated MRI Reconstruction. MICCAI.
        https://arxiv.org/abs/2004.06688
    fastmri.data.subsample.EquispacedMaskFractionFunc — máscara equispaced
        com cálculo exato de fração central (versão atual da lib).
    fastmri.data.transforms.VarNetDataTransform — transform que entrega
        VarNetSample (NamedTuple com masked_kspace, mask, target, etc.).
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Callable, Iterable, Optional, Set, Union

from fastmri.data import SliceDataset
from fastmri.data.subsample import EquispacedMaskFractionFunc
from fastmri.data.transforms import VarNetDataTransform

logger = logging.getLogger(__name__)

# Hiperparametros do checkpoint brain 4x oficial. center_fraction=0.08 e
# acceleration=4 sao os valores em que o leaderboard model foi treinado
# (Sriram et al., 2020, secao 4.1). Confirmados contra a linha
# correspondente de run_pretrained_varnet_inference.py em
# facebookresearch/fastMRI@91f2df47.
DEFAULT_ACCELERATION = 4
DEFAULT_CENTER_FRACTION = 0.08

# Splits estratificados pelo make_splits.py do S3.
VALID_SPLIT_NAMES = ('train', 'val', 'cal', 'test')


def load_split(
    split_name: str,
    splits_dir: Union[str, Path],
) -> Set[str]:
    """Carrega volume IDs de splits/{split_name}.txt.

    Cada linha do arquivo e um volume ID (stem sem extensao .h5), por exemplo:
    `file_brain_AXFLAIR_200_6002460`. Linhas em branco e whitespace nas
    pontas sao ignorados.

    Parameters
    ----------
    split_name : str
        Um dos splits validos: train, val, cal, test.
    splits_dir : str | Path
        Diretorio contendo os arquivos .txt do split.

    Returns
    -------
    set[str]
        Conjunto de volume IDs (stems sem extensao).

    Raises
    ------
    ValueError
        Se split_name nao e valido, ou se o arquivo esta vazio.
    FileNotFoundError
        Se splits_dir/{split_name}.txt nao existe.
    """
    if split_name not in VALID_SPLIT_NAMES:
        raise ValueError(
            f'split_name deve ser um de {VALID_SPLIT_NAMES}, '
            f'recebido: {split_name!r}'
        )

    split_file = Path(splits_dir) / f'{split_name}.txt'
    if not split_file.is_file():
        raise FileNotFoundError(f'Split file nao encontrado: {split_file}')

    volume_ids = {
        line.strip()
        for line in split_file.read_text(encoding='utf-8').splitlines()
        if line.strip()
    }
    if not volume_ids:
        raise ValueError(f'Split file esta vazio: {split_file}')

    return volume_ids


def make_volume_filter(allowed_volume_ids: Iterable[str]) -> Callable:
    """Cria filtro para SliceDataset.raw_sample_filter.

    O SliceDataset do fastmri chama o filtro para cada slice descoberto,
    passando uma tupla `(fname, dataslice, metadata)` onde fname e um
    Path apontando para o .h5 do volume. O filtro retorna True se
    Path(fname).stem esta no conjunto de volumes permitidos.

    A comparacao e por igualdade do stem, NAO por prefixo: isso evita
    que 'file_brain_AXFLAIR_200_6002460' case com
    'file_brain_AXFLAIR_200_60024600' (extra digit).

    Parameters
    ----------
    allowed_volume_ids : Iterable[str]
        IDs (stems sem extensao) dos volumes a aceitar.

    Returns
    -------
    Callable
        Funcao filter pronta para SliceDataset.raw_sample_filter.

    Raises
    ------
    ValueError
        Se allowed_volume_ids esta vazio.
    """
    allowed = set(allowed_volume_ids)
    if not allowed:
        raise ValueError('allowed_volume_ids esta vazio.')

    def _filter(raw_sample) -> bool:
        fname = raw_sample[0]
        return Path(fname).stem in allowed

    return _filter


def make_brain_mask_func(
    acceleration: int = DEFAULT_ACCELERATION,
    center_fraction: float = DEFAULT_CENTER_FRACTION,
) -> EquispacedMaskFractionFunc:
    """Cria a EquispacedMaskFractionFunc esperada pelo checkpoint brain
    4x oficial.

    Usa EquispacedMaskFractionFunc (e nao EquiSpacedMaskFunc legado) porque:
    1. O run_pretrained_varnet_inference.py oficial usa essa classe;
    2. O calculo de linhas centrais via floor() garante que a fracao
       seja exatamente preservada — usar round() (Func legado) introduz
       1-2 linhas de diferenca no centro, degradando levemente a SSIM.

    Os defaults (acc=4, center=0.08) batem com os hiperparametros de
    treinamento do leaderboard model (Sriram et al., 2020).
    """
    return EquispacedMaskFractionFunc(
        center_fractions=[center_fraction],
        accelerations=[acceleration],
    )


def build_brain_kspace_dataset(
    data_path: Union[str, Path],
    volume_ids: Optional[Iterable[str]] = None,
    split_name: Optional[str] = None,
    splits_dir: Optional[Union[str, Path]] = None,
    acceleration: int = DEFAULT_ACCELERATION,
    center_fraction: float = DEFAULT_CENTER_FRACTION,
    use_seed: bool = True,
) -> SliceDataset:
    """Builder do dataset brain multicoil para inferencia da VarNet 4x.

    Aceita 1 de 2 modos:
    - **Modo split**: passe split_name ('val', etc.) e splits_dir.
      Internamente carrega volume_ids de splits/{split_name}.txt.
    - **Modo explicito**: passe volume_ids diretamente (lista, set, etc.).
      Util para sanity check em N volumes especificos.

    Os dois modos sao mutuamente exclusivos: passar ambos ou nenhum
    levanta ValueError.

    Parameters
    ----------
    data_path : str | Path
        Raiz com os arquivos .h5 do fastMRI brain multicoil. Tipicamente
        a saida do download oficial (ex.: D:/Mri/anotados/).
    volume_ids : Iterable[str], optional
        IDs (stems sem extensao) dos volumes a incluir. Mutuamente
        exclusivo com split_name.
    split_name : str, optional
        Nome do split em VALID_SPLIT_NAMES. Mutuamente exclusivo com
        volume_ids.
    splits_dir : str | Path, optional
        Diretorio dos arquivos .txt. Obrigatorio quando split_name e dado.
    acceleration : int, default 4
        Fator de aceleracao da mascara. Mantenha em 4 para o checkpoint
        oficial — outros valores produzem reconstrucoes degradadas.
    center_fraction : float, default 0.08
        Fracao de linhas centrais sempre amostradas. Idem: 0.08 para o
        checkpoint oficial.
    use_seed : bool, default True
        Se True, a seed da mascara e derivada do fname — produzindo
        mascaras determinísticas e reproduziveis entre runs.

    Returns
    -------
    fastmri.data.SliceDataset
        Dataset iteravel por slice. Cada __getitem__ retorna um
        VarNetSample (NamedTuple) com masked_kspace, mask, target,
        fname, slice_num, max_value, etc.

    Raises
    ------
    ValueError
        Em caso de uso invalido (ambos ou nenhum modo, splits_dir faltando).
    RuntimeError
        Se o dataset construido esta vazio (provavel mismatch entre
        data_path e volume_ids).
    """
    if (volume_ids is None) == (split_name is None):
        raise ValueError(
            'Passe EXATAMENTE UM de volume_ids ou split_name '
            f'(volume_ids={volume_ids is not None}, '
            f'split_name={split_name!r}).'
        )

    if split_name is not None:
        if splits_dir is None:
            raise ValueError(
                'splits_dir e obrigatorio quando split_name e fornecido.'
            )
        volume_ids = load_split(split_name, splits_dir)

    # Materializa volume_ids para podermos contar e re-iterar.
    volume_ids_set = set(volume_ids)

    volume_filter = make_volume_filter(volume_ids_set)
    mask_func = make_brain_mask_func(acceleration, center_fraction)
    transform = VarNetDataTransform(mask_func=mask_func, use_seed=use_seed)

    # SliceDataset varre data_path e descobre slices. use_dataset_cache=False
    # garante que a varredura roda fresca a cada chamada — importante
    # quando alternamos entre splits sem invalidar cache stale.
    dataset = SliceDataset(
        root=Path(data_path),
        challenge='multicoil',
        transform=transform,
        raw_sample_filter=volume_filter,
        use_dataset_cache=False,
    )

    if len(dataset) == 0:
        raise RuntimeError(
            f'Dataset vazio apos filtro. Verifique:\n'
            f'  (1) data_path={data_path} contem os .h5 esperados;\n'
            f'  (2) os {len(volume_ids_set)} volume_ids batem com os '
            f'nomes dos arquivos (stem sem .h5);\n'
            f'  (3) os arquivos sao multicoil (com chave kspace de '
            f'shape (slices, coils, h, w)).'
        )

    logger.info(
        f'Brain dataset construido: {len(dataset)} slices de '
        f'{len(volume_ids_set)} volumes (acc={acceleration}x, '
        f'center={center_fraction}).'
    )
    return dataset
