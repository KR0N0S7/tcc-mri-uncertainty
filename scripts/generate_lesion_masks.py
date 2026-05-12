"""
Pipeline batch: gera mascaras de lesao para os 724 volumes anotados.
Output: um .pt por volume em data/masks/{volume_stem}.pt
Uso:
    python scripts/generate_lesion_masks.py \
        --anotados D:/Mri/anotados \
        --brain-csv D:/Mri/brain.csv \
        --out data/masks
"""
import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.lesion_masks import volume_masks_from_h5


def main(anotados_dir: Path, brain_csv: Path, output_dir: Path, overwrite: bool):
    logging.basicConfig(level=logging.WARNING)  # silencia info por volume
    output_dir.mkdir(parents=True, exist_ok=True)

    brain_df = pd.read_csv(brain_csv)
    volumes = sorted(anotados_dir.glob('*.h5'))
    print(f'Volumes a processar: {len(volumes)}')

    stats = {'with_lesion': 0, 'no_lesion': 0, 'skipped': 0, 'error': 0}

    for h5_path in tqdm(volumes, desc='Mascaras'):
        out_path = output_dir / f'{h5_path.stem}.pt'
        if out_path.exists() and not overwrite:
            stats['skipped'] += 1
            continue
        try:
            masks = volume_masks_from_h5(h5_path, brain_df, apply_y_flip=True)
            torch.save({
                'masks': masks,
                'volume': h5_path.stem,
                'shape': tuple(masks.shape),
                'apply_y_flip': True,
                'fastmri_plus_ref': 'Zhao et al. 2022, Sci Data 9:152',
            }, out_path)
            stats['with_lesion' if masks.sum() > 0 else 'no_lesion'] += 1
        except Exception as e:
            print(f'\nERRO em {h5_path.name}: {e}')
            stats['error'] += 1

    print(f'\nResumo: {stats}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--anotados', type=Path, required=True)
    parser.add_argument('--brain-csv', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    main(args.anotados, args.brain_csv, args.out, args.overwrite)