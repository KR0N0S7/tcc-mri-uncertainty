# Autor: Massanori
# Data: 13/05/2026
# Descrição: Pipeline batch de geração das máscaras binárias para todos os
#            volumes anotados. Itera com tqdm e verifica integridade. Recebe:
#            pasta anotados/, brain.csv e pasta de saída (via env vars com
#            defaults). Retorna: um arquivo .pt por volume em data/masks/
#            contendo dict {masks: tensor (n_slices, H, W) float32, volume:
#            stem do nome, shape, apply_y_flip: True, fastmri_plus_ref}.
#            Idempotente — pula volumes já processados a menos que --overwrite
#            seja passado. Imprime resumo final com {with_lesion, no_lesion,
#            skipped, error}.


"""Pipeline batch: gera mascaras para os volumes anotados."""
import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config
from src.data.lesion_masks import volume_masks_from_h5


def main(anotados_dir: Path, brain_csv_path: Path, output_dir: Path, overwrite: bool):
    logging.basicConfig(level=logging.WARNING)
    output_dir.mkdir(parents=True, exist_ok=True)

    brain_df = pd.read_csv(brain_csv_path)
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
    parser.add_argument('--anotados', type=Path, default=None)
    parser.add_argument('--brain-csv', type=Path, default=None)
    parser.add_argument('--out', type=Path, default=None)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    main(
        anotados_dir=args.anotados or config.anotados_dir(),
        brain_csv_path=args.brain_csv or config.brain_csv(),
        output_dir=args.out or config.masks_dir(),
        overwrite=args.overwrite,
    )