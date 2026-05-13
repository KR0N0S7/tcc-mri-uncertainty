# Autor: Massanori
# Data: 13/05/2026
# Descrição: Smoke test do pipeline de geração de máscaras ponta a ponta. Roda
#            em 1 volume específico (default: file_brain_AXFLAIR_200_6002493
#            fatia 6, o volume suspeito da validação inicial). Recebe: pasta
#            anotados/, brain.csv, nome do volume (--volume) e índice da fatia
#            (--slice). Retorna: figura PNG com 3 painéis (reconstruction_rss
#            original, máscara binária pós-flip, overlay) salva em
#            figures/validacao_bbox/smoke_test_mask.png. Confirma que a
#            correção de flip vertical em bbox_to_mask funciona em produção,
#            não só nos testes unitários.


"""Smoke test do pipeline de mascaras."""
import argparse
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config
from src.data.lesion_masks import volume_masks_from_h5


def main(anotados: Path, brain_csv_path: Path, out_fig: Path,
         volume: str, slice_idx: int):
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    brain_df = pd.read_csv(brain_csv_path)
    masks = volume_masks_from_h5(anotados / f'{volume}.h5', brain_df,
                                 apply_y_flip=True)
    with h5py.File(anotados / f'{volume}.h5', 'r') as hf:
        img = hf['reconstruction_rss'][slice_idx]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('reconstruction_rss (fastMRI)')
    axes[1].imshow(masks[slice_idx], cmap='Reds')
    axes[1].set_title(f'Mascara pos-flip\n({int(masks[slice_idx].sum().item())} pixels)')
    axes[2].imshow(img, cmap='gray')
    axes[2].imshow(masks[slice_idx], cmap='Reds', alpha=0.4)
    axes[2].set_title('Overlay')
    for ax in axes:
        ax.axis('off')
    plt.suptitle(f'{volume} fatia {slice_idx}', fontsize=11)
    plt.tight_layout()
    plt.savefig(out_fig, dpi=120, bbox_inches='tight')
    plt.close()

    print(f'Shape: {tuple(masks.shape)}')
    print(f'Fatias com lesao: {int((masks.sum(dim=(1,2)) > 0).sum().item())}/{masks.shape[0]}')
    print(f'Pixels totais: {int(masks.sum().item())}')
    print(f'Figura: {out_fig.resolve()}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--anotados', type=Path, default=None)
    parser.add_argument('--brain-csv', type=Path, default=None)
    parser.add_argument('--figures-dir', type=Path, default=None)
    parser.add_argument('--volume', default='file_brain_AXFLAIR_200_6002493',
                        help='Nome do volume sem extensao (default: o suspeito da validacao)')
    parser.add_argument('--slice', type=int, default=6)
    args = parser.parse_args()

    figures = args.figures_dir or config.figures_dir()
    main(
        anotados=args.anotados or config.anotados_dir(),
        brain_csv_path=args.brain_csv or config.brain_csv(),
        out_fig=figures / 'validacao_bbox' / 'smoke_test_mask.png',
        volume=args.volume,
        slice_idx=args.slice,
    )