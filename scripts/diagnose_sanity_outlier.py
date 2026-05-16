# Autor: Massanori
# Data: 14/05/2026
# Descrição: Diagnóstico slice-a-slice dos volumes do sanity cache, para
#            identificar se uma SSIM volumétrica baixa vem (a) do volume
#            inteiro estar com erro, ou (b) de 1-2 slices anômalos
#            puxando a média. Recebe: nada (lê de config.recons_dir()/
#            sanity/cache/). Retorna: para cada volume cachado, imprime
#            média, std, mínimo, máximo e os 3 piores slices com seus
#            índices. Decisão sobre o pipeline (seguir para passo 5 ou
#            investigar bug) fica visível pelo padrão estatístico.
#            Roda em ~1 segundo (zero inferência, só lê numpy do disco).
#            Usa: python scripts/diagnose_sanity_outlier.py


"""Diagnostico slice-a-slice do sanity check cache.

Para cada volume em <output>/sanity/cache/, calcula SSIM slice-a-slice
e imprime a distribuicao + 3 slices mais fracos. Util para decidir se
um outlier do sanity_check_varnet.py veio do volume todo ou de slices
especificos.

Pressupoe que sanity_check_varnet.py foi rodado com --use-cache.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
from fastmri.evaluate import ssim as ssim_fn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402


def ssim_one_slice(target_2d: np.ndarray, recon_2d: np.ndarray, max_val: float) -> float:
    """SSIM em 1 slice, usando o ssim_fn do fastmri com shape (1, H, W)."""
    raw = ssim_fn(target_2d[None], recon_2d[None], maxval=max_val)
    return float(np.asarray(raw).mean())


def diagnose_volume(cache_file: Path) -> dict:
    data = np.load(cache_file)
    recon = data['recon_vol']    # (S, H, W)
    target = data['target_vol']  # (S, H, W)
    max_val = float(data['max_val'])

    per_slice = np.array([
        ssim_one_slice(target[i], recon[i], max_val)
        for i in range(recon.shape[0])
    ])

    # 3 piores e 3 melhores indices
    worst_idx = np.argsort(per_slice)[:3].tolist()
    best_idx = np.argsort(per_slice)[-3:][::-1].tolist()

    return {
        'stem': cache_file.stem,
        'n_slices': int(recon.shape[0]),
        'mean': float(per_slice.mean()),
        'std': float(per_slice.std()),
        'min': float(per_slice.min()),
        'max': float(per_slice.max()),
        'worst_slices': [(i, float(per_slice[i])) for i in worst_idx],
        'best_slices': [(i, float(per_slice[i])) for i in best_idx],
        'per_slice': per_slice.tolist(),
    }


def print_volume_report(r: dict) -> None:
    print(f'\n{r["stem"]}')
    print(f'  n_slices:  {r["n_slices"]}')
    print(f'  mean:      {r["mean"]:.4f}  (sanity volumetrico)')
    print(f'  std:       {r["std"]:.4f}')
    print(f'  range:     [{r["min"]:.4f}, {r["max"]:.4f}]')
    print(f'  3 piores:  {[(i, f"{v:.4f}") for i, v in r["worst_slices"]]}')
    print(f'  3 melhores:{[(i, f"{v:.4f}") for i, v in r["best_slices"]]}')


def interpret(reports):
    """Imprime interpretacao agregada — eh um volume ruim, ou slices ruins?"""
    print('\n' + '=' * 72)
    print('INTERPRETACAO')
    print('=' * 72)

    for r in reports:
        if r['mean'] >= 0.94:
            verdict = 'OK'
        elif r['std'] > 0.05 and r['min'] < 0.85:
            verdict = (
                f'SLICES RUINS: std alto ({r["std"]:.4f}) + slice minimo '
                f'baixo ({r["min"]:.4f}). Provavel: 1-2 slices anomalos '
                f'puxando a media. Acao: visualizar slice(s) '
                f'{[i for i, _ in r["worst_slices"]]} para confirmar artefato.'
            )
        else:
            verdict = (
                f'VOLUME RUIM HOMOGENEO: std baixo ({r["std"]:.4f}) e SSIMs '
                f'todos abaixo do esperado. Possiveis causas: caracteristica '
                f'sistematica do exame (movimento, ruido, anatomia), nao '
                f'1 slice especifico.'
            )
        print(f'\n{r["stem"]}: {verdict}')


def main():
    cache_dir = config.recons_dir() / 'sanity' / 'cache'
    if not cache_dir.is_dir():
        print(f'Cache nao encontrado em {cache_dir}.')
        print('Rode primeiro: python scripts/sanity_check_varnet.py --use-cache')
        sys.exit(1)

    cache_files = sorted(cache_dir.glob('*.npz'))
    if not cache_files:
        print(f'Nenhum .npz em {cache_dir}.')
        sys.exit(1)

    print('=' * 72)
    print('DIAGNOSTICO SLICE-A-SLICE - sanity cache')
    print('=' * 72)

    reports = []
    for cf in cache_files:
        r = diagnose_volume(cf)
        print_volume_report(r)
        reports.append(r)

    interpret(reports)


if __name__ == '__main__':
    main()
