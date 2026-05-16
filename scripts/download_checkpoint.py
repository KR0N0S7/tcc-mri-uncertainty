# Autor: Massanori
# Data: 14/05/2026
# Descrição: Baixa o checkpoint brain 4x oficial do fastMRI (Sriram et al.,
#            2020) hospedado em dl.fbaipublicfiles.com. Recebe: nada (paths
#            via env vars do src.config). Retorna: nada — escreve o .pt em
#            checkpoints/brain_leaderboard_state_dict.pt e imprime SHA-256
#            + tamanho. Idempotente: se o arquivo ja existe e o SHA-256
#            bate com o registrado em checkpoints/MANIFEST.md, pula download.
#            Roda com: python scripts/download_checkpoint.py
#            Tamanho esperado: ~390 MB.


"""Download determinístico do checkpoint VarNet brain 4x oficial.

URL fonte: https://dl.fbaipublicfiles.com/fastMRI/trained_models/varnet/brain_leaderboard_state_dict.pt
"""
from __future__ import annotations
import sys
from pathlib import Path

import requests
from tqdm import tqdm

# Adiciona a raiz do repo ao sys.path para importar src.* quando o script
# e chamado diretamente (python scripts/download_checkpoint.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.varnet_loader import CHECKPOINT_URL, compute_sha256  # noqa: E402

CHUNK_SIZE = 8 * 1024 * 1024  # 8 MiB — mesmo chunk do script oficial fastMRI
DEFAULT_OUT_DIR = Path(__file__).resolve().parent.parent / 'checkpoints'


def download(url: str, out_path: Path) -> None:
    """Streaming download com barra de progresso e overwrite seguro
    (escreve em .tmp e renomeia ao final).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + '.tmp')

    with requests.get(url, stream=True, timeout=30) as response:
        response.raise_for_status()
        total = int(response.headers.get('content-length', 0))
        with open(tmp_path, 'wb') as fh, tqdm(
            total=total,
            unit='iB',
            unit_scale=True,
            desc=out_path.name,
        ) as bar:
            for chunk in response.iter_content(CHUNK_SIZE):
                fh.write(chunk)
                bar.update(len(chunk))

    tmp_path.replace(out_path)


def main() -> None:
    fname = CHECKPOINT_URL.rsplit('/', 1)[-1]
    out_path = DEFAULT_OUT_DIR / fname

    if out_path.exists():
        existing_sha = compute_sha256(out_path)
        print(f'Ja existe: {out_path}')
        print(f'SHA-256: {existing_sha}')
        print(f'Tamanho: {out_path.stat().st_size / 1024 / 1024:.1f} MB')
        print('Para forcar redownload, delete o arquivo e rode de novo.')
        return

    print(f'Baixando de: {CHECKPOINT_URL}')
    print(f'Destino:     {out_path}')
    download(CHECKPOINT_URL, out_path)

    sha = compute_sha256(out_path)
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f'\nDownload concluido.')
    print(f'SHA-256: {sha}')
    print(f'Tamanho: {size_mb:.1f} MB')
    print(f'\nProximos passos:')
    print(f'1. Anote o SHA-256 acima em checkpoints/MANIFEST.md')
    print(f'2. Valide com: python -c "from src.models import load_pretrained_varnet; '
          f'm, s = load_pretrained_varnet(\'{out_path}\'); print(s)"')


if __name__ == '__main__':
    main()
