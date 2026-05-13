# Autor: Massanori
# Data: 13/05/2026
# Descrição: Copia para a pasta destino apenas os volumes do fastMRI Brain que
#            possuem pelo menos UMA anotação bounding-box slice-level no
#            fastMRI+. Volumes com apenas anotação study-level (Normal for age,
#            Small vessel chronic WM ischemic change, etc., n=398) são
#            EXCLUÍDOS, pois não produzem máscara binária e são incompatíveis
#            com L_QR-Lesion e métricas regionais (Coverage_lesion, ULAS).
#            Recebe: pasta origem (--origem) com .h5 extraídos, pasta destino
#            (env TCC_ANOTADOS_DIR ou --destino) e brain.csv (env TCC_BRAIN_CSV).
#            Retorna: cópia idempotente dos arquivos elegíveis com verificação
#            de integridade tripla (tamanho, HDF5 válido, reconstruction_rss
#            presente) e relatório com distribuição por sequência.

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config

parser = argparse.ArgumentParser()
parser.add_argument('--origem', type=Path, required=True,
                    help='Pasta com os .h5 brutos extraidos (ex: D:\\Mri\\multi_train9)')
parser.add_argument('--destino', type=Path, default=None,
                    help='Pasta destino (default: $TCC_ANOTADOS_DIR)')
parser.add_argument('--brain-csv', type=Path, default=None,
                    help='Caminho do brain.csv (default: $TCC_BRAIN_CSV)')
args = parser.parse_args()

ORIGEM = args.origem
DESTINO = args.destino or config.anotados_dir()
BRAIN_CSV = args.brain_csv or config.brain_csv()