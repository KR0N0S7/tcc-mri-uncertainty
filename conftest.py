"""
Configuracao pytest: adiciona a raiz do projeto ao sys.path
para que 'from src.data.lesion_masks import ...' funcione nos testes.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))