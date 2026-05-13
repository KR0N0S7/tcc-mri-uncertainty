# Autor: Massanori
# Data: 13/05/2026
# Descrição: Configuração pytest na raiz do projeto. Adiciona a raiz ao
#            sys.path para que imports do tipo 'from src.data.lesion_masks
#            import ...' funcionem nos testes sem necessidade de instalar o
#            projeto como pacote. Carregado automaticamente pelo pytest antes
#            da coleta dos testes.


"""
Configuracao pytest: adiciona a raiz do projeto ao sys.path
para que 'from src.data.lesion_masks import ...' funcione nos testes.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))