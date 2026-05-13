# Autor: Massanori
# Data: 13/05/2026
# Descrição: Módulo central de configuração que lê variáveis de ambiente
#            (TCC_ANOTADOS_DIR, TCC_BRAIN_CSV e opcionais) com fallback para
#            defaults relativos à raiz do projeto. Recebe: variáveis de ambiente
#            do shell ou arquivo .env. Retorna: objetos Path resolvidos via
#            funções lazy (anotados_dir, brain_csv, masks_dir, recons_dir,
#            splits_dir, configs_dir, figures_dir). Permite que o mesmo código
#            rode em Windows, Linux e Kaggle sem alteração. Aplicação do padrão
#            12-factor app (https://12factor.net/config).

"""
Configuracao centralizada de caminhos do projeto via variaveis de ambiente.

Permite o mesmo codigo rodar em Windows, Linux e Kaggle sem alteracoes.

Variaveis OBRIGATORIAS (especificas da maquina):
    TCC_ANOTADOS_DIR    Pasta com volumes .h5 elegiveis (com bbox slice-level)
    TCC_BRAIN_CSV       Caminho para o brain.csv do fastMRI+

Variaveis OPCIONAIS (defaults relativos a raiz do projeto):
    TCC_DATA_DIR        ./data
    TCC_MASKS_DIR       ./data/masks
    TCC_RECONS_DIR      ./data/recons
    TCC_SPLITS_DIR      ./splits
    TCC_CONFIGS_DIR     ./configs
    TCC_FIGURES_DIR     ./figures

Definicao em 3 modos (escolha um):

Modo A - .env na raiz do projeto (recomendado para desenvolvimento)
    Crie um arquivo .env com o conteudo:
        TCC_ANOTADOS_DIR=D:\\Mri\\anotados
        TCC_BRAIN_CSV=D:\\Mri\\brain.csv
    Requer python-dotenv (pip install python-dotenv).
    O arquivo .env NAO deve ser commitado (esta no .gitignore).

Modo B - shell (Windows PowerShell)
    $env:TCC_ANOTADOS_DIR = "D:\\Mri\\anotados"
    $env:TCC_BRAIN_CSV    = "D:\\Mri\\brain.csv"

Modo C - shell (Linux/Mac/Git Bash, Kaggle)
    export TCC_ANOTADOS_DIR=/kaggle/input/anotados
    export TCC_BRAIN_CSV=/kaggle/input/fastmri-plus/brain.csv

Refs:
    The Twelve-Factor App. (https://12factor.net/config)
"""
from __future__ import annotations
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Carrega .env automaticamente se python-dotenv estiver disponivel
try:
    from dotenv import load_dotenv
    _env_file = PROJECT_ROOT / '.env'
    if _env_file.exists():
        # encoding='utf-8-sig' aceita arquivos UTF-8 com OU sem BOM.
        # PowerShell 5.1 do Windows tende a salvar UTF-8 com BOM via
        # Set-Content -Encoding UTF8, o que confunde o parser default.
        try:
            load_dotenv(_env_file, encoding='utf-8-sig')
        except TypeError:
            # Versoes antigas do python-dotenv nao aceitam encoding kwarg.
            # Fallback: strip BOM manualmente em memoria.
            content = _env_file.read_text(encoding='utf-8-sig')
            for line in content.splitlines():
                line = line.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                key, _, value = line.partition('=')
                os.environ.setdefault(key.strip(), value.strip())
except ImportError:
    pass


class ConfigError(EnvironmentError):
    """Erro de configuracao de ambiente."""


def _required(var_name: str) -> Path:
    value = os.environ.get(var_name)
    if not value:
        raise ConfigError(
            f"\n  Variavel de ambiente '{var_name}' nao definida.\n"
            f"  Defina-a antes de rodar o script.\n"
            f"  Veja README.md secao 'Configuracao do ambiente' para opcoes."
        )
    p = Path(value).expanduser().resolve()
    return p


def _optional(var_name: str, default_relative: str) -> Path:
    value = os.environ.get(var_name)
    if value:
        return Path(value).expanduser().resolve()
    return (PROJECT_ROOT / default_relative).resolve()


# Funcoes (lazy) - so falham se forem chamadas e a env var nao existir
def anotados_dir() -> Path:
    return _required('TCC_ANOTADOS_DIR')


def brain_csv() -> Path:
    return _required('TCC_BRAIN_CSV')


def data_dir() -> Path:
    return _optional('TCC_DATA_DIR', 'data')


def masks_dir() -> Path:
    return _optional('TCC_MASKS_DIR', 'data/masks')


def recons_dir() -> Path:
    return _optional('TCC_RECONS_DIR', 'data/recons')


def splits_dir() -> Path:
    return _optional('TCC_SPLITS_DIR', 'splits')


def configs_dir() -> Path:
    return _optional('TCC_CONFIGS_DIR', 'configs')


def figures_dir() -> Path:
    return _optional('TCC_FIGURES_DIR', 'figures')


def summary() -> str:
    """Util para debug: imprime todas as configuracoes resolvidas."""
    lines = ['Configuracao do projeto TCC:']
    for name, fn in [
        ('PROJECT_ROOT', lambda: PROJECT_ROOT),
        ('ANOTADOS_DIR', anotados_dir),
        ('BRAIN_CSV', brain_csv),
        ('DATA_DIR', data_dir),
        ('MASKS_DIR', masks_dir),
        ('RECONS_DIR', recons_dir),
        ('SPLITS_DIR', splits_dir),
        ('CONFIGS_DIR', configs_dir),
        ('FIGURES_DIR', figures_dir),
    ]:
        try:
            lines.append(f'  {name:15s} = {fn()}')
        except ConfigError as e:
            lines.append(f'  {name:15s} = <NAO DEFINIDA>')
    return '\n'.join(lines)


if __name__ == '__main__':
    print(summary())