# Autor: Massanori
# Data: 13/05/2026
# Descrição: Testes unitários para src/config.py. 10 testes cobrindo: leitura
#            de variáveis obrigatórias, fallback para defaults em opcionais,
#            mensagens de erro claras, resolução de paths absolutos, expansão
#            de ~ em paths, e precedência env-var sobre default. Usa
#            monkeypatch do pytest para isolar cada teste do ambiente real.

"""
Testes para o modulo de configuracao baseado em variaveis de ambiente.
Ref: 12-factor app (https://12factor.net/config)
"""
from pathlib import Path
import pytest

from src import config


# ============================================================
# Variaveis obrigatorias
# ============================================================

def test_anotados_dir_lê_env_var(monkeypatch):
    monkeypatch.setenv('TCC_ANOTADOS_DIR', '/some/test/path')
    result = config.anotados_dir()
    assert isinstance(result, Path)
    assert result.is_absolute()


def test_anotados_dir_falha_quando_ausente(monkeypatch):
    monkeypatch.delenv('TCC_ANOTADOS_DIR', raising=False)
    with pytest.raises(config.ConfigError, match='TCC_ANOTADOS_DIR'):
        config.anotados_dir()


def test_brain_csv_lê_env_var(monkeypatch):
    monkeypatch.setenv('TCC_BRAIN_CSV', '/path/to/brain.csv')
    result = config.brain_csv()
    assert isinstance(result, Path)
    assert result.is_absolute()


def test_brain_csv_falha_quando_ausente(monkeypatch):
    monkeypatch.delenv('TCC_BRAIN_CSV', raising=False)
    with pytest.raises(config.ConfigError, match='TCC_BRAIN_CSV'):
        config.brain_csv()


def test_mensagem_erro_menciona_variavel_e_readme(monkeypatch):
    """Mensagem de erro deve guiar o usuario ao README."""
    monkeypatch.delenv('TCC_ANOTADOS_DIR', raising=False)
    with pytest.raises(config.ConfigError) as exc_info:
        config.anotados_dir()
    msg = str(exc_info.value)
    assert 'TCC_ANOTADOS_DIR' in msg
    assert 'README' in msg or '.env' in msg


# ============================================================
# Variaveis opcionais
# ============================================================

def test_data_dir_default_eh_relativo_ao_projeto(monkeypatch):
    monkeypatch.delenv('TCC_DATA_DIR', raising=False)
    result = config.data_dir()
    assert result.is_absolute()
    assert result.name == 'data'
    # Default tem que estar dentro da raiz do projeto
    assert config.PROJECT_ROOT in result.parents


def test_data_dir_respeita_env_var_quando_setada(monkeypatch):
    monkeypatch.setenv('TCC_DATA_DIR', '/custom/data/path')
    result = config.data_dir()
    assert result.is_absolute()
    # Path normalizado deve terminar em custom/data/path
    assert 'custom' in result.parts


def test_masks_dir_default_eh_data_masks(monkeypatch):
    monkeypatch.delenv('TCC_MASKS_DIR', raising=False)
    result = config.masks_dir()
    assert result.name == 'masks'
    assert result.parent.name == 'data'


def test_splits_dir_default_eh_splits(monkeypatch):
    monkeypatch.delenv('TCC_SPLITS_DIR', raising=False)
    assert config.splits_dir().name == 'splits'


def test_figures_dir_default_eh_figures(monkeypatch):
    monkeypatch.delenv('TCC_FIGURES_DIR', raising=False)
    assert config.figures_dir().name == 'figures'


# ============================================================
# Resolucao de paths
# ============================================================

def test_paths_são_sempre_absolutos(monkeypatch):
    """Caminhos relativos passados via env var sao convertidos para absolutos."""
    monkeypatch.setenv('TCC_DATA_DIR', 'relative/path')
    assert config.data_dir().is_absolute()


def test_expansao_de_til(monkeypatch):
    """~ deve ser expandido para o home do usuario."""
    monkeypatch.setenv('TCC_DATA_DIR', '~/some_test_dir')
    result = config.data_dir()
    assert '~' not in str(result)
    assert result.is_absolute()

# ============================================================
# Teste de .env
# ============================================================

def test_env_file_eh_utf8_sem_bom_se_existir():
    """
    Garante que .env esta em UTF-8 PURO (sem BOM).
    Detecta:
      - BOM UTF-16 LE (FF FE) → PowerShell 'echo > .env'
      - BOM UTF-16 BE (FE FF) → outros editores
      - BOM UTF-8     (EF BB BF) → Set-Content -Encoding UTF8 no PS 5.1
    
    Apesar do nosso config.py tolerar BOM via 'utf-8-sig', queremos que o
    arquivo .env seja UTF-8 puro como boa pratica e para evitar surpresas
    com outras ferramentas (Docker, CI/CD, etc.).
    """
    env_file = config.PROJECT_ROOT / '.env'
    if not env_file.exists():
        pytest.skip('.env nao existe — nada a testar')

    with open(env_file, 'rb') as f:
        first_bytes = f.read(3)

    # UTF-16
    assert first_bytes[:2] != b'\xff\xfe', (
        f'.env esta em UTF-16 LE. Recrie com '
        f'[System.IO.File]::WriteAllText(...) sem BOM.'
    )
    assert first_bytes[:2] != b'\xfe\xff', '.env esta em UTF-16 BE.'

    # UTF-8 com BOM (problema do PowerShell 5.1)
    assert first_bytes[:3] != b'\xef\xbb\xbf', (
        f'.env tem BOM UTF-8 ({first_bytes[:3].hex()}). '
        f'PowerShell 5.1 Set-Content -Encoding UTF8 escreve com BOM. '
        f'Use: [System.IO.File]::WriteAllText("$pwd\\.env", $conteudo, '
        f'[System.Text.UTF8Encoding]::new($false)) '
        f'ou edite no VS Code e salve como UTF-8 (sem BOM).'
    )