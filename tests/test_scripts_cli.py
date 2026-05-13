# Autor: Massanori
# Data: 13/05/2026
# Descrição: Smoke tests de CLI dos scripts em scripts/. 6 testes que executam
#            cada script com --help e verificam: (a) script importa sem erro
#            (catch de SyntaxError e ImportError), (b) argparse responde a
#            --help com código de saída 0, (c) flags principais aparecem na
#            documentação gerada. Não executa o pipeline real — só valida a
#            integridade do CLI.

"""
Smoke tests de CLI dos scripts. Garante que a refatoracao nao quebrou
nenhum entrypoint de linha de comando.
"""
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / 'scripts'

SCRIPTS_COM_FLAGS_ESPERADAS = {
    'audit_dataset.py':        ['--anotados', '--brain-csv'],
    'make_splits.py':          ['--anotados', '--brain-csv', '--out'],
    'generate_lesion_masks.py':['--anotados', '--brain-csv', '--out', '--overwrite'],
    'lesion_area_histogram.py':['--brain-csv', '--splits-dir', '--figures-dir'],
    'smoke_test_masks.py':     ['--anotados', '--brain-csv', '--volume', '--slice'],
    'filtra_anotados.py':      ['--origem', '--destino', '--brain-csv'],
}


@pytest.mark.parametrize('script,flags_esperadas',
                          list(SCRIPTS_COM_FLAGS_ESPERADAS.items()))
def test_script_help_funciona(script, flags_esperadas):
    """Cada script deve responder a --help sem erro e listar suas flags."""
    script_path = SCRIPTS_DIR / script
    assert script_path.exists(), f'{script} nao encontrado em {SCRIPTS_DIR}'

    result = subprocess.run(
        [sys.executable, str(script_path), '--help'],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, (
        f'{script} --help falhou com codigo {result.returncode}.\n'
        f'STDOUT: {result.stdout}\n'
        f'STDERR: {result.stderr}'
    )
    # Todas as flags esperadas devem aparecer na documentacao
    for flag in flags_esperadas:
        assert flag in result.stdout, (
            f'{script} --help nao menciona {flag}.\nSTDOUT: {result.stdout}'
        )