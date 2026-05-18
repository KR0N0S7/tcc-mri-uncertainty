# Autor: Massanori
# Data: 17/05/2026
# Descrição: DualLogger: escreve metricas em TensorBoard E em CSV em paralelo
#            (D5). Recebe: log_dir (Path) para TB ou None, csv_path (Path)
#            para CSV append-only. Retorna: objeto com .log_scalar(key, value,
#            iteration) e .close(). Justificativa: TB para inspecao visual
#            durante o treino; CSV plano-texto append-only para resiliencia
#            (se o Kaggle cair, CSV ja tem tudo escrito ate o crash, sem
#            depender de flush do TB).


"""DualLogger: TensorBoard + CSV em paralelo.

TB visualiza em runtime (curves, histograms); CSV e plano-texto append-only
resiliente a crashes. Se o Kaggle cair, CSV ja tem tudo escrito ate o crash,
permitindo retomar a analise sem depender do estado interno do TB writer.

Line-buffered (buffering=1): cada \n forca flush, garantindo que linhas
gravadas estao em disco antes de a proxima escrita ocorrer.

Refs:
    Abadi, M. et al. (2016). TensorFlow: A System for Large-Scale Machine
        Learning. OSDI 2016. (TensorBoard como ferramenta de inspecao)
"""
from __future__ import annotations

import csv
import logging
import time
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)


class DualLogger:
    """Logger duplo: TensorBoard + CSV.

    O TB writer e lazy (so importa torch.utils.tensorboard quando usado);
    se a importacao falhar (e.g. tensorboard nao instalado), continua
    apenas com CSV e loga warning.

    Parameters
    ----------
    log_dir : Path or None
        Diretorio do TensorBoard. None desabilita TB (CSV-only).
    csv_path : Path
        Caminho do CSV. Diretorio pai e criado se necessario. Append-only:
        se o arquivo ja existe, novas linhas sao acrescentadas. Header e
        escrito apenas se o arquivo nao existir ou estiver vazio.

    Notes
    -----
    Suporta uso como context manager (with DualLogger(...) as log:).
    """

    CSV_HEADER = ['iteration', 'key', 'value', 'timestamp']

    def __init__(
        self,
        log_dir: Optional[Union[str, Path]],
        csv_path: Union[str, Path],
    ) -> None:
        self.csv_path = Path(csv_path).expanduser().resolve()
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)

        # Header se arquivo nao existe ou esta vazio
        if not self.csv_path.exists() or self.csv_path.stat().st_size == 0:
            with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(self.CSV_HEADER)

        # CSV em append, line-buffered
        self._csv_file = open(
            self.csv_path, 'a', newline='',
            encoding='utf-8', buffering=1,
        )
        self._csv_writer = csv.writer(self._csv_file)

        # TB lazy
        self._tb_writer = None
        self._tb_dir = (
            Path(log_dir).expanduser().resolve() if log_dir else None
        )
        if self._tb_dir is not None:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._tb_dir.mkdir(parents=True, exist_ok=True)
                self._tb_writer = SummaryWriter(str(self._tb_dir))
            except ImportError:
                logger.warning(
                    'tensorboard nao disponivel; usando apenas CSV.'
                )

    def log_scalar(
        self,
        key: str,
        value: float,
        iteration: int,
    ) -> None:
        """Loga uma metrica escalar em TB e CSV."""
        timestamp = time.time()
        self._csv_writer.writerow([iteration, key, float(value), timestamp])
        # Flush implicito pelo buffering=1 do open
        if self._tb_writer is not None:
            self._tb_writer.add_scalar(key, value, iteration)

    def close(self) -> None:
        """Fecha handles. Chame ao final do treino."""
        try:
            self._csv_file.close()
        except Exception as exc:
            logger.debug(f'Erro fechando CSV: {exc}')
        if self._tb_writer is not None:
            try:
                self._tb_writer.close()
            except Exception as exc:
                logger.debug(f'Erro fechando TB: {exc}')

    def __enter__(self) -> 'DualLogger':
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
