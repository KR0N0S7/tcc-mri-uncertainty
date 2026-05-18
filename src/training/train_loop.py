# Autor: Massanori
# Data: 18/05/2026
# Descrição: Loop de treino unificado para os Grupos A/B/C do S5. Recebe:
#            modulo nn.Module (ResM ou QR/QR-Lesion), loss_fn (resm_loss,
#            qr_loss ou qr_lesion_loss), DataLoaders train/val, total_iters,
#            config dict (subset de configs/training_base.json), run_dir.
#            Retorna: dict com metricas finais. Efeitos colaterais:
#            run_dir/last.pt, run_dir/best.pt, run_dir/metrics.csv,
#            run_dir/tb/ (TensorBoard). Polimorfico via interface
#            unificada D3: pred = module(recon) pode ser tensor (Grupo A)
#            ou dict {'lower', 'upper'} (Grupos B/C); loss_fn consome o
#            tipo correto sem branches no codigo. Atomicidade nos saves
#            (.tmp + rename) garante que crashes do Kaggle nao corrompam
#            checkpoints anteriores. Resume automatico de last.pt se existir.
#            Protegido por testes em tests/test_train_loop.py.


"""Loop de treino unificado para os Grupos A/B/C do S5.

Desenho:
    - cycle_loader: itera infinitamente sobre o DataLoader (Giannakopoulos
      et al., 2026 conta em iters, nao em epocas)
    - AdamW + linear warmup ate warmup_steps, depois constante
      (replicacao do paper base; Loshchilov & Hutter, 2019, para AdamW)
    - Checkpoint atomico com .tmp + rename (mesma estrategia do S4)
    - CSV + TensorBoard logging em paralelo (CSV e resiliente a crashes)
    - Resume automatico de run_dir/last.pt se existir

Refs:
    Giannakopoulos, C. et al. (2026). Pixelwise Uncertainty Quantification
        of Accelerated MRI Reconstruction. arXiv:2601.13236. (Sec. III.D)
    Loshchilov, I.; Hutter, F. (2019). Decoupled Weight Decay Regularization.
        ICLR. arXiv:1711.05101.
    Paszke, A. et al. (2019). PyTorch: An Imperative Style, High-Performance
        Deep Learning Library. NeurIPS 32.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Iterator, Optional, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.training.random_seed import set_global_seed

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Optimizer e scheduler
# ---------------------------------------------------------------------------

def make_optimizer(
    module: nn.Module,
    config: dict,
) -> torch.optim.Optimizer:
    """AdamW com hiperparametros do config (replicacao Giannakopoulos et al., 2026).

    AdamW e preferido sobre Adam pelo decoupled weight decay, que evita
    interacao entre L2 regularization e lr scheduling (Loshchilov & Hutter,
    2019).

    Parameters
    ----------
    module : nn.Module
        Modulo cujos parameters() serao otimizados.
    config : dict
        Subset do configs/training_base.json, esperado conter chave
        'optimizer' com lr, weight_decay, betas, eps.

    Returns
    -------
    torch.optim.AdamW
    """
    cfg = config['optimizer']
    return torch.optim.AdamW(
        module.parameters(),
        lr=cfg['lr'],
        weight_decay=cfg['weight_decay'],
        betas=tuple(cfg['betas']),
        eps=cfg['eps'],
    )


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    config: dict,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Linear warmup ate warmup_steps, depois constante.

    lr(step) = lr_base * min(1, step / warmup_steps)

    Replicacao do paper base (Giannakopoulos et al., 2026, secao III.D):
    warmup_steps=7500 sobre 210000 iters totais. Apos warmup, lr fica
    constante (sem decay) ate o fim.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
    config : dict
        Espera chave 'scheduler' com 'warmup_steps'.

    Returns
    -------
    torch.optim.lr_scheduler.LambdaLR
    """
    warmup = config['scheduler']['warmup_steps']

    def lr_lambda(step: int) -> float:
        if warmup <= 0:
            return 1.0
        if step < warmup:
            return float(step) / float(warmup)
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Iterators e steps
# ---------------------------------------------------------------------------

def cycle_loader(loader: DataLoader) -> Iterator:
    """Itera infinitamente sobre um DataLoader.

    O paper base conta em iterations, nao em epocas. cycle_loader permite
    que o loop principal faca `for it in range(total_iters)` independente
    do tamanho do dataset. Diferente de itertools.cycle, NAO cacheia
    batches (que estouraria RAM com tensores grandes); apenas reinicia
    o iterador ao chegar no fim.
    """
    while True:
        for batch in loader:
            yield batch


def train_step(
    module: nn.Module,
    batch: dict,
    loss_fn: Callable,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: Union[str, torch.device],
    loss_kwargs: Optional[dict] = None,
) -> float:
    """Um passo de treino.

    Polimorfico: aceita qualquer modulo cujo forward(recon) retorne tensor
    (Grupo A) ou dict {'lower', 'upper'} (Grupos B/C), desde que loss_fn
    saiba consumir o tipo apropriado.

    Parameters
    ----------
    module : nn.Module
    batch : dict
        Saida do ReconsSliceDataset. Esperado conter 'recon', 'target',
        'lesion_mask'.
    loss_fn : Callable
        resm_loss, qr_loss ou qr_lesion_loss.
    optimizer, scheduler : ja construidos via make_optimizer/make_scheduler.
    device : 'cpu', 'cuda' ou torch.device.
    loss_kwargs : dict or None
        Kwargs extras para loss_fn (e.g. alpha=0.10, lambda_lesion=5.0).

    Returns
    -------
    float
        Valor da loss neste step (escalar).
    """
    module.train()
    optimizer.zero_grad(set_to_none=True)
    recon = batch['recon'].to(device, non_blocking=True)
    target = batch['target'].to(device, non_blocking=True)
    lesion_mask = batch['lesion_mask'].to(device, non_blocking=True)

    pred = module(recon)
    loss = loss_fn(pred, recon, target, lesion_mask, **(loss_kwargs or {}))

    loss.backward()
    optimizer.step()
    scheduler.step()
    return loss.item()


@torch.no_grad()
def validate(
    module: nn.Module,
    loader: DataLoader,
    loss_fn: Callable,
    device: Union[str, torch.device],
    loss_kwargs: Optional[dict] = None,
) -> float:
    """Loss media no val split, em modo eval.

    Restaura modo train() ao final para nao quebrar o loop principal.
    """
    module.eval()
    losses = []
    for batch in loader:
        recon = batch['recon'].to(device, non_blocking=True)
        target = batch['target'].to(device, non_blocking=True)
        lesion_mask = batch['lesion_mask'].to(device, non_blocking=True)
        pred = module(recon)
        loss = loss_fn(pred, recon, target, lesion_mask, **(loss_kwargs or {}))
        losses.append(loss.item())
    module.train()
    return sum(losses) / len(losses) if losses else float('nan')


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: Path,
    iteration: int,
    module: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    best_val_loss: float,
    config_snapshot: dict,
) -> None:
    """Salva checkpoint atomico (.tmp + rename).

    Inclui optimizer e scheduler state_dict para retomada exata do schedule
    de lr. Sem isso, retomar treino apos timeout do Kaggle corrompe o
    schedule de lr (Paszke et al., 2019, secao 2.5).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    payload = {
        'iteration': iteration,
        'model_state_dict': module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_loss': float(best_val_loss),
        'config_snapshot': config_snapshot,
    }
    torch.save(payload, tmp)
    tmp.replace(path)  # rename atomico no mesmo filesystem


def load_checkpoint(
    path: Path,
    module: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    device: Union[str, torch.device] = 'cpu',
) -> dict:
    """Carrega checkpoint, restaurando module/optimizer/scheduler in-place.

    Returns
    -------
    dict
        Payload completo do checkpoint, util para inspecionar iteration e
        best_val_loss.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Checkpoint nao encontrado: {path}')
    # weights_only=False porque o payload contem dicts Python (config_snapshot,
    # optimizer state). Em projeto pessoal sem fonte externa de checkpoints,
    # o risco de pickle malicioso e nulo.
    ckpt = torch.load(path, map_location=device, weights_only=False)
    module.load_state_dict(ckpt['model_state_dict'])
    if optimizer is not None and 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if scheduler is not None and 'scheduler_state_dict' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    return ckpt


# ---------------------------------------------------------------------------
# Loop principal
# ---------------------------------------------------------------------------

def train(
    module: nn.Module,
    loss_fn: Callable,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    total_iters: int,
    config: dict,
    run_dir: Union[str, Path],
    device: Union[str, torch.device] = 'cuda',
    seed: int = 42,
    val_every: int = 1000,
    log_every: int = 100,
    checkpoint_every: int = 10000,
    resume: bool = True,
    loss_kwargs: Optional[dict] = None,
    config_snapshot: Optional[dict] = None,
) -> dict:
    """Loop de treino unificado para os Grupos A/B/C do S5.

    Parameters
    ----------
    module : nn.Module
        ResidualMagnitudeModule, QuantileRegressionModule ou
        QuantileRegressionLesionModule.
    loss_fn : Callable
        resm_loss, qr_loss ou qr_lesion_loss. Assinatura:
        loss_fn(pred, recon, target, lesion_mask, **loss_kwargs).
    train_loader, val_loader : DataLoader
        Configurados a partir de ReconsSliceDataset. val_loader pode ser
        None (smoke test).
    total_iters : int
        Total de iteracoes (NAO epocas). MVP=210000.
    config : dict
        Subset de configs/training_base.json com chaves 'optimizer' e
        'scheduler'.
    run_dir : Path
        Diretorio para checkpoints/logs/tb. Criado se nao existir.
    device : str or torch.device, default 'cuda'
    seed : int, default 42
        MESMO entre Grupos A/B/C (controle experimental, D4).
    val_every, log_every, checkpoint_every : int
        Frequencias em iteracoes. Para smoke test, reduzir todas.
    resume : bool, default True
        Se True e run_dir/last.pt existe, retoma a partir dele.
    loss_kwargs : dict or None
        Kwargs extras para loss_fn (e.g. {'alpha': 0.10, 'lambda_lesion': 5.0}).
    config_snapshot : dict or None
        Snapshot do config completo para gravar no checkpoint (auditoria).
        Se None, usa o proprio `config`.

    Returns
    -------
    dict
        Metricas finais: final_train_loss, final_val_loss, best_val_loss,
        total_iters, elapsed_seconds.

    Notes
    -----
    Atomicidade dos saves protege contra timeouts do Kaggle. Se a sessao
    cair entre `tmp` e `replace`, o last.pt anterior continua valido e o
    proximo resume retoma dele.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1. Setup determinístico (D4, Demsar 2006)
    set_global_seed(seed)

    # 2. Device transfer ANTES do optimizer (parametros devem estar no device
    #    final para o optimizer referenciar os tensores corretos)
    module = module.to(device)

    # 3. Optimizer e scheduler
    optimizer = make_optimizer(module, config)
    scheduler = make_scheduler(optimizer, config)

    # 4. Resume (se houver last.pt)
    start_iter = 0
    best_val_loss = float('inf')
    last_ckpt = run_dir / 'last.pt'
    if resume and last_ckpt.is_file():
        ckpt = load_checkpoint(last_ckpt, module, optimizer, scheduler, device)
        start_iter = int(ckpt['iteration'])
        best_val_loss = float(ckpt.get('best_val_loss', float('inf')))
        logger.info(
            f'Resumindo de iter {start_iter} (best_val_loss={best_val_loss:.6f})'
        )

    if start_iter >= total_iters:
        logger.info(
            f'start_iter ({start_iter}) >= total_iters ({total_iters}). '
            f'Nada a fazer.'
        )
        return {
            'final_train_loss': float('nan'),
            'final_val_loss': float('nan'),
            'best_val_loss': best_val_loss,
            'total_iters': total_iters,
            'elapsed_seconds': 0.0,
        }

    # 5. Snapshot do config (auditoria)
    snapshot = config_snapshot if config_snapshot is not None else config
    snapshot_path = run_dir / 'config_snapshot.json'
    if not snapshot_path.exists():
        snapshot_path.write_text(json.dumps(snapshot, indent=2), encoding='utf-8')

    # 6. CSV logger (resiliente a crashes; TensorBoard pode falhar, CSV nao)
    csv_path = run_dir / 'metrics.csv'
    csv_existed = csv_path.is_file()
    csv_file = csv_path.open('a', encoding='utf-8')
    if not csv_existed:
        csv_file.write('iteration,train_loss,val_loss,lr,elapsed_s\n')

    # 7. TensorBoard logger (opcional; cai elegantemente se ausente)
    tb_writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        tb_writer = SummaryWriter(str(run_dir / 'tb'))
    except ImportError:
        logger.warning('tensorboard nao disponivel; logando apenas CSV')

    # 8. Loop principal
    train_iter = cycle_loader(train_loader)
    start_time = time.time()
    final_train_loss = float('nan')
    final_val_loss = float('nan')

    try:
        for it in range(start_iter, total_iters):
            batch = next(train_iter)
            train_loss = train_step(
                module, batch, loss_fn,
                optimizer, scheduler, device,
                loss_kwargs,
            )
            final_train_loss = train_loss

            step = it + 1  # 1-indexed para log

            # Log de treino
            if step % log_every == 0 or step == total_iters:
                lr = optimizer.param_groups[0]['lr']
                elapsed = time.time() - start_time
                csv_file.write(
                    f'{step},{train_loss:.6f},,{lr:.6e},{elapsed:.1f}\n'
                )
                csv_file.flush()
                if tb_writer is not None:
                    tb_writer.add_scalar('train/loss', train_loss, step)
                    tb_writer.add_scalar('train/lr', lr, step)
                logger.info(
                    f'iter {step}/{total_iters}  '
                    f'train_loss={train_loss:.6f}  lr={lr:.2e}'
                )

            # Validacao + best.pt
            if val_loader is not None and (step % val_every == 0 or step == total_iters):
                val_loss = validate(module, val_loader, loss_fn, device, loss_kwargs)
                final_val_loss = val_loss
                lr = optimizer.param_groups[0]['lr']
                elapsed = time.time() - start_time
                csv_file.write(
                    f'{step},,{val_loss:.6f},{lr:.6e},{elapsed:.1f}\n'
                )
                csv_file.flush()
                if tb_writer is not None:
                    tb_writer.add_scalar('val/loss', val_loss, step)
                logger.info(f'  val_loss={val_loss:.6f}')

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(
                        run_dir / 'best.pt',
                        step, module, optimizer, scheduler,
                        best_val_loss, snapshot,
                    )
                    logger.info(
                        f'  novo best (val_loss={val_loss:.6f}) salvo'
                    )

            # Checkpoint periodico
            if step % checkpoint_every == 0:
                save_checkpoint(
                    run_dir / 'last.pt',
                    step, module, optimizer, scheduler,
                    best_val_loss, snapshot,
                )
                logger.info(f'  last.pt salvo (iter {step})')
    finally:
        # Sempre salva last.pt no final, mesmo se houver excecao
        save_checkpoint(
            run_dir / 'last.pt',
            it + 1 if 'it' in locals() else start_iter,
            module, optimizer, scheduler,
            best_val_loss, snapshot,
        )
        csv_file.close()
        if tb_writer is not None:
            tb_writer.close()

    elapsed = time.time() - start_time
    return {
        'final_train_loss': final_train_loss,
        'final_val_loss': final_val_loss,
        'best_val_loss': best_val_loss,
        'total_iters': total_iters,
        'elapsed_seconds': elapsed,
    }
