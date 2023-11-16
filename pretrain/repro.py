"""
This script is adapted from TinyLlama:
https://github.com/jzhang38/TinyLlama/blob/main/pretrain/tinyllama.py
"""
import math
import sys
import time
from functools import partial
from pathlib import Path
from typing import Tuple, Union

import lightning as L
import torch
import torch.nn as nn
from lightning.data import StreamingDataset
from lightning.data.streaming.item_loader import TokensLoader
from lightning.fabric.strategies import FSDPStrategy
from lightning.fabric.utilities.throughput import Throughput, get_available_flops, measure_flops
from lightning.fabric.loggers import CSVLogger
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader
from tqdm import tqdm

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_gpt.model import GPT, Block, Config, LLaMAMLP
from lit_gpt.packed_dataset import CombinedDataset
from lit_gpt.utils import chunked_cross_entropy, num_parameters

# System settings
model_name = "tiny-llama-1.1b"
name = "training-debug-loss"
out_dir = Path("out") / name
use_wandb = False

# Hyperparameters
devices = 1
micro_batch_size = 16
eval_iters = 100

hparams = {k: v for k, v in locals().items() if isinstance(v, (int, float, str)) and not k.startswith("_")}


def setup(resume: Union[bool, Path] = False):
    if devices > 1:
        strategy = FSDPStrategy(
            auto_wrap_policy={Block},
            activation_checkpointing_policy=None,
            state_dict_type="full",
            limit_all_gathers=True,
            cpu_offload=False,
        )
    else:
        strategy = "auto"

    fabric = L.Fabric(devices=devices, strategy=strategy, precision="bf16-mixed")
    fabric.launch()

    fabric.print(hparams)
    if use_wandb:
        logger.log_hyperparams(hparams)

    main(fabric, resume)


def main(fabric, resume):
    config = Config.from_name(model_name)

    val_dataloader = create_dataloaders(
        fabric, batch_size=micro_batch_size, block_size=config.block_size
    )
    val_dataloader = fabric.setup_dataloaders(val_dataloader)

    for _ in tqdm(val_dataloader):
        pass

def create_dataloaders(fabric: L.Fabric, batch_size: int, block_size: int) -> Tuple[DataLoader, DataLoader]:
    # Increase by one because we need the next word as well
    effective_block_size = block_size + 1
    val_dataset = StreamingDataset(
        input_dir="data/slimpajama/val",
        item_loader=TokensLoader(block_size=effective_block_size), 
        shuffle=False,
        # Consider setting to False, but we would lose some samples due to truncation when world size > 1
        drop_last=False,
    )
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True, num_workers=8)
    return val_dataloader


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    from jsonargparse import CLI

    CLI(setup)
