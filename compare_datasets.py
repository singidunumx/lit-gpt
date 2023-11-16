import math
import sys
import time
from functools import partial
from pathlib import Path
from typing import Tuple, Union
import glob
import random
from tqdm import tqdm

import lightning as L
import torch
from torch.utils.data import DataLoader
from lightning.data import StreamingDataset
from lightning.data.streaming.item_loader import TokensLoader
from lit_gpt.packed_dataset import CombinedDataset, PackedDataset


def create_dataloader_packed(block_size):
    data_dir = Path("data/packed/slimpajama/val")
    filenames = sorted(glob.glob(str(data_dir / "val*")))

    dataset = PackedDataset(
        filenames,
        n_chunks=1,
        block_size=block_size,
        shuffle=False,
        seed=42,
        num_processes=1,
        process_rank=0,
    )
    return DataLoader(dataset, batch_size=2, num_workers=2)


def create_dataloader_streaming(block_size):
    dataset = StreamingDataset(
        input_dir="data/streaming/slimpajama/val",
        item_loader=TokensLoader(block_size=block_size), 
        shuffle=False,
        drop_last=False,
    )
    return DataLoader(dataset, batch_size=2, num_workers=2)


def compare():
    dataloader_packed = create_dataloader_packed(block_size=10)
    dataloader_streaming = create_dataloader_streaming(block_size=10)
    iter_packed = iter(dataloader_packed)
    iter_streaming = iter(dataloader_streaming)

    for _ in range(10000):
       p = next(iter_packed).int()
       s = next(iter_streaming).int()
       print("p = ", p[1].tolist())
       print("s = ", s[1].tolist())
       assert torch.equal(p, s)


if __name__ == "__main__":
    compare()


"""
Packed, num_procs=1

Processing /teamspace/s3_connections/tiny-llama-template/SlimPajama-627B/validation/chunk1/example_holdout_0.jsonl.zst
tensor([ 5011, 22196, 29899, 29907,   417,  2650,   472, 21091, 28052,   663],
       dtype=torch.int32)
tensor([ 1976, 29884,   360,  7308, 29875, 29892, 29871, 29896, 29955,  6339],
       dtype=torch.int32)
tensor([ 5011,  5845, 20939,   265,   338,   263, 11443,  2706, 28107, 29892],
       dtype=torch.int32)
tensor([  379,  6180, 29892,  4335,   275,  4112,    13,  1124,   597, 19527],
       dtype=torch.int32)
tensor([1551, 2306, 3131,  310,  450, 7927,  383, 3568,  310, 5765],
       dtype=torch.int32)


Processor, num_procs=1

Processing /cache/data/chunk1/example_holdout_0.jsonl.zst  
tensor([ 5011, 22196, 29899, 29907,   417,  2650,   472, 21091, 28052,   663],
       dtype=torch.int32)
tensor([ 1976, 29884,   360,  7308, 29875, 29892, 29871, 29896, 29955,  6339],
       dtype=torch.int32)
tensor([ 5011,  5845, 20939,   265,   338,   263, 11443,  2706, 28107, 29892],
       dtype=torch.int32)
tensor([  379,  6180, 29892,  4335,   275,  4112,    13,  1124,   597, 19527],
       dtype=torch.int32)
tensor([1551, 2306, 3131,  310,  450, 7927,  383, 3568,  310, 5765],
       dtype=torch.int32)


if "this_studio" not in self.input_dir.path: 
            self.input_dir.path = _try_create_cache_dir(input_dir=self.input_dir.path, shard_rank=env.shard_rank)

"""
