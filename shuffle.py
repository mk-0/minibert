import os
import glob
from collections import defaultdict
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

from io_utils import get_mask_fn, get_dataset_size


def reader(paths, row_len):
    for p in paths:
        memmap = np.memmap(p, dtype=np.uint16).reshape(-1, row_len)
        meta, tokens = np.split(memmap, [3], axis=1)
        batch = mask(tokens)
        yield from np.concatenate(
            (meta, batch["input"], batch["target"], batch["mask"]), axis=1
        )


# See https://blog.janestreet.com/how-to-shuffle-a-big-dataset/
# The only catch is that we need to know file sizes in advance for memmap
if __name__ == "__main__":
    np.random.seed(420)
    cfg = OmegaConf.load("config.yaml")
    assert cfg.data.num_shards <= np.iinfo(np.uint8).max
    paths = glob.glob(f"{cfg.data.raw_dir}/*.npy")
    paths = sorted(paths, key=lambda p: int(Path(p).stem))

    input_row_len = cfg.data.sequence_size + 3  # tokens and id
    output_row_len = 3 * cfg.data.sequence_size + 3  # input, target, mask, id

    dataset_size = get_dataset_size(paths, row_size=input_row_len)
    print(f"Total of {dataset_size} rows in the dataset")

    mask = get_mask_fn(
        mask_token=cfg.data.mask_token,
        vocab_size=cfg.data.vocab_size,
        mask_p=cfg.data.masking.mask_p,
        random_p=cfg.data.masking.random_p,
        keep_p=cfg.data.masking.keep_p,
    )

    os.makedirs(cfg.data.processed_dir, exist_ok=True)
    distribution = np.random.randint(0, cfg.data.num_shards, (dataset_size,), np.uint8)

    shards = [
        np.memmap(
            f"{cfg.data.processed_dir}/{i}.npy",
            mode="w+",
            dtype=np.uint16,
            shape=((distribution == i).sum(), output_row_len),
        )
        for i in range(cfg.data.num_shards)
    ]

    counters = defaultdict(int)
    for i, row in tqdm(
        zip(range(dataset_size), reader(paths, input_row_len)),  # ignore padding
        desc="Distributing rows",
        total=dataset_size,
    ):
        shard_num = distribution[i]
        shards[shard_num][counters[shard_num]] = row
        counters[shard_num] += 1

    for shard in shards:
        shard.flush()

    del shards
    del distribution
    assert sum(counters.values()) == dataset_size

    for path in tqdm(
        glob.glob(f"{cfg.data.processed_dir}/*.npy"),
        desc="Shuffling shards",
        total=cfg.data.num_shards,
    ):
        shard = np.memmap(path, dtype=np.uint16, mode="r+").reshape(-1, output_row_len)
        np.random.shuffle(shard)
        shard.flush()
