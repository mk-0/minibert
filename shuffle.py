import os
import glob
from collections import defaultdict
from bisect import bisect_left
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm


# See https://blog.janestreet.com/how-to-shuffle-a-big-dataset/
# The only catch is that we need to know file sizes in advance for memmap
if __name__ == "__main__":
    np.random.seed(420)
    cfg = OmegaConf.load("config.yaml")
    assert cfg.data.num_shards <= np.iinfo(np.uint8).max
    sample_len = cfg.data.sequence_size * 3 + 3  # 3 id fields + input + target + mask
    paths = sorted(
        glob.glob(f"{cfg.data.raw_dir}/*.npy"), key=lambda p: int(Path(p).stem)
    )
    last = np.memmap(paths[-1], dtype=np.uint16).reshape(-1, sample_len)

    # find first all-zero row. Everything after it is padding to be ignored
    last_size = bisect_left(last, 0, key=lambda x: -sum(x))
    total_size = (len(paths) - 1) * cfg.data.download_file_size + last_size
    del last

    os.makedirs(cfg.data.processed_dir, exist_ok=True)
    distribution = np.random.randint(0, cfg.data.num_shards, (total_size,), np.uint8)

    shards = [
        np.memmap(
            f"{cfg.data.processed_dir}/{i}.npy",
            mode="w+",
            dtype=np.uint16,
            shape=((distribution == i).sum(), sample_len),
        )
        for i in range(cfg.data.num_shards)
    ]

    def reader(paths, sample_len):
        for p in paths:
            yield from np.memmap(p, dtype=np.uint16).reshape(-1, sample_len)

    counters = defaultdict(int)
    for i, elt in tqdm(
        zip(range(total_size), reader(paths, sample_len)),
        desc="Distributing rows",
        total=total_size,
    ):
        shard_num = distribution[i]
        shards[shard_num][counters[shard_num]] = elt
        counters[shard_num] += 1

    for shard in shards:
        shard.flush()

    del shards
    del distribution
    assert sum(counters.values()) == total_size

    for path in tqdm(
        glob.glob(f"{cfg.data.processed_dir}/*.npy"),
        desc="Shuffling shards",
        total=cfg.data.num_shards,
    ):
        shard = np.memmap(path, dtype=np.uint16, mode="r+").reshape(-1, sample_len)
        np.random.shuffle(shard)
        shard.flush()
