import os
import glob
import random

import numpy as np
import jax.numpy as jnp
from omegaconf import OmegaConf
from tqdm import tqdm
from tokenizers import Tokenizer, processors
from datasets import load_dataset

from io_utils import MmapWriter, extract_sequences, pack_article


def iter_books(paths):
    for num, path in enumerate(paths):
        with open(path) as f:
            lines = f.read().split("\n\n")
            yield {
                "id": num,
                "text": "\n".join(line for line in lines if len(line) > 20),
            }


if __name__ == "__main__":
    random.seed(420)
    np.random.seed(420)
    cfg = OmegaConf.load("config.yaml")
    assert cfg.data.vocab_size <= np.iinfo(np.uint16).max

    # dataset = load_dataset("wikipedia", "20220301.en", split="train", streaming=True)
    # total = dataset.info.splits["train"].num_examples

    root = "pile/books3/the-eye.eu/public/Books/Bibliotik"
    files = sorted(glob.glob(f"{root}/*/*.txt"))
    # random.shuffle(files)
    dataset = iter_books(files)
    total = len(files)

    tokenizer = Tokenizer.from_file(cfg.tokenizer.checkpoint_path)
    # turn off autoinsertion of [SEP] and [CLS]
    tokenizer.post_processor = processors.TemplateProcessing(single="$0")

    os.makedirs(cfg.data.raw_dir, exist_ok=True)
    mmap_shape = (cfg.data.download_file_size, cfg.data.sequence_size + 3)
    writer = MmapWriter(cfg.data.raw_dir, mmap_shape)
    for article in tqdm(dataset, total=total):
        tokens = np.array(tokenizer.encode(article["text"]).ids)
        chunk = extract_sequences(tokens, sequence_size=cfg.data.sequence_size)
        output = pack_article(chunk, int(article["id"]))
        writer.write(output)

    # f = np.memmap(f"{cfg.data.raw_dir}/{fnum}.npy", dtype=np.uint16, mode="r").reshape(mmap_shape)
