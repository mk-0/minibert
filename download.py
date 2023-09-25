import os

import numpy as np
import jax.numpy as jnp
from omegaconf import OmegaConf
from tqdm import tqdm
from tokenizers import Tokenizer, processors
from datasets import load_dataset

from io_utils import MmapWriter, extract_sequences, pack_article


if __name__ == "__main__":
    np.random.seed(420)
    cfg = OmegaConf.load("config.yaml")
    assert cfg.data.vocab_size <= np.iinfo(np.uint16).max

    dataset = load_dataset("wikipedia", "20220301.en", split="train", streaming=True)
    tokenizer = Tokenizer.from_file(cfg.tokenizer.checkpoint_path)
    # turn off autoinsertion of [SEP] and [CLS]
    tokenizer.post_processor = processors.TemplateProcessing(single="$0")

    os.makedirs(cfg.data.raw_dir, exist_ok=True)
    mmap_shape = (cfg.data.download_file_size, cfg.data.sequence_size + 3)
    writer = MmapWriter(cfg.data.raw_dir, mmap_shape)
    for article in tqdm(dataset, total=dataset.info.splits["train"].num_examples):
        tokens = np.array(tokenizer.encode(article["text"]).ids)
        chunk = extract_sequences(tokens, sequence_size=cfg.data.sequence_size)
        output = pack_article(chunk, int(article["id"]))
        writer.write(output)

    # f = np.memmap(f"{cfg.data.raw_dir}/{fnum}.npy", dtype=np.uint16, mode="r").reshape(mmap_shape)
