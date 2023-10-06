from bisect import bisect_left
from pathlib import Path

import numpy as np
import jax.numpy as jnp
from omegaconf import OmegaConf


def load_config(path="config.yaml"):
    cfg = OmegaConf.load("config.yaml")
    OmegaConf.set_struct(cfg, True)  # disallow adding new keys to catch typos
    return OmegaConf.merge(cfg, OmegaConf.from_cli())


class MmapWriter:
    def __init__(self, root_dir, file_shape, dtype=np.uint16):
        self.root_dir = root_dir
        self.file_shape = file_shape
        self.dtype = dtype

        self.file = None
        self.file_num = -1
        self.cursor = None

    def rotate(self):
        self.flush()
        self.file_num += 1

        self.file = np.memmap(
            f"{self.root_dir}/{self.file_num}.npy",
            dtype=self.dtype,
            mode="w+",
            shape=self.file_shape,
        )
        self.cursor = 0

    def write(self, chunk):
        assert len(chunk.shape) == 2

        if (chunk == 0).all():
            raise ValueError("Zero-only rows are invalid (used for paddding)")

        if self.file is None:
            self.rotate()

        chunk = chunk.astype(self.dtype)
        next_cursor = min(self.cursor + len(chunk), len(self.file))
        # simply crop the chunk if it does not fit
        self.file[self.cursor : next_cursor] = chunk[: next_cursor - self.cursor]
        if next_cursor < len(self.file):
            self.cursor = next_cursor
        else:
            self.rotate()

    def flush(self):
        if self.file is not None:
            self.file.flush()

    def __del__(self):
        self.flush()


def extract_sequences(tokens, sequence_size):
    # crop the part of the article that doesn't fit
    cropped_size = len(tokens) // sequence_size * sequence_size
    # Cramming Bert: no padding, no split at sentence boundaries. Just a contigious chunk
    return tokens[:cropped_size].reshape(-1, sequence_size)


def pack_article(chunk, aid):
    # prepend article id (2 columns) and row number
    chunk = np.insert(chunk, 0, values=np.arange(len(chunk)), axis=1)
    chunk = np.insert(chunk, (0, 0), values=encode_id(aid), axis=1)
    return chunk


def load_batch(array, copy=False):
    array = jnp.array(array, copy=copy)
    meta, data = jnp.split(array, [3], axis=1)
    batch = dict(zip(("input", "target", "mask"), jnp.split(data, 3, axis=1)))
    batch["meta"] = meta
    return batch


def encode_id(article_id):
    "Article id can be too large to store in uint16, splitting in half"
    return article_id >> 16, article_id & 0xFFFF


def decode_id(first_part, second_part):
    return (first_part << 16) | (second_part & 0xFFFF)


def get_dataset_size(paths, row_size, dtype=np.uint16):
    # Last file might be padded. The rest are full and of the same shape
    paths = sorted(paths, key=lambda p: int(Path(p).stem))
    last = np.memmap(paths[-1], dtype=dtype).reshape(-1, row_size)
    last_size = bisect_left(last, True, key=lambda x: (x == 0).all())
    total_size = (len(paths) - 1) * len(last) + last_size
    return total_size


def get_mask_fn(mask_token, vocab_size, mask_p, random_p, keep_p):
    if not (0 <= mask_p + keep_p + random_p <= 1):
        raise ValueError("Probabilities must be within [0, 1]")

    probs = np.array([1 - mask_p - random_p - keep_p, mask_p, random_p, keep_p])

    def mask(tokens):
        # 0 - input context, ignore when calculating loss
        # 1 - mask, expect reconstruction of original token
        # 2 - replace by a random token, expect reconstruction
        # 3 - keep original token and expect it to be copied to output
        choice = np.random.choice(4, tokens.shape, p=probs)
        masked = np.where(choice == 1, mask_token, tokens)
        masked = np.where(
            choice == 2,
            np.random.randint(0, vocab_size, tokens.shape),
            masked,
        )
        return {"input": masked, "target": tokens, "mask": choice > 0}

    return mask


def batched(array, batch_size, drop_last=True):
    assert len(array) >= batch_size > 0
    cropped_len = len(array) // batch_size * batch_size
    cropped, rest = jnp.split(array, [cropped_len])

    for i in range(0, cropped_len, batch_size):
        yield cropped[i : i + batch_size]

    if len(rest) and not drop_last:
        yield rest
