import os

import numpy as np
import jax.numpy as jnp
from omegaconf import OmegaConf
from tqdm import tqdm
from tokenizers import Tokenizer, processors
from datasets import load_dataset


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


def pack_batch(batch, aid):
    array = np.concatenate((batch["input"], batch["target"], batch["mask"]), axis=1)

    # prepend article id and chunk number
    array = np.insert(array, 0, values=np.arange(len(array)), axis=1)
    array = np.insert(array, (0, 0), values=encode_id(aid), axis=1)
    return array


def unpack_batch(array):
    # ignore article id and chunk number
    array = jnp.array(array[:, 3:])
    return dict(zip(("input", "target", "mask"), jnp.split(array, 3, axis=1)))


def encode_id(article_id):
    "Article id can be too large to store in uint16, splitting in half"
    return article_id >> 16, article_id & 0xFFFF


def decode_id(first_part, second_part):
    return (first_part << 16) | (second_part & 0xFFFF)


if __name__ == "__main__":
    np.random.seed(420)
    cfg = OmegaConf.load("config.yaml")
    assert cfg.data.vocab_size <= np.iinfo(np.uint16).max

    dataset = load_dataset(
        "wikipedia", "20220301.en", split="train", streaming=True
    )
    tokenizer = Tokenizer.from_file(cfg.tokenizer.checkpoint_path)
    # turn off autoinsertion of [SEP] and [CLS]
    tokenizer.post_processor = processors.TemplateProcessing(single="$0")

    mask = get_mask_fn(
        mask_token=cfg.data.mask_token,
        vocab_size=cfg.data.vocab_size,
        mask_p=cfg.data.masking.mask_p,
        random_p=cfg.data.masking.random_p,
        keep_p=cfg.data.masking.keep_p,
    )

    os.makedirs(cfg.data.raw_dir, exist_ok=True)
    mmap_shape = (cfg.data.download_file_size, cfg.data.sequence_size * 3 + 3)
    fnum = 0
    cursor = 0
    f = np.memmap(
        f"{cfg.data.raw_dir}/{fnum}.npy", dtype=np.uint16, mode="w+", shape=mmap_shape
    )
    for article in tqdm(dataset, total=dataset.info.splits["train"].num_examples):
        tokens = np.array(tokenizer.encode(article["text"]).ids)
        data_end = (  # crop to enable reshaping
            len(tokens) // cfg.data.sequence_size * cfg.data.sequence_size
        )
        # Cramming Bert: no padding, no split at sentence boundaries. Just a contigious chunk
        chunk = tokens[:data_end].reshape(-1, cfg.data.sequence_size)
        batch = mask(chunk)
        output = pack_batch(batch, int(article["id"]))

        next_cursor = min(cursor + len(output), len(f))
        f[cursor:next_cursor] = output[: next_cursor - cursor]  # crop if does not fit
        if next_cursor < len(f):
            cursor = next_cursor
        else:
            f.flush()
            fnum += 1
            cursor = 0
            f = np.memmap(
                f"{cfg.data.raw_dir}/{fnum}.npy",
                dtype=np.uint16,
                mode="w+",
                shape=mmap_shape,
            )
    f.flush()

    # f = np.memmap(f"{cfg.data.raw_dir}/{fnum}.npy", dtype=np.uint16, mode="r").reshape(mmap_shape)
