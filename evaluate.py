from pathlib import Path
import random

import pandas as pd
import equinox as eqx
import jax.numpy as jnp
import jax.random as jrandom
from tokenizers import Tokenizer
from tqdm import tqdm

from io_utils import load_config, batched
from model import Bert, BertClassifier
from finetune import forward, make_key_mask, classify


if __name__ == "__main__":
    jnp.set_printoptions(precision=3, suppress=True, linewidth=1000)

    random.seed(420)
    key = jrandom.PRNGKey(420)
    cfg = load_config()

    tokenizer = Tokenizer.from_file(cfg.tokenizer.checkpoint_path)
    tokenizer.enable_truncation(cfg.data.sequence_size)
    tokenizer.enable_padding(pad_id=cfg.data.mask_token)
    bert = Bert(
        vocab_size=cfg.data.vocab_size,
        sequence_size=cfg.data.sequence_size,
        num_blocks=cfg.model.num_blocks,
        num_attention_heads=cfg.model.num_attention_heads,
        num_features=cfg.model.embedding_size,
        num_feedforward_features=cfg.model.feedforward_inner_size,
        dropout_p=cfg.tuning.dropout_p,
        key=key,
    )

    model = BertClassifier(bert, num_classes=3, key=key)
    model = eqx.tree_deserialise_leaves(cfg.evaluation.checkpoint, model)

    root = Path(cfg.tuning.dataset_dir)
    cols = ["index", "sentence1", "sentence2"]
    df_m = pd.read_csv(
        root / "test_matched.tsv", sep="\t", usecols=cols, keep_default_na=False
    )
    df_mm = pd.read_csv(
        root / "test_mismatched.tsv", sep="\t", usecols=cols, keep_default_na=False
    )
    df_m["subset"] = "matched"
    df_mm["subset"] = "mismatched"
    df = pd.concat([df_m, df_mm], axis=0, ignore_index=True)
    sentences = df[["sentence1", "sentence2"]].values.tolist()
    data = jnp.asarray([s.ids for s in tokenizer.encode_batch(sentences)])

    predictions = []
    for batch in tqdm(
        batched(data, cfg.tuning.batch_size, drop_last=False),
        total=len(data) // cfg.tuning.batch_size,
    ):
        mask = make_key_mask(batch, cfg.data.mask_token)
        logits = eqx.filter_jit(forward)(model, batch, mask, key, inference=True)
        predictions.extend(logits.argmax(-1).tolist())

    df["prediction"] = predictions
    df.prediction = df.prediction.replace(
        {0: "neutral", 1: "entailment", 2: "contradiction"}
    )

    df[df.subset == "matched"][["index", "prediction"]].to_csv(
        "MNLI-m.tsv", sep="\t", index=False, header=False
    )
    df[df.subset == "mismatched"][["index", "prediction"]].to_csv(
        "MNLI-mm.tsv", sep="\t", index=False, header=False
    )
