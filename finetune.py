import os
import sys
from pathlib import Path
import random
import pickle

import jax
import numpy as np
import pandas as pd
import equinox as eqx
import jax.numpy as jnp
import jax.random as jrandom
import optax
import wandb
from tokenizers import Tokenizer
from tqdm import tqdm

from io_utils import load_config, batched
from model import Bert, BertClassifier, precision
from train import cross_entropy


precision.half = jnp.float32


def filter_trainable(model):
    frozen = [
        model.bert.position_embedding.weight,
    ]
    return lambda leaf: eqx.is_array(leaf) and not any(leaf is f for f in frozen)


def load_mnli(paths, tokenizer, label=True):
    cols = ["sentence1", "sentence2"]
    if label:
        cols.append("gold_label")
    dfs = [pd.read_csv(p, sep="\t", usecols=cols) for p in paths]
    df = pd.concat(dfs, axis=0, ignore_index=True)
    valid = df[(df.sentence1.str.len() < 1000) & (df.sentence2.str.len() < 1000)]
    valid = valid.sample(frac=1)
    sentences = valid[["sentence1", "sentence2"]].values.tolist()
    batch = jnp.asarray([s.ids for s in tokenizer.encode_batch(sentences)])
    if label:
        labels = valid["gold_label"].replace(
            {"neutral": 0, "entailment": 1, "contradiction": 2}
        )
        labels = jnp.asarray(labels.to_numpy())
        return batch, labels
    return batch


def forward(model, x, mask, key, inference=False):
    keys = jrandom.split(key, len(x))
    return jax.vmap(model, (0, 0, 0, None))(x, keys, mask, inference)


@eqx.filter_jit
def loss(diff, static, x, y, mask, key, inference=False):
    model = eqx.combine(diff, static)
    logits = forward(model, x, mask, key, inference=inference)
    accuracy = logits.argmax(-1) == y
    return cross_entropy(logits, y).mean(), accuracy.mean()


def make_key_mask(tokens, pad_token):
    # make padded token keys unaccessible for attention
    mask = tokens != pad_token
    mask = jnp.tile(mask[..., None], tokens.shape[-1])
    return mask.swapaxes(-1, -2)


def reset_dropout_p(model, new_value):
    # Dropout probability is copied from checkpoint. Resetting it manually
    is_dropout = lambda x: isinstance(x, eqx.nn.Dropout)
    get_probs = lambda tree: [
        leaf.p
        # do not traverse inside dropouts
        for leaf in jax.tree_util.tree_leaves(tree, is_leaf=is_dropout)
        # ignore leaves that aren't dropouts
        if is_dropout(leaf)
    ]
    new_values = [new_value] * len(get_probs(model))
    # replace every dropout.p with a new value
    return eqx.tree_at(get_probs, model, new_values)


def classify(cfg, tokenizer, model, pair):
    key = jrandom.PRNGKey(0)  # unused on inference
    batch = jnp.asarray(tokenizer.encode(*pair).ids)[None, :]
    mask = make_key_mask(batch, cfg.data.mask_token)
    predictions = forward(model, batch, mask, key, inference=True)
    probabilities = jax.nn.softmax(predictions[0])
    return {
        label: round(prob.item(), 3)
        for label, prob in sorted(
            zip(["neutral", "entailment", "conctradiction"], probabilities),
            key=lambda label_and_prob: -label_and_prob[1],
        )
    }


@eqx.filter_jit
def step(diff, static, opt, opt_state, x, y, mask, key):
    (loss_value, accuracy), grads = eqx.filter_value_and_grad(loss, has_aux=True)(
        diff, static, x, y, mask, key
    )
    updates, opt_state = opt.update(grads, opt_state, diff)
    diff = eqx.apply_updates(diff, updates)
    return diff, opt_state, loss_value, accuracy


if __name__ == "__main__":
    jnp.set_printoptions(precision=3, suppress=True, linewidth=1000)

    random.seed(420)
    key = jrandom.PRNGKey(420)
    key, bertkey, clfkey = jrandom.split(key, 3)
    cfg = load_config()

    os.makedirs(cfg.training.checkpoint_dir, exist_ok=True)
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
        key=bertkey,
    )

    bert = eqx.tree_deserialise_leaves("checkpoints/run1/model_14000.eqx", bert)
    bert = reset_dropout_p(bert, cfg.tuning.dropout_p)

    model = BertClassifier(bert, num_classes=3, key=clfkey)
    opt = optax.chain(
        optax.clip(cfg.optimizer.clip_value),
        optax.adamw(
            learning_rate=optax.cosine_decay_schedule(
                init_value=cfg.tuning.base_learning_rate,
                decay_steps=cfg.tuning.expected_steps,
            ),
            b1=cfg.optimizer.beta1,
            b2=cfg.optimizer.beta2,
            eps=cfg.optimizer.epsilon,
            weight_decay=cfg.optimizer.decay,
        ),
    )

    diff, static = eqx.partition(model, filter_trainable(model))
    opt_state = opt.init(diff)

    print("Loading data...")
    data_root = Path(cfg.tuning.dataset_dir)
    train_x, train_y = load_mnli([data_root / "train.tsv"], tokenizer)
    val_x, val_y = load_mnli(
        [data_root / "dev_matched.tsv", data_root / "dev_mismatched.tsv"],
        tokenizer,
    )

    run_id = wandb.util.generate_id()
    wandb.init(
        project="MNLI",
        id=run_id,
        config={k: v for nested in cfg.values() for k, v in nested.items()},
        notes=f"Command: {' '.join(sys.argv)} ; id={run_id}",
        mode="online" if cfg.training.wandb else "disabled",
    )

    batch_size = cfg.tuning.batch_size
    for epoch in range(cfg.tuning.epochs):
        wandb.log({"epoch": epoch})
        for x, y in (
            pbar := tqdm(
                zip(batched(train_x, batch_size), batched(train_y, batch_size)),
                desc=f"Training epoch {epoch}",
                total=len(train_x) // batch_size,
            )
        ):
            # key mask to make padding token keys unaccessible
            mask = make_key_mask(x, cfg.data.mask_token)
            key, modelkey = jrandom.split(key)
            diff, opt_state, loss_value, accuracy = step(
                diff, static, opt, opt_state, x, y, mask, modelkey
            )
            assert not jnp.isnan(loss_value)
            pbar.set_postfix({"loss": round(loss_value.item(), 2)})
            wandb.log({"Training loss": loss_value.item()})
            wandb.log({"Training accuracy": accuracy.item()})

        print(f"Checkpointing at epoch {epoch}")
        model = eqx.combine(diff, static)
        model_path = f"{cfg.training.checkpoint_dir}/mnli_model_{epoch}.eqx"
        optim_path = f"{cfg.training.checkpoint_dir}/mnli_optimizer_{epoch}.pkl"
        eqx.tree_serialise_leaves(model_path, model)
        with open(optim_path, "wb") as f:
            pickle.dump(opt_state, f)

        val_loss = []
        val_acc = []
        for x, y in (
            pbar := tqdm(
                zip(batched(val_x, batch_size), batched(val_y, batch_size)),
                desc=f"Validation epoch {epoch}",
                total=len(val_x) // batch_size,
            )
        ):
            mask = make_key_mask(x, cfg.data.mask_token)
            key, modelkey = jrandom.split(key)
            loss_value, accuracy = loss(
                diff, static, x, y, mask, modelkey, inference=True
            )
            assert not jnp.isnan(loss_value)
            val_loss.append(loss_value.item())
            val_acc.append(accuracy.item())
            pbar.set_postfix({"loss": round(loss_value.item(), 2)})
        wandb.log({"Validation loss": sum(val_loss) / len(val_loss)})
        wandb.log({"Validation accuracy": sum(val_acc) / len(val_acc)})

    wandb.finish()


# training rows = 390_998
# training batches = 390_998 // 16 = 122_185
# validation rows = 19_543
# validation batches = 1_221
# Best baseline validation accuracy: 0.77
# Target crammed model validation accuracy: 0.84
