import os
import sys
import glob
import time
import random
import pickle

import jax
import numpy as np
import equinox as eqx
import jax.numpy as jnp
import jax.random as jrandom
import optax
import wandb
from tokenizers import Tokenizer

from io_utils import load_config, load_batch
from model import Bert, precision


def filter_trainable(model):
    frozen = [
        model.position_embedding.weight,
    ]
    return lambda leaf: eqx.is_array(leaf) and not any(leaf is f for f in frozen)


def tree_as_type(tree, dtype):
    return jax.tree_map(
        lambda leaf: leaf.astype(dtype) if eqx.is_inexact_array(leaf) else leaf, tree
    )


def cross_entropy(logits, target):
    return -jnp.sum(
        jax.nn.log_softmax(logits, axis=-1)
        * jax.nn.one_hot(target, num_classes=logits.shape[-1]),
        axis=-1,
    )


def forward(model, batch, key, inference=False):
    keys = jrandom.split(key, len(batch["input"]))
    output = jax.vmap(model, (0, 0, None))(batch["input"], keys, inference)
    return jax.vmap(model.project)(output)


def loss(diff, static, batch, key, return_logits=False, inference=False):
    model = eqx.combine(diff, static)
    logits = forward(model, batch, key, inference)
    tokenwise_loss = cross_entropy(logits.astype(precision.full), batch["target"])
    loss_value = jnp.average(tokenwise_loss, weights=batch["mask"])
    return (loss_value, logits) if return_logits else loss_value


@eqx.filter_jit
def step(diff, static, scale, opt, opt_state, batch, key):
    loss_scaled = lambda *args: loss(*args) * scale
    loss_value, grads = eqx.filter_value_and_grad(loss_scaled)(
        tree_as_type(diff, precision.half),
        tree_as_type(static, precision.half),
        batch,
        key,
    )
    grads_unscaled = jax.tree_map(
        lambda leaf: None if leaf is None else leaf / scale,
        tree_as_type(grads, precision.full),
    )
    updates, opt_state = opt.update(grads_unscaled, opt_state, diff)
    diff = eqx.apply_updates(diff, updates)
    return diff, opt_state, loss_value / scale


@eqx.filter_jit
def infer(diff, static, batch, key, return_logits=False):
    return loss(diff, static, batch, key, return_logits, inference=True)


if __name__ == "__main__":
    jnp.set_printoptions(precision=3, suppress=True, linewidth=1000)

    random.seed(420)
    key = jrandom.PRNGKey(420)
    key, subkey = jrandom.split(key)
    cfg = load_config()
    os.makedirs(cfg.training.checkpoint_dir, exist_ok=True)

    tokenizer = Tokenizer.from_file(cfg.tokenizer.checkpoint_path)

    model = Bert(
        vocab_size=cfg.data.vocab_size,
        sequence_size=cfg.data.sequence_size,
        num_blocks=cfg.model.num_blocks,
        num_attention_heads=cfg.model.num_attention_heads,
        num_features=cfg.model.embedding_size,
        num_feedforward_features=cfg.model.feedforward_inner_size,
        dropout_p=cfg.training.dropout_p,
        key=subkey,
    )
    opt = optax.chain(
        optax.clip(cfg.optimizer.clip_value),
        optax.adamw(
            learning_rate=optax.linear_onecycle_schedule(
                transition_steps=cfg.training.expected_steps,
                peak_value=cfg.optimizer.peak_learning_rate,
                pct_start=cfg.optimizer.warmup_share,
            ),
            b1=cfg.optimizer.beta1,
            b2=cfg.optimizer.beta2,
            eps=cfg.optimizer.epsilon,
            weight_decay=cfg.optimizer.decay,
        ),
    )
    accumulation_rate = (
        cfg.training.peak_batches_accumulated / cfg.training.expected_steps
    )
    opt = optax.MultiSteps(opt, lambda step: step * accumulation_rate)

    files = glob.glob(f"{cfg.data.processed_dir}/*.npy")
    random.shuffle(files)

    batch_size = cfg.training.batch_size
    sample_len = cfg.data.sequence_size * 3 + 3

    if cfg.training.mixed_precision:
        scale = cfg.training.loss_scale
    else:
        scale = 1
        precision.half = jnp.float32

    diff, static = eqx.partition(model, filter_trainable(model))
    opt_state = opt.init(diff)

    wandb.init(
        project="bert",
        config={k: v for nested in cfg.values() for k, v in nested.items()},
        notes=f"Command: {' '.join(sys.argv)}",
        mode="online" if cfg.wandb else "disabled",
    )

    for file in files:
        fp = np.memmap(file, dtype=np.uint16, mode="r").reshape(-1, sample_len)
        for i in range(0, len(fp), batch_size):  # drop the rest
            gradstep = opt_state.gradient_step.item()
            ministep = opt_state.mini_step.item()

            batch = load_batch(fp[i : i + batch_size])
            key, modelkey = jrandom.split(key)
            diff, opt_state, loss_value = step(
                diff, static, scale, opt, opt_state, batch, modelkey
            )
            assert not jnp.isnan(loss_value)
            print(f"{gradstep:5} ({ministep:3}): {loss_value.item():.10}")

            wandb.log({"loss": loss_value.item()})

            if (
                gradstep in [0, 100, 500]
                or gradstep % cfg.training.checkpoint_frequency == 0
            ) and ministep == 0:
                if gradstep == 0:
                    sample_batch = batch

                print(f"Checkpointing at step {gradstep}")
                model = eqx.combine(diff, static)
                model_path = f"{cfg.training.checkpoint_dir}/model_{gradstep}.eqx"
                optim_path = f"{cfg.training.checkpoint_dir}/optimizer_{gradstep}.pkl"
                eqx.tree_serialise_leaves(model_path, model)
                with open(optim_path, "wb") as f:
                    pickle.dump(opt_state, f)

                wandb.save(model_path)
                wandb.save(optim_path)

                key, modelkey = jrandom.split(key)
                sample_pred = jax.vmap(model.predict_greedy, (0, None))(
                    sample_batch["input"], modelkey
                )
                samples = [
                    [tokenizer.decode(item, skip_special_tokens=False) for item in row]
                    for row in zip(
                        sample_batch["input"], sample_pred, sample_batch["target"]
                    )
                ]
                table = wandb.Table(
                    columns=["input", "prediction", "target"],
                    data=[
                        [
                            tokenizer.decode(item, skip_special_tokens=False).replace("[MASK]", "**[MASK]**")
                            for item in row
                        ]
                        for row in zip(sample_batch["input"], sample_pred, sample_batch["target"])
                    ],
                )
                wandb.log({"Sample prediction": table})

    wandb.finish()
