import os
import random
import pickle

import jax
import numpy as np
import jax.numpy as jnp
import jax.random as jrandom
import equinox as eqx
import optax
import pytest

from tokenizers import Tokenizer

from model import (
    precision,
    TokenEmbedding,
    PositionEmbedding,
    attention,
    MultiheadAttention,
    ResNormAttention,
    ResNormFeedForward,
    Block,
    Bert,
)

from io_utils import (
    MmapWriter,
    get_dataset_size,
    get_mask_fn,
    encode_id,
    decode_id,
    pack_article,
    load_batch,
)
from train import tree_as_type, filter_trainable, cross_entropy, loss, step


random.seed(420)


@pytest.fixture
def getkey():
    return lambda: jrandom.PRNGKey(random.randint(0, 100_000))


@pytest.mark.skipif(
    not os.path.exists("tokenizer.json"), reason="Trained tokenizer not found"
)
def test_trained_tokenizer():
    t = Tokenizer.from_file("tokenizer.json")

    assert t.decode(t.encode("").ids) == ""
    assert t.decode(t.encode("   trim    whitespaces   ").ids) == "trim whitespaces"
    # assert t.decode(t.encode("but keep \n linebreaks").ids) == "but keep\nlinebreaks"  # TODO
    assert t.decode(t.encode("LowerCase").ids) == "lowercase"
    assert t.decode(t.encode("keep punctuation?!").ids) == "keep punctuation?!"
    assert t.decode(t.encode("remove 彁 weird").ids) == "remove weird"
    assert t.decode(t.encode("split numbers 911").ids) == "split numbers 9 1 1"
    assert t.decode(t.encode("UnreasonablyLongWordsAreProbablyBroken" * 10).ids) == ""


@pytest.mark.parametrize("dtype", [np.uint8, np.int64])
def test_mmap_writer(tmp_path, dtype):
    # writing to one file
    path0 = tmp_path / "0"
    path0.mkdir()
    writer = MmapWriter(root_dir=path0, file_shape=(10, 1), dtype=dtype)
    stream = np.arange(10, 20, dtype=dtype)[:, None]
    for sample in stream:
        chunk = sample[None, :]
        writer.write(chunk)

    output = np.memmap(path0 / "0.npy", dtype=dtype, shape=(10, 1))
    assert (stream == output).all()

    # padding with zeroes
    path1 = tmp_path / "1"
    path1.mkdir()
    writer = MmapWriter(root_dir=path1, file_shape=(10, 1), dtype=dtype)
    chunk = np.array([[4]])
    writer.write(chunk)
    output = np.memmap(path1 / "0.npy", dtype=dtype, shape=(10, 1))
    assert len(output) == 10
    assert (output[0] == chunk).all()
    assert (output[1:] == 0).all()

    # rotating files
    path2 = tmp_path / "2"
    path2.mkdir()
    writer = MmapWriter(root_dir=path2, file_shape=(10, 1), dtype=dtype)
    stream = np.arange(1, 95)[:, None]
    for sample in stream:
        chunk = sample[None, :]
        writer.write(chunk)

    assert len(list(path2.iterdir())) == 10
    output = np.memmap(path2 / "8.npy", dtype=dtype, shape=(10, 1))
    assert (output == np.arange(81, 91)[:, None]).all()

    # all-zero rows prohibited
    path3 = tmp_path / "3"
    path3.mkdir()
    writer = MmapWriter(root_dir=path3, file_shape=(10, 3), dtype=dtype)
    writer.write(np.array([[0, 0, 1]]))
    writer.write(np.array([[-1, 0, 0]]))
    with pytest.raises(ValueError):
        writer.write(np.array([[0, 0, 0]]))


@pytest.mark.parametrize("dtype", [np.int8, np.int64])
def test_get_dataset_size(tmp_path, dtype):
    # one file
    path0 = tmp_path / "0"
    path0.mkdir()
    writer = MmapWriter(root_dir=path0, file_shape=(10, 2), dtype=dtype)
    stream = np.arange(-1, 4, dtype=dtype)
    stream = np.stack([stream, np.ones_like(stream)], axis=1)
    for sample in stream:
        chunk = sample[None, :]
        writer.write(chunk)

    assert get_dataset_size(list(path0.iterdir()), row_size=2, dtype=dtype) == 5

    # many files
    path1 = tmp_path / "1"
    path1.mkdir()
    writer = MmapWriter(root_dir=path1, file_shape=(10, 2), dtype=dtype)
    stream = np.arange(-1, 32, dtype=dtype)
    stream = np.stack([stream, np.ones_like(stream)], axis=1)
    for sample in stream:
        chunk = sample[None, :]
        writer.write(chunk)

    assert get_dataset_size(list(path1.iterdir()), row_size=2, dtype=dtype) == 33


# Native vectorization
@pytest.mark.parametrize("x", [np.arange(20), np.arange(60).reshape(3, 20)])
def test_mask(x):
    mask_token = 77
    vocab_size = x.shape[-1]

    # Do nothing
    batch = get_mask_fn(mask_token, vocab_size, mask_p=0, random_p=0, keep_p=0)(x)
    assert (batch["target"] == x).all()
    assert (batch["input"] == x).all()
    assert (batch["mask"] == False).all()

    # Mask everything
    batch = get_mask_fn(mask_token, vocab_size, mask_p=1, random_p=0, keep_p=0)(x)
    assert (batch["target"] == x).all()
    assert (batch["input"] == 77).all()
    assert (batch["mask"] == True).all()

    # Randomize everything
    batch = get_mask_fn(mask_token, vocab_size, mask_p=0, random_p=1, keep_p=0)(x)
    assert (batch["target"] == x).all()
    assert (batch["input"] != x).any()
    assert (batch["mask"] == True).all()

    # Keep everything (but mark as active for loss)
    batch = get_mask_fn(mask_token, vocab_size, mask_p=0, random_p=0, keep_p=1)(x)
    assert (batch["target"] == x).all()
    assert (batch["input"] == x).all()
    assert (batch["mask"] == True).all()

    # Invalid probabilities
    with pytest.raises(ValueError):
        get_mask_fn(mask_token, vocab_size, mask_p=0.6, random_p=0.4, keep_p=0.2)


@pytest.mark.parametrize("aid", [0, -150, np.iinfo(np.uint16).max, 1_999_403_492])
def test_id_encoding(aid):
    assert max(encode_id(aid)) <= np.iinfo(np.uint16).max
    assert decode_id(*encode_id(aid)) == aid


@pytest.mark.parametrize("batch_size", [1, 10])
def test_packing(batch_size):
    batch = {
        "input": np.random.randint(0, 65_000, batch_size * 20).reshape(batch_size, 20),
        "target": np.random.randint(0, 65_000, batch_size * 20).reshape(batch_size, 20),
        "mask": np.random.randint(0, 1, batch_size * 20).reshape(batch_size, 20),
    }
    chunk = np.concatenate((batch["input"], batch["target"], batch["mask"]), axis=1)
    assert (batch["input"] == load_batch(pack_article(chunk, aid=0))["input"]).all()
    assert (batch["target"] == load_batch(pack_article(chunk, aid=325))["target"]).all()
    assert (batch["mask"] == load_batch(pack_article(chunk, aid=154439))["mask"]).all()
    assert load_batch(pack_article(chunk, aid=11))["meta"].shape == (batch_size, 3)


def test_token_embedding(getkey):
    # Shape sanity
    assert TokenEmbedding(1, 1, key=getkey()).weight.shape == (1, 1)
    assert TokenEmbedding(11, 13, key=getkey()).weight.shape == (11, 13)

    # Distinct and deterministic
    t = TokenEmbedding(vocab_size=2, embedding_size=5, key=getkey())
    assert (t(jnp.array(0)) == t(jnp.array(0))).all()
    assert (t(jnp.array(0)) != t(jnp.array(1))).any()

    # Native vectorization
    out = t(jnp.array([[0, 1], [0, 1]]))
    assert (out[0] == out[1]).all()
    assert (out[:, 0] != out[:, 1]).any()

    with pytest.raises(TypeError):
        t(jnp.array(1.5))


def test_position_embedding():
    # Shape sanity
    assert PositionEmbedding(1, 2).weight.shape == (1, 2)
    assert PositionEmbedding(11, 14).weight.shape == (11, 14)

    # Odd embedding size prohibited
    with pytest.raises(ValueError):
        PositionEmbedding(max_sequence_size=2, embedding_size=3)

    p = PositionEmbedding(max_sequence_size=4, embedding_size=8)

    # Distinct and deterministic
    assert (p(3) == p(4)[:3]).all()
    out = p(4)
    assert (out[0] != out[1]).any()
    assert (out[1] != out[2]).any()

    # Bound
    assert jnp.abs(PositionEmbedding(100, 100)(100)).max() <= 1


def test_feedforward(getkey):
    # Deterministic without dropout
    m = ResNormFeedForward(io_size=5, inner_size=10, dropout_p=0.0, key=getkey())
    m = jax.vmap(m, (0, None))
    x = jrandom.uniform(getkey(), (3, 5))
    assert (m(x, getkey()) == m(x, getkey())).all()

    # Residuals must be the same if the inputs are the same up to normalization
    x2 = jnp.arange(15).reshape(3, 5) - 10
    out = m(x2, getkey())
    residual = out - x2
    assert out[0] != pytest.approx(out[1], abs=1e-5)
    assert residual[0] == pytest.approx(residual[1], abs=1e-5)

    # Dropout applied correctly
    m2 = ResNormFeedForward(io_size=5, inner_size=10, dropout_p=0.5, key=getkey())
    m2 = jax.vmap(m2, (0, None, None))  # (x, key, inference)
    assert (m2(x, getkey(), False) != m2(x, getkey(), False)).any()
    assert (m2(x, getkey(), True) == m2(x, getkey(), True)).all()


@pytest.mark.parametrize("shape", [[1, 1], [14, 2], [10, 1]])
def test_attention(shape, getkey):
    mask_shape = [shape[0], shape[0]]
    # Same output as eqx.attention
    ground_truth = eqx.nn._attention.dot_product_attention
    q = jrandom.uniform(getkey(), minval=-1, maxval=1, shape=shape)
    k = jrandom.uniform(getkey(), minval=-1, maxval=1, shape=shape)
    v = jrandom.uniform(getkey(), minval=-1, maxval=1, shape=shape)
    assert attention(q, k, v) == pytest.approx(ground_truth(q, k, v))

    # Same vmap behaviour
    q = jrandom.uniform(getkey(), minval=-1, maxval=1, shape=[4, *shape])
    k = jrandom.uniform(getkey(), minval=-1, maxval=1, shape=[4, *shape])
    v = jrandom.uniform(getkey(), minval=-1, maxval=1, shape=[4, *shape])
    assert jax.vmap(attention)(q, k, v) == pytest.approx(
        jax.vmap(ground_truth)(q, k, v), abs=1e-6
    )

    # Same masking behaviour
    ground_truth = eqx.nn._attention.dot_product_attention
    q = jrandom.uniform(getkey(), minval=-1, maxval=1, shape=shape)
    k = jrandom.uniform(getkey(), minval=-1, maxval=1, shape=shape)
    v = jrandom.uniform(getkey(), minval=-1, maxval=1, shape=shape)
    mask = jrandom.uniform(getkey(), minval=0, maxval=1, shape=mask_shape) > 0.5
    assert attention(q, k, v, mask) == pytest.approx(
        ground_truth(q, k, v, mask), abs=1e-6
    )


@pytest.mark.parametrize("shape", [[14, 2], [10, 1], [17, 17]])
def test_attention_mask(shape, getkey):
    mask_shape = [shape[0], shape[0]]
    q = jrandom.uniform(getkey(), minval=-1, maxval=1, shape=shape)
    k = jrandom.uniform(getkey(), minval=-1, maxval=1, shape=shape)
    v = jrandom.uniform(getkey(), minval=-1, maxval=1, shape=shape)

    # Masked out keys have no effect on output
    k1 = jrandom.uniform(getkey(), minval=-1, maxval=1, shape=shape)
    # only the last key is blocked from attention
    mask = jnp.ones(mask_shape, dtype=jnp.bool_).at[:, -1].set(False)
    k2 = k1.at[-1].set(k[-1])
    assert (k1[-1] != k2[-1]).all()
    assert (attention(q, k1, v, mask) == attention(q, k2, v, mask)).all()
    assert (attention(q, k1, v, mask) != attention(q, k, v, mask)).any()

    # Masked out queries have no effect on output
    q1 = jrandom.uniform(getkey(), minval=-1, maxval=1, shape=shape)
    # only the last query is blocked from attention
    mask = jnp.ones(mask_shape, dtype=jnp.bool_).at[-1].set(False)
    q2 = q1.at[-1].set(q[-1])
    assert (q1[-1] != q2[-1]).all()
    assert (attention(q1, k, v, mask) == attention(q2, k, v, mask)).all()
    assert (attention(q1, k, v, mask) != attention(q, k, v, mask)).any()


@pytest.mark.parametrize("shape", [[2, 2], [2, 14], [8, 2]])
def test_multihead_attention(shape, getkey):
    mask_shape = [shape[0], shape[0]]
    # Same output as eqx.MultiheadAttention
    key = getkey()
    attention = MultiheadAttention(num_heads=2, num_features=shape[-1], key=key)
    ground_truth = eqx.nn.MultiheadAttention(num_heads=2, query_size=shape[-1], key=key)
    q = jrandom.uniform(getkey(), minval=-1, maxval=1, shape=shape)
    k = jrandom.uniform(getkey(), minval=-1, maxval=1, shape=shape)
    v = jrandom.uniform(getkey(), minval=-1, maxval=1, shape=shape)
    assert attention(q, k, v) == pytest.approx(ground_truth(q, k, v), abs=1e-6)

    # Same masking behaviour
    mask = jrandom.uniform(getkey(), minval=0, maxval=1, shape=mask_shape) < 0.5
    assert attention(q, k, v, mask) == pytest.approx(
        ground_truth(q, k, v, mask), abs=1e-6
    )


def test_selfattention(getkey):
    # Deterministic without dropout
    a = ResNormAttention(num_heads=2, num_features=4, dropout_p=0.0, key=getkey())
    x = jrandom.uniform(getkey(), (3, 4))
    assert (a(x, key=getkey()) == a(x, key=getkey())).all()

    # Residuals must be the same if the inputs are the same up to normalization
    x2 = jnp.arange(12).reshape(3, 4) - 6
    out = a(x2, getkey())
    residual = out - x2
    assert out[0] != pytest.approx(out[1], abs=1e-5)
    assert residual[0] == pytest.approx(residual[1], abs=1e-5)

    # Identity layer if value projection is zero
    a2 = eqx.tree_at(
        lambda a: a.attention.v_proj.weight,
        a,
        jnp.zeros_like(a.attention.v_proj.weight),
    )
    assert (x == a2(x, getkey())).all()

    # Dropout applied correctly
    a3 = ResNormAttention(2, 4, 0.5, key=getkey())
    assert (a3(x, getkey()) != a3(x, getkey())).any()
    assert (a3(x, getkey(), inference=True) == a3(x, getkey(), inference=True)).all()


def test_block(getkey):
    # Deterministic without dropout
    b = Block(
        num_attention_heads=2,
        num_features=8,
        num_feedforward_features=16,
        dropout_p=0.0,
        key=getkey(),
    )
    x = jrandom.uniform(getkey(), (3, 8))
    assert (b(x, getkey()) == b(x, getkey())).all()

    # Dropout applied correctly
    b2 = Block(
        num_attention_heads=2,
        num_features=8,
        num_feedforward_features=16,
        dropout_p=0.5,
        key=getkey(),
    )
    assert (b2(x, getkey()) != b2(x, getkey())).any()
    assert (b2(x, getkey(), inference=True) == b2(x, getkey(), inference=True)).all()


def test_bert(getkey):
    # Deterministic without dropout
    b = Bert(
        vocab_size=20,
        sequence_size=20,
        num_blocks=3,
        num_attention_heads=2,
        num_features=16,
        num_feedforward_features=16,
        dropout_p=0.0,
        key=getkey(),
    )
    x = jrandom.randint(getkey(), (10,), minval=0, maxval=20)
    assert (b(x, getkey()) == b(x, getkey())).all()

    # Temperature is correct
    assert b.output_temperature == 4

    # Dropout applied correctly
    b2 = Bert(
        vocab_size=20,
        sequence_size=20,
        num_blocks=3,
        num_attention_heads=2,
        num_features=16,
        num_feedforward_features=16,
        dropout_p=0.5,
        key=getkey(),
    )
    assert (b2(x, getkey()) != b2(x, getkey())).any()
    assert (b2(x, getkey(), inference=True) == b2(x, getkey(), inference=True)).all()

    # JIT compilation works. Quite some error is expected due to XLA optimizations
    keys = jrandom.split(getkey(), 3)
    x = jrandom.randint(getkey(), (3, 10), minval=0, maxval=20)
    assert jax.vmap(b2)(x, keys) == pytest.approx(
        jax.jit(jax.vmap(b2))(x, keys), abs=1e-3
    )


def test_crossentropy():
    target = jnp.array([2, 1, 2, 0])
    logits = jnp.array(
        [
            [100.0, 2, -9],  # confidently wrong, see https://youtu.be/HQwxUAF_YCQ
            [5, 100, 8],  # confidently correct
            [7, 7, 7],  # unsure
            [-7, -7, -7],  # exactly the same thing
        ]
    )
    out = cross_entropy(logits, target)
    assert out[1] == 0
    assert out[2] == out[3]
    assert out[1] < out[2] < out[0]

    # Native vectorization
    assert (jax.vmap(cross_entropy)(logits, target) == out).all()


def test_gradient_mask(getkey):
    model = Bert(
        vocab_size=20,
        sequence_size=20,
        num_blocks=3,
        num_attention_heads=2,
        num_features=16,
        num_feedforward_features=16,
        dropout_p=0.0,  # deterministic
        key=getkey(),
    )
    diff, static = eqx.partition(model, filter_trainable(model))

    # Masked inputs do not contribute to loss or gradient
    mask_one = jnp.stack([jnp.zeros(10), jnp.ones(10)], axis=0)
    mask_none = jnp.ones(20).reshape(2, 10)
    inputs1 = jnp.stack(
        [jrandom.randint(getkey(), shape=[10], minval=0, maxval=10), jnp.arange(10)],
        axis=0,
    )
    inputs2 = jnp.stack(
        [jrandom.randint(getkey(), shape=[10], minval=0, maxval=10), jnp.arange(10)],
        axis=0,
    )
    value1, grad1 = eqx.filter_value_and_grad(loss)(
        diff, static, {"input": inputs1, "target": inputs1, "mask": mask_one}, getkey()
    )
    value2, grad2 = eqx.filter_value_and_grad(loss)(
        diff, static, {"input": inputs2, "target": inputs2, "mask": mask_one}, getkey()
    )
    value3, grad3 = eqx.filter_value_and_grad(loss)(
        diff, static, {"input": inputs2, "target": inputs2, "mask": mask_none}, getkey()
    )
    assert value1 == value2
    assert grad1 == grad2
    assert value2 != value3
    assert grad2 != grad3


def test_checkpoint(getkey, tmp_path):
    precision.half = jnp.float32  # turn off mixed precision

    model1 = Bert(
        vocab_size=20,
        sequence_size=20,
        num_blocks=3,
        num_attention_heads=2,
        num_features=16,
        num_feedforward_features=16,
        dropout_p=0.5,
        key=getkey(),
    )
    opt = optax.adam(0.001)
    batch = {
        "input": jnp.arange(20).reshape(2, 10),
        "target": jnp.arange(20).reshape(2, 10),
        "mask": jnp.ones(20).reshape(2, 10),
    }
    diff1, static = eqx.partition(model1, filter_trainable(model1))
    opt_state1 = opt.init(diff1)
    diff2, opt_state2, loss_value = step(
        diff1, static, 1, opt, opt_state1, batch, key=getkey()
    )
    model2 = eqx.combine(diff2, static)

    # Computes
    assert not jnp.isnan(loss_value)

    # Optimizer state changed
    assert opt_state1 != opt_state2

    # Hyperparameters unchanged, learnable parameters changed
    params1, static1 = eqx.partition(model1, eqx.is_array)
    params2, static2 = eqx.partition(model2, eqx.is_array)
    assert static1 == static2
    assert params1 != params2
    assert (model1.token_embedding.weight != model2.token_embedding.weight).any()
    assert (model1.position_embedding.weight == model2.position_embedding.weight).all()
    assert model1.output_temperature == model2.output_temperature

    # Model serialization recovered correctly
    eqx.tree_serialise_leaves(tmp_path / "model.eqx", model2)
    model3 = eqx.tree_deserialise_leaves(tmp_path / "model.eqx", like=model1)
    params3, static3 = eqx.partition(model3, eqx.is_array)
    assert static2 == static3
    assert params2 == params3

    # Optimizer serialization recovered correctly
    with open(tmp_path / "optimizer.pkl", "wb") as f:
        pickle.dump(opt_state2, f)

    with open(tmp_path / "optimizer.pkl", "rb") as f:
        opt_state3 = pickle.load(f)

    assert opt_state2 == opt_state3

    precision.half = jnp.float16


def test_mixed_precision(getkey):
    iter_leaves = jax.tree_util.tree_leaves  # annoyngly long
    batch = {
        "input": jnp.arange(20).reshape(2, 10),
        "target": jnp.arange(20).reshape(2, 10),
        "mask": jnp.ones(20).reshape(2, 10),
    }

    model = Bert(
        vocab_size=20,
        sequence_size=20,
        num_blocks=3,
        num_attention_heads=2,
        num_features=16,
        num_feedforward_features=16,
        dropout_p=0,  # deterministic
        key=getkey(),
    )
    diff, static = eqx.partition(model, filter_trainable(model))

    scale = 8192

    # Half-precision gradients in half precision mode
    precision.half = jnp.float16
    loss_value_half, grads_half = eqx.filter_value_and_grad(
        lambda *args: loss(*args) * scale
    )(
        tree_as_type(diff, precision.half),
        tree_as_type(static, precision.half),
        batch,
        getkey(),
    )
    assert all(leaf.dtype == jnp.float16 for leaf in iter_leaves(grads_half))
    assert all(jnp.isfinite(leaf).all() for leaf in iter_leaves(grads_half))
    assert jnp.isfinite(loss_value_half).all()

    # Full-precision gradients in full precision mode
    precision.half = jnp.float32
    loss_value_full, grads_full = eqx.filter_value_and_grad(loss)(
        tree_as_type(diff, precision.half),
        tree_as_type(static, precision.half),
        batch,
        getkey(),
    )
    assert all(leaf.dtype == jnp.float32 for leaf in iter_leaves(grads_full))
    assert all(jnp.isfinite(leaf).all() for leaf in iter_leaves(grads_full))
    assert jnp.isfinite(loss_value_full).all()

    # Values equal up to loss scale
    loss_value_unscaled = loss_value_half / scale
    grads_unscaled = jax.tree_map(
        lambda leaf: leaf / scale, tree_as_type(grads_half, precision.full)
    )
    grad_pairs = zip(iter_leaves(grads_unscaled), iter_leaves(grads_full))

    assert max(jnp.abs(x - y).max() for x, y in grad_pairs) < 1e-4
    assert loss_value_full == pytest.approx(loss_value_unscaled, abs=1e-4)
