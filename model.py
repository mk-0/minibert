import math
import dataclasses

import jax
import jax.numpy as jnp
import jax.random as jrandom
import equinox as eqx


@dataclasses.dataclass
class Precision:
    full: type
    half: type


precision = Precision(full=jnp.float32, half=jnp.float16)


def full_precision(f):
    """
    Wrapper for (stateless) functions that must enforce full precision for computation
    These are usually aggregation functions (layernorm, softmax)
    Matrix multiplication does this at hardware level with Volta GPUs and newer
    """
    return lambda x, **kwargs: f(x.astype(precision.full), **kwargs).astype(x.dtype)


class TokenEmbedding(eqx.Module):
    weight: jax.Array

    def __init__(self, vocab_size, embedding_size, key):
        super().__init__()
        self.weight = jax.random.normal(key, [vocab_size, embedding_size])

    def __call__(self, tokens):
        return self.weight[tokens]


class PositionEmbedding(eqx.Module):
    """
    Why both sin and cos? To easily attend to relative positions.
    Because this way attending to a constant offset i is a linear transformation:
    sin(pos + i) = sin_i*cos(pos) + cos_i*sin(pos)
    """

    weight: jax.Array

    def __init__(self, max_sequence_size, embedding_size, encoding_period=1_000):
        """
        encoding_period > max_sequence_size allows for (limited) extrapolation during inference
        """
        super().__init__()
        if embedding_size % 2:
            raise ValueError("Embedding size must be even")

        wavelength = jnp.linspace(0, 1, embedding_size // 2, endpoint=False)
        wv_geometric = jnp.power(encoding_period, wavelength)
        angle = jnp.arange(0, max_sequence_size)[..., None] / wv_geometric[None, ...]
        self.weight = jnp.concatenate([jnp.sin(angle), jnp.cos(angle)], axis=1)

    def __call__(self, length):
        return self.weight[:length]


def attention(q, k, v):
    """
    Not using eqx.nn.MultiheadAttention to enable mixed precision for softmax
    Inputs shape: [sequence, embedding]
    """
    logits = q @ k.T / math.sqrt(q.shape[-1])
    weights = full_precision(jax.nn.softmax)(logits, axis=-1)
    return weights @ v


class MultiheadAttention(eqx.Module):
    num_heads: int = eqx.field(static=True)
    num_features: int = eqx.field(static=True)
    q_proj: eqx.nn.Linear
    k_proj: eqx.nn.Linear
    v_proj: eqx.nn.Linear
    o_proj: eqx.nn.Linear

    def __init__(self, num_heads, num_features, key):
        assert (
            num_features % num_heads == 0
        ), f"Can't split {num_features} into {num_heads} heads"
        self.num_heads = num_heads
        self.num_features = num_features
        qk, kk, vk, ok = jrandom.split(key, 4)
        self.q_proj = eqx.nn.Linear(num_features, num_features, use_bias=False, key=qk)
        self.k_proj = eqx.nn.Linear(num_features, num_features, use_bias=False, key=kk)
        self.v_proj = eqx.nn.Linear(num_features, num_features, use_bias=False, key=vk)
        self.o_proj = eqx.nn.Linear(num_features, num_features, use_bias=False, key=ok)

    def __call__(self, q, k, v):
        feat_per_head = self.num_features // self.num_heads
        q = jax.vmap(self.q_proj)(q).reshape(-1, self.num_heads, feat_per_head)
        k = jax.vmap(self.k_proj)(k).reshape(-1, self.num_heads, feat_per_head)
        v = jax.vmap(self.v_proj)(v).reshape(-1, self.num_heads, feat_per_head)
        output = jax.vmap(attention, in_axes=1, out_axes=1)(q, k, v)
        return jax.vmap(self.o_proj)(output.reshape(-1, self.num_features))


class ResNormAttention(eqx.Module):
    attention: eqx.Module
    layernorm: eqx.Module
    dropout: eqx.Module

    def __init__(self, num_heads, num_features, dropout_p, key):
        super().__init__()
        self.attention = MultiheadAttention(num_heads, num_features, key=key)
        self.layernorm = eqx.nn.LayerNorm(num_features)
        self.dropout = eqx.nn.Dropout(dropout_p)

    def __call__(self, x, key, inference=False):
        # https://arxiv.org/abs/2002.04745
        # Normalizing before residual (like in ResNet) improves the gradient flow
        # Otherwise (original Transformer) layernorm distorts residual connections
        z = jax.vmap(full_precision(self.layernorm))(x)
        z = self.attention(z, z, z)
        return x + self.dropout(z, key=key, inference=inference)


class ResNormFeedForward(eqx.Module):
    first: eqx.Module
    second: eqx.Module
    layernorm: eqx.Module
    dropout: eqx.Module

    def __init__(self, io_size, inner_size, dropout_p, key):
        super().__init__()
        k1, k2 = jrandom.split(key)

        # Cramming Bert turns off biases
        self.first = eqx.nn.Linear(io_size, inner_size, use_bias=False, key=k1)
        self.second = eqx.nn.Linear(inner_size, io_size, use_bias=False, key=k2)
        self.layernorm = eqx.nn.LayerNorm(io_size)
        self.dropout = eqx.nn.Dropout(dropout_p)

    def __call__(self, x, key, inference=False):
        # https://arxiv.org/abs/2002.04745
        # Normalizing before residual (like in ResNet) improves the gradient flow
        # Otherwise (original Transformer) layernorm distorts residual connections
        z = full_precision(self.layernorm)(x)
        # Bert uses GELU. Elementwise, but complicated. Full precision just in case
        z = self.second(full_precision(jax.nn.gelu)(self.first(z)))
        return x + self.dropout(z, key=key, inference=inference)


class Block(eqx.Module):
    selfattention: eqx.Module
    feedforward: eqx.Module

    def __init__(
        self,
        num_attention_heads,
        num_features,
        num_feedforward_features,
        dropout_p,
        key,
    ):
        super().__init__()
        k1, k2 = jrandom.split(key)
        self.selfattention = ResNormAttention(
            num_heads=num_attention_heads,
            num_features=num_features,
            dropout_p=dropout_p,
            key=k1,
        )
        self.feedforward = ResNormFeedForward(
            io_size=num_features,
            inner_size=num_feedforward_features,
            dropout_p=dropout_p,
            key=k2,
        )

    def __call__(self, x, key, inference=False):
        k1, k2 = jrandom.split(key)
        x = self.selfattention(x, key=k1, inference=inference)
        # applying feedforward block independently to every position in a sequence
        positionwise_ff = jax.vmap(self.feedforward, (0, 0, None))
        return positionwise_ff(x, jrandom.split(k2, x.shape[0]), inference)


class Bert(eqx.Module):
    """
    https://arxiv.org/abs/1608.05859 (2016) recommends shared i/o embeddings
    https://arxiv.org/abs/1706.03762 (2017) (Attention Is All You Need) uses the idea
    https://arxiv.org/abs/2010.12821 (2020) insists on decoupling them instead
    https://arxiv.org/abs/2212.14034 (2022) (Cramming BERT) finds no improvements from decoupling
    So we keep input and generator embeddings shared
    """

    token_embedding: eqx.Module
    position_embedding: eqx.Module
    dropout: eqx.Module
    blocks: list
    initial_layernorm: eqx.Module
    final_layernorm: eqx.Module
    output_temperature: jax.Array

    def __init__(
        self,
        vocab_size,
        sequence_size,
        num_blocks,
        num_attention_heads,
        num_features,
        num_feedforward_features,
        dropout_p,
        key,
    ):
        super().__init__()
        k1, k2 = jrandom.split(key)
        self.token_embedding = TokenEmbedding(vocab_size, num_features, key=k1)
        self.position_embedding = PositionEmbedding(sequence_size, num_features)
        # Cramming Bert includes layernorm after embedding
        self.initial_layernorm = eqx.nn.LayerNorm(num_features)
        self.dropout = eqx.nn.Dropout(dropout_p)
        self.blocks = [
            Block(
                num_attention_heads=num_attention_heads,
                num_features=num_features,
                num_feedforward_features=num_feedforward_features,
                dropout_p=dropout_p,
                key=k,
            )
            for k in jrandom.split(k2, num_blocks)
        ]
        # https://arxiv.org/abs/2002.04745 recommends additional layernorm before output
        self.final_layernorm = eqx.nn.LayerNorm(num_features)
        # Divide output embedding layers by sqrt(d_model)
        self.output_temperature = math.sqrt(num_features)

    def __call__(self, x, key, inference=False):
        """
        Why is it ok to add positional embedding to token embedding
         instead of concatenating them to ensure that they are orthogonal?
        https://www.reddit.com/r/MachineLearning/comments/cttefo/comment/exs7d08
        Self-attention weight is computed as (Qx)'(Kx) = x'(Q'K)x
        Q'K tells how much attention to pay to x' given x
        (Q(t+p))'(K(t+p)) = t'(Q'K)p + t'(Q'K)t + p'(Q'K)t + p'(Q'K)p
        i.e. the same matrix now simply considers more factors
        Also they are probably almost orthogonal anyway
        """
        k1, k2 = jrandom.split(key)

        x = self.token_embedding(x) + self.position_embedding(x.shape[0])
        x = jax.vmap(full_precision(self.initial_layernorm))(x)
        x = self.dropout(x, key=k1, inference=inference)

        for block, bkey in zip(self.blocks, jrandom.split(k2, len(self.blocks))):
            x = block(x, key=bkey, inference=inference)
        return jax.vmap(full_precision(self.final_layernorm))(x)

    def project(self, x):
        # thinking of the final pre-softmax projection as the output embedding
        # and tying it to the input embedding (same weights, transposed)
        return x @ self.token_embedding.weight.T / self.output_temperature

    def predict_greedy(self, sample, key):
        return self.project(self(sample, key=key, inference=True)).argmax(-1)


if __name__ == "__main__":
    jnp.set_printoptions(precision=3, suppress=True, linewidth=1000)
    key = jrandom.PRNGKey(0)
