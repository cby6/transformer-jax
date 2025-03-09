import flax.linen as nn
import haiku as hk
import jax
import jax.numpy as jnp

vocab_size = 10
embed_dim = 64
n_heads = 4

text = ["a", "b", "c", "d", "e", "f"]
text_encoding = [55, 64, 23, 53, 44, 50]

class SelfAttention(hk.Module):
    def __init__(self, vocab_size: int, embed_dim: int, n_heads: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = 1
        self.vocab_size = vocab_size
        self.embed = hk.Embed(vocab_size, embed_dim)
        self.layer_norm = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        self.linear = hk.Linear(self.embed_dim)

    def __call__(self, x):
        seq_length = len(x)
        pos_encoding = self.positional_encoding(seq_length)
        x = self.embed(x) + pos_encoding
        query_size = int(self.embed_dim / self.n_heads)
        w_q_init = hk.initializers.VarianceScaling(scale=1.0, mode="fan_in", distribution="truncated_normal")
        w_k_init = hk.initializers.VarianceScaling(scale=1.0, mode="fan_in", distribution="truncated_normal")
        w_v_init = hk.initializers.VarianceScaling(scale=1.0, mode="fan_in", distribution="truncated_normal")
        w_q = hk.get_parameter("w_q", shape=[self.embed_dim, query_size], init=w_q_init)
        w_k = hk.get_parameter("w_k", shape=[self.embed_dim, query_size], init=w_k_init)
        w_v = hk.get_parameter("w_v", shape=[self.embed_dim, query_size], init=w_v_init)
        x = self.attention(x, w_q, w_k, w_v)
        x = self.linear(x)
        x = x + pos_encoding
        x = self.layer_norm(x)
        return x
    
    def positional_encoding(self, seq_length):
        pos_encoding = jnp.zeros((seq_length, self.embed_dim))
        positions = jnp.arange(seq_length).reshape(-1, 1)
        div_term = jnp.exp(
            jnp.arange(0, self.embed_dim, 2) * -(jnp.log(10000.0) / self.embed_dim)
        )
        odd_indices = jnp.arange(pos_encoding.shape[1]) % 2 == 1
        even_indices = jnp.arange(pos_encoding.shape[1]) % 2 == 0
        pos_encoding = pos_encoding.at[:, even_indices].set(
            jnp.sin(positions * div_term)
        )
        pos_encoding = pos_encoding.at[:, odd_indices].set(
            jnp.cos(positions * div_term)
        )
        return pos_encoding

    def attention(self, x, w_q, w_k, w_v):
        d_k = int(self.embed_dim / self.n_heads)
        q = x @ w_q
        k = x @ w_k
        v = x @ w_v
        score = (q @ jnp.transpose(k, (1, 0))) / jnp.sqrt(d_k)
        att = jax.nn.softmax(score) @ v
        return att


class FeedForward(hk.Module):
    def __init__(self, embed_dim: int, n_heads: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads

    def __call__(self, x):
        x = hk.Linear(self.embed_dim)(x)
        x = jax.nn.relu(x)
        x = hk.Linear(self.embed_dim)(x)
        return x


@hk.transform
def transformer(x):
    self_attention_layer = SelfAttention(
        vocab_size=vocab_size, embed_dim=embed_dim, n_heads=n_heads
    )
    feedforward_layer = FeedForward(
        embed_dim=embed_dim, n_heads=n_heads
    )
    x = self_attention_layer(x)
    x = feedforward_layer(x)
    return x


rng = jax.random.PRNGKey(42)
params = transformer.init(rng, jnp.array(text_encoding))
embeddings = transformer.apply(params, None, jnp.array(text_encoding))

print(embeddings.shape)
