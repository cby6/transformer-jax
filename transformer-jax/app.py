import haiku as hk
import jax
import jax.numpy as jnp

vocab_size = 10
embed_dim = 64
n_heads = 4

text = ['a', 'b', 'c', 'd', 'e', 'f']
text_encoding = [55, 64, 23, 53, 44, 50]

class EmbedModel(hk.Module):
    def __init__(self, vocab_size: int, embed_dim: int, n_heads: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.embed = hk.Embed(vocab_size, embed_dim)

    def __call__(self, indices):
        return self.embed(indices) + self.positional_encoding(indices)
    
    def positional_encoding(self, text_encoding):
        seq_length = len(text_encoding)
        pos_encoding = jnp.zeros((seq_length, self.embed_dim))
        positions = jnp.arange(seq_length).reshape(-1, 1)
        div_term = jnp.exp(jnp.arange(0, self.embed_dim, 2) * -(jnp.log(10000.0) / self.embed_dim))
        odd_indices = jnp.arange(pos_encoding.shape[1]) % 2 == 1
        even_indices = jnp.arange(pos_encoding.shape[1]) % 2 == 0
        pos_encoding = pos_encoding.at[:, even_indices].set(jnp.sin(positions * div_term))
        pos_encoding = pos_encoding.at[:, odd_indices].set(jnp.cos(positions * div_term))
        return pos_encoding 
    
    def attention(self, pos_encoding, w_q, w_k, w_v):
        d_k = int(self.embed_dim / self.n_heads)
        q = pos_encoding @ w_q
        k = pos_encoding @ w_k
        v = pos_encoding @ w_v
        score = (q @ jnp.transpose(k, (0,2,1))) / jnp.sqrt(d_k)
        att = jax.nn.softmax(score) @ v
        return att
    
def forward(indices):
    model = EmbedModel(vocab_size=vocab_size, embed_dim=embed_dim)
    return model(indices)

embed_fn = hk.transform(forward)

rng = jax.random.PRNGKey(42)
params = embed_fn.init(rng, jnp.array(text_encoding))
embeddings = embed_fn.apply(params, None, jnp.array(text_encoding))

print(embeddings.shape)

