import equinox
import jax
import jax.numpy as jnp
import jax.random as jrand
from equinox import nn
from jax import Array

from .data_loader import TransformerDataLoader


class BigramLanguageModel(equinox.Module):
    token_embedding_table: nn.Embedding
    prng_key: Array

    def __init__(self, vocab_size: int, jrand_seed: int = 0):
        self.prng_key = jrand.PRNGKey(jrand_seed)
        self.token_embedding_table = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_size=vocab_size,
            key=self.prng_key,
        )

    def __call__(self, x: Array) -> Array:
        """
        @param x - model input 1d array

        for each entry in `input`, looks up the embedding for that entry
        ex:
        input [a, b]
        logits [<a_embedding>, <b_embedding>]
        where <*_embedding> is an Array representing next token probabilities
        """

        def f(x: Array) -> Array:
            return jax.vmap(self.token_embedding_table)(x)

        return f(x)


def main() -> None:
    data_loader = TransformerDataLoader("tiny_shakespeare.txt")
    if not data_loader.load_encoding("tiny_shakespeare_encoding.pkl"):
        print("Failed to load encoding")

    xb, yb = data_loader.get_batch()

    blm = BigramLanguageModel(data_loader.vocab_size)

    @equinox.filter_jit
    def compute_loss(model: equinox.Module, x: Array, y: Array) -> float:
        out = jax.vmap(model)(x)  # vmap along batch dimension

        # cross entropy loss implementation
        B, T, C = out.shape
        out = out.reshape((B * T, C))
        targets = y.reshape(B * T)

        # softmax output
        out = jax.nn.softmax(out, axis=-1)

        # python for loop equivalent : y_pred = jnp.array([out[i][targets[i]] for i in range(B * T)])
        y_pred = jnp.take_along_axis(
            out, jnp.expand_dims(targets, axis=1), axis=-1
        ).squeeze(axis=1)

        Hs = -jnp.log(y_pred)

        # reduction = mean
        return jnp.mean(Hs)

    loss = compute_loss(blm, xb, yb)
    print(loss)


if __name__ == "__main__":
    main()
