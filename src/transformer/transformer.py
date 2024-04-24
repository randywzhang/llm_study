import equinox
import jax
import jax.numpy as jnp
import jax.random as jrand
import optax
from equinox import nn
from jax import Array

from .data_loader import BLOCK_SIZE, TransformerDataLoader, get_bespoke_tokenizer

JRAND_SEED = 0

N_EMBED = 32
HEAD_SIZE = 16


class AttentionHead(equinox.Module):
    head_size: int
    block_size: int  # T

    key: nn.Linear
    query: nn.Linear
    value: nn.Linear

    tril: Array

    def __init__(
        self,
        n_embed: int = N_EMBED,
        head_size: int = HEAD_SIZE,
        block_size: int = BLOCK_SIZE,  # T
        jrand_seed: int = JRAND_SEED,
    ):
        layer_prng_key = jrand.PRNGKey(jrand_seed)

        self.head_size = head_size
        self.block_size = block_size

        layer_prng_key, subkey = jrand.split(layer_prng_key)
        self.key = nn.Linear(
            in_features=n_embed,
            out_features=head_size,
            use_bias=False,
            key=subkey,
        )

        layer_prng_key, subkey = jrand.split(layer_prng_key)
        self.query = nn.Linear(
            in_features=n_embed,
            out_features=head_size,
            use_bias=False,
            key=subkey,
        )

        layer_prng_key, subkey = jrand.split(layer_prng_key)
        self.value = nn.Linear(
            in_features=n_embed,
            out_features=head_size,
            use_bias=False,
            key=subkey,
        )

        self.tril = jnp.tril(jnp.ones((block_size, block_size)))

    def __call__(self, x: Array) -> Array:
        # obtain keys and queries for each token in the context, where
        # keys hold the information of the embedding and queries ask for
        # the information the embedding requires for context
        k = jax.vmap(self.key)(x)  # (T, head_size)
        q = jax.vmap(self.query)(x)  # (T, head_size)

        # obtain a value vector for each token
        v = jax.vmap(self.value)(x)  # (T, head_size)

        # obtain context matrix for relationships between embeddings
        # by determining similarities between each embedding's query
        # and the other embedding's keys (are you what I am asking for)
        # Done via dot product of queries and keys.
        # multiply the result by 1/sqrt(head_size) to preserve variance
        context_mat = q @ k.T * self.head_size**-0.5  # (T, T)

        # --DECODER LOGIC--
        # mask the weight matrix so that embeddings only look for context
        # from previous embeddings
        context_mat = jnp.where(self.tril == 0, -jnp.inf, context_mat)
        # --END DECODER LOGIC--

        # apply softmax to context weights
        context_mat = jax.nn.softmax(context_mat, axis=-1)

        # compute the output to the attention head by multiplying the
        # context weights by the input value.
        return context_mat @ v  # (T, head_size)


# TODO: needs work, don't like the for loops
# use vmap instead like in nn.MultiheadAttention._project
class MultiHeadAttention(equinox.Module):

    def __init__(
        self,
        num_heads: int,
        n_embed: int = N_EMBED,
        head_size: int = HEAD_SIZE,
        block_size: int = BLOCK_SIZE,
        jrand_seed: int = JRAND_SEED,
    ):
        self.heads = [
            AttentionHead(n_embed, head_size, block_size, jrand_seed)
            for _ in range(num_heads)
        ]

    def __call__(self, x: Array) -> Array:
        return jnp.concat([h(x) for h in self.heads], axis=-1)


class Transformer(equinox.Module):
    token_embedding_table: nn.Embedding
    position_embedding_table: nn.Embedding
    sa_head: AttentionHead  # self_attention_head
    lm_head: nn.Linear  # language_model_head

    block_size: int  # T

    def __init__(
        self,
        vocab_size: int,  # C_vocab
        n_embed: int = N_EMBED,  # C_embed
        block_size: int = BLOCK_SIZE,  # T
        head_size: int = HEAD_SIZE,
        jrand_seed: int = JRAND_SEED,
    ):
        layer_prng_key = jrand.PRNGKey(jrand_seed)

        layer_prng_key, subkey = jrand.split(layer_prng_key)
        self.token_embedding_table = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_size=n_embed,
            key=subkey,
        )

        layer_prng_key, subkey = jrand.split(layer_prng_key)
        self.position_embedding_table = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_size=n_embed,
            key=subkey,
        )

        self.sa_head = AttentionHead(n_embed, head_size, block_size, jrand_seed)

        layer_prng_key, subkey = jrand.split(layer_prng_key)
        self.lm_head = nn.Linear(
            in_features=head_size,
            out_features=vocab_size,
            key=subkey,
        )

        self.block_size = block_size

    def generate_token(self, x: Array, prng_key: Array) -> int:
        """
        @param x - model input context 1d array

        generates the next token index given the input context
        """
        logits = self.__call__(x)

        # equivalent to torch.multinomial(softmax(logits, axis=-1), num_samples = 1)
        return jax.random.categorical(prng_key, logits)

    def __call__(self, x: Array) -> Array:
        """
        @param x - model input context
        """
        # compute embedding
        tok_emb = jax.vmap(self.token_embedding_table)(x)  # (T, C_embed)
        pos_emb = jax.vmap(self.position_embedding_table)(
            jnp.arange(self.block_size)
        )  # (T, C_embed)
        emb = tok_emb + pos_emb

        # feed embedding through attention head
        emb = self.sa_head(emb)  # (T, head_size)

        logits = jax.vmap(self.lm_head)(emb)  # (T, C_vocab)

        return logits


@equinox.filter_value_and_grad
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

    # mean reduction
    return jnp.mean(Hs)


@equinox.filter_jit
def make_step(
    model: equinox.Module,
    batch_input: Array,
    batch_expected: Array,
    opt_state: optax.OptState,
    opt_update: optax.TransformUpdateFn,
):
    loss, grads = compute_loss(model, batch_input, batch_expected)
    updates, opt_state = opt_update(grads, opt_state)
    model = equinox.apply_updates(model, updates)
    return loss, model, opt_state


NUM_NEW_TOKENS = 100


def generate_tokens(
    model: equinox.Module, num_new_tokens: int = NUM_NEW_TOKENS, seed: int = 0
) -> list[int]:
    prng_key = jrand.PRNGKey(seed)
    generated_tokens = []
    next_token = jnp.array([0])
    print(next_token.shape)
    for _ in range(num_new_tokens):
        prng_key, subkey = jrand.split(prng_key)
        next_token = model.generate_token(next_token, subkey)
        generated_tokens.append(next_token[0].item())

    return generated_tokens


# Train the model
def main() -> None:
    data_loader = TransformerDataLoader("tiny_shakespeare.txt")
    if not data_loader.load_encoding("tiny_shakespeare_encoding.pkl"):
        print("Failed to load encoding")

    # tokenizer = get_bespoke_tokenizer(
    #     raw_text=data_loader.raw_text, load_from_file=True
    # )

    # prng_key = jrand.PRNGKey(0)
    # prng_key, subkey = jrand.split(prng_key)
    # blm = BigramLanguageModel(data_loader.vocab_size, subkey)

    # lr = 3e-4
    # opt = optax.adabelief(lr)
    # opt_state = opt.init(equinox.filter(blm, equinox.is_inexact_array))

    # # Train
    # num_steps = 1_000_000
    # total_loss = 0
    # total_size = 0
    # print_every = 10_000
    # import math

    # for step in range(num_steps):
    #     batch_input, batch_expected = data_loader.get_batch()

    #     loss, blm, opt_state = make_step(
    #         blm, batch_input, batch_expected, opt_state, opt.update
    #     )
    #     if math.isnan(loss):
    #         if total_size == 0:
    #             total_size = 1

    #         if (step % print_every) == 0 or step == num_steps - 1:
    #             print(f"Loss: {loss}, Total: {total_loss}, Size: {total_size}")
    #             print(f"Step={step} Loss={total_loss / total_size}")
    #             total_loss = 0
    #             total_size = 0
    #         continue

    #     total_loss += loss
    #     total_size += 1
    #     if (step % print_every) == 0 or step == num_steps - 1:
    #         print(f"Loss: {loss}, Total: {total_loss}, Size: {total_size}")
    #         print(f"Step={step} Loss={total_loss / total_size}")
    #         total_loss = 0
    #         total_size = 0

    #         equinox.tree_serialise_leaves("blm.eqx", blm)

    # generated_tokens = generate_tokens(blm)
    # print(generated_tokens)
    # print(tokenizer.decode(generated_tokens))


if __name__ == "__main__":
    main()
