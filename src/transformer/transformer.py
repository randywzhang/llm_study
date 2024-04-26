from argparse import ArgumentParser

import equinox
import jax
import jax.numpy as jnp
import jax.random as jrand
import optax
from equinox import nn
from jax import Array

from util import get_file_path

from .data_loader import (
    BLOCK_SIZE,
    SAVE_FILE,
    TransformerDataLoader,
    get_bespoke_tokenizer,
)

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
        n_embed: int = N_EMBED,  # query_size (and value_size and key_size)
        head_size: int = HEAD_SIZE,  # qk_size (and vo_size)
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


NUM_HEADS = 4


class NumHeadsError(Exception):
    def __init__(self) -> None:
        super().__init__(msg="num_heads doesn't evenly divide out_size")


class MultiHeadAttention(equinox.Module):
    num_heads: int
    head_size: int
    block_size: int  # T

    key: nn.Linear
    query: nn.Linear
    value: nn.Linear

    multi_head_tril: Array

    def __init__(
        self,
        num_heads: int,
        n_embed: int = N_EMBED,  # query_size (and value_size and key_size)
        out_size: int = HEAD_SIZE,  # num_heads * qk_size (vo_size = qk_size)
        block_size: int = BLOCK_SIZE,
        jrand_seed: int = JRAND_SEED,
    ):
        head_size = out_size // num_heads
        if out_size != head_size * num_heads:
            raise NumHeadsError

        self.num_heads = num_heads
        self.head_size = head_size
        self.block_size = block_size

        layer_prng_key = jrand.PRNGKey(jrand_seed)

        layer_prng_key, subkey = jrand.split(layer_prng_key)
        self.key = nn.Linear(
            in_features=n_embed,
            out_features=head_size * num_heads,
            use_bias=False,
            key=subkey,
        )

        layer_prng_key, subkey = jrand.split(layer_prng_key)
        self.query = nn.Linear(
            in_features=n_embed,
            out_features=head_size * num_heads,
            use_bias=False,
            key=subkey,
        )

        layer_prng_key, subkey = jrand.split(layer_prng_key)
        self.value = nn.Linear(
            in_features=n_embed,
            out_features=head_size * num_heads,
            use_bias=False,
            key=subkey,
        )

        tril = jnp.tril(jnp.ones((block_size, block_size)))
        single_head_tril = jnp.expand_dims(tril, axis=0)
        self.multi_head_tril = jnp.tile(single_head_tril, reps=(num_heads, 1, 1))

        """
        Naive impl

        # self.heads = [
        #     AttentionHead(n_embed, head_size, block_size, jrand_seed)
        #     for _ in range(num_heads)
        # ]
        """

    def __call__(self, x: Array) -> Array:
        """
        Naive impl

        # return jnp.concat([h(x) for h in self.heads], axis=-1)
        """

        # get the individual attention heads via self._project
        query_heads = self._project(self.query, x)  # (T, num_heads, head_size)
        key_heads = self._project(self.key, x)
        value_heads = self._project(self.value, x)

        # reshape heads for matmul
        # combine key_heads transposes in context_mat calculation:
        #   transpose(key_heads, (1, 0, 2)) -> (num_heads, T, head_size)
        #   transpose(key_heads, (1, 2, 0)) -> (num_heads, head_size, T)
        query_heads = jnp.transpose(query_heads, (1, 0, 2))  # (num_heads, T, head_size)
        value_heads = jnp.transpose(value_heads, (1, 0, 2))

        # obtain the context matrices for each attention head
        context_mat = (
            query_heads
            @ jnp.transpose(key_heads, axes=(1, 2, 0))
            * self.head_size**-0.5
        )  # (num_heads, T, T)

        # --DECODER LOGIC--
        context_mat = jnp.where(self.multi_head_tril == 0, -jnp.inf, context_mat)
        # --END DECODER LOGIC--

        context_mat = jax.nn.softmax(context_mat, axis=-1)

        output = context_mat @ value_heads  # (num_heads, T, head_size)
        return jnp.transpose(output, axes=(1, 0, 2)).reshape(
            self.block_size, self.num_heads * self.head_size
        )  # transpose to (T, num_heads, head_size), reshape to (T, num_heads * head_size)

    # same as equinox.nn._attention.MultiheadAttention._project
    def _project(self, proj: nn.Linear, x: Array) -> Array:
        seq_length, _ = x.shape  # (T, embed_size)
        projection = jax.vmap(proj)(x)  # feeds x through each attn head, (T, out_size)
        return projection.reshape(
            seq_length, self.num_heads, -1
        )  # reshapes output (T, num_heads, head_size)


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
        num_heads: int = 1,  # equivalent to single attention head
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

        self.sa_head = MultiHeadAttention(
            num_heads, n_embed, head_size, block_size, jrand_seed
        )

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
) -> tuple[float, equinox.Module, optax.OptState]:
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
    for _ in range(num_new_tokens):
        prng_key, subkey = jrand.split(prng_key)
        next_token = model.generate_token(next_token, subkey)
        generated_tokens.append(next_token[0].item())

    return generated_tokens


# ruff: noqa: PLR0913
def main(
    lr: int = 3e-4,
    num_steps: int = 50_000,
    epoch_size: int = 10_000,
    train: bool = False,
    num_heads: int = NUM_HEADS,
    data_file: str = "tiny_shakespeare.txt",
    encoding_save_file: str = "tiny_shakespeare_encoding.pkl",
    tokenizer_save_file: str = SAVE_FILE,
    load_pytree: bool = False,
    model_pytree_save_file: str = get_file_path("transformer.eqx", __file__),
    rand_seed: int = JRAND_SEED,
    num_new_tokens: int = NUM_NEW_TOKENS,
) -> None:
    """
    Prerequisites:

    Must have already trained and saved a tokenizer on the data_file.
    See data_loader.py

    Some of the model parameters will be defined by tokenizer fields,
    such as vocab_size.
    """
    data_loader = TransformerDataLoader(data_file)
    tokenizer = get_bespoke_tokenizer(
        raw_text=data_loader.raw_text,
        load_from_file=True,
        load_file=tokenizer_save_file,
    )

    if not data_loader.load_encoding(encoding_save_file):
        print("Failed to load encoding")
        data_loader.generate_encoding(tokenizer)

    transformer = Transformer(
        data_loader.vocab_size, num_heads=num_heads, jrand_seed=rand_seed
    )

    if load_pytree:
        transformer = equinox.tree_deserialise_leaves(
            model_pytree_save_file, transformer
        )

    opt = optax.adabelief(lr)
    opt_state = opt.init(equinox.filter(transformer, equinox.is_inexact_array))

    if train:
        total_loss = 0
        total_size = 0
        import math

        for step in range(num_steps):
            batch_input, batch_expected = data_loader.get_batch()

            loss, transformer, opt_state = make_step(
                transformer, batch_input, batch_expected, opt_state, opt.update
            )
            # TODO: figure out why ~25% of steps result in nan loss
            if math.isnan(loss):
                if total_size == 0:
                    total_size = 1

                if (step % epoch_size) == 0 or step == num_steps - 1:
                    print(f"Loss: {loss}, Total: {total_loss}, Size: {total_size}")
                    print(f"Step={step} Loss={total_loss / total_size}")
                    total_loss = 0
                    total_size = 0

                    equinox.tree_serialise_leaves(model_pytree_save_file, transformer)

                continue

            total_loss += loss
            total_size += 1
            if (step % epoch_size) == 0 or step == num_steps - 1:
                print(f"Loss: {loss}, Total: {total_loss}, Size: {total_size}")
                print(f"Step={step} Loss={total_loss / total_size}")
                total_loss = 0
                total_size = 0

                equinox.tree_serialise_leaves(model_pytree_save_file, transformer)

        equinox.tree_serialise_leaves(model_pytree_save_file, transformer)

    generated_tokens = generate_tokens(
        model=transformer,
        num_new_tokens=num_new_tokens,
        seed=rand_seed,
    )
    print(tokenizer.decode(generated_tokens))


# TODO: clean up args, create input file for hyperparameters
# lr, num_heads, etc.
parser = ArgumentParser()
parser.add_argument(
    "-lr",
    "--learning-rate",
    type=float,
    help="Learning rate, default 3e-4",
    default=3e-4,
)
parser.add_argument(
    "-i",
    "--num-steps",
    type=int,
    help="Number of training iterations, default 50_000",
    default=50_000,
)
parser.add_argument(
    "-e",
    "--epoch-size",
    type=int,
    help="Number of training iterations per epoch, default 10_000. Loss averages will be printed at the end of a training epoch.",
    default=10_000,
)
parser.add_argument(
    "-t",
    "--train",
    action="store_true",
    help="If set will enable the training loop",
    default=False,
)
parser.add_argument(
    "-nh",
    "--num-heads",
    type=int,
    help="Number of attention heads, default 1",
    default=NUM_HEADS,
)
parser.add_argument(
    "--data-file",
    type=str,
    help="File to load training data from, default 'tiny_shakespeare.txt'. Must have trained a RegexTokenizer on this data, see data_loader.py",
    default="tiny_shakespeare.txt",
)
parser.add_argument(
    "--encoding-save-file",
    type=str,
    help="File to load encoding from. Default is 'tiny_shakespeare_encoding.pkl'",
    default="tiny_shakespeare_encoding.pkl",
)
parser.add_argument(
    "--tokenizer-save-file",
    type=str,
    help="File to load tokenizer from. Default is 'regex_tokenizer.pkl'",
    default=SAVE_FILE,
)
parser.add_argument(
    "-l",
    "--load-pytree",
    action="store_true",
    help="If set will load a model from the model_pytree_save_file.",
    default=False,
)
parser.add_argument(
    "-m",
    "--model-pytree-save-file",
    type=str,
    help="File to load pytree from. Default is 'blm.eqx'",
    default=get_file_path("transformer.eqx", __file__),
)
parser.add_argument(
    "-n",
    "--num-new-tokens",
    type=int,
    help="Number of tokens to generate, default 100",
    default=NUM_NEW_TOKENS,
)
parser.add_argument(
    "-s",
    "--rand-seed",
    type=int,
    help="Seed for jax.random, default 0",
    default=JRAND_SEED,
)


if __name__ == "__main__":
    args = parser.parse_args()
    main(
        args.learning_rate,
        args.num_steps,
        args.epoch_size,
        args.train,
        args.num_heads,
        args.data_file,
        args.encoding_save_file,
        args.tokenizer_save_file,
        args.load_pytree,
        args.model_pytree_save_file,
        args.rand_seed,
        args.num_new_tokens,
    )
