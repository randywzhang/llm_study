import time
from argparse import ArgumentParser
from typing import Optional

import equinox
import jax
import jax.numpy as jnp
import jax.random as jrand
import optax
from equinox import nn
from jax import Array

from util import get_file_path

from .data_loader import (
    BATCH_SIZE,
    CONTEXT_SIZE,
    SAVE_FILE,
    TransformerDataLoader,
    get_bespoke_tokenizer,
)

JRAND_SEED = 0

N_EMBED = 32
HEAD_SIZE = 32

DROPOUT = 0.2


class AttentionHead(equinox.Module):
    """Implementation of a single attention head for learning, unused in Transformer class"""

    head_size: int
    context_size: int  # T

    key: nn.Linear
    query: nn.Linear
    value: nn.Linear

    tril: Array

    def __init__(
        self,
        layer_prng_key: Array,
        n_embed: int = N_EMBED,  # query_size (and value_size and key_size)
        head_size: int = HEAD_SIZE,  # qk_size (and vo_size)
        context_size: int = CONTEXT_SIZE,  # T
    ):
        self.head_size = head_size
        self.context_size = context_size

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

        self.tril = jnp.tril(jnp.ones((context_size, context_size)))

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
    context_size: int  # T

    key: nn.Linear
    query: nn.Linear
    value: nn.Linear

    multi_head_tril: Array

    dropout: nn.Dropout

    def __init__(
        self,
        num_heads: int,
        layer_prng_key: Array,
        n_embed: int = N_EMBED,
        context_size: int = CONTEXT_SIZE,
        dropout: float = DROPOUT,
    ):
        """
        n_embed is the in_size and the out_size for the module in order to
        allow for residual blocks.

        TODO: add output_proj linear layer
        Alternatively, we can introduce an output_proj layer that projects
        the output of the attention heads back up to the input_size which
        allows for more granular control of the attention hyperparameters


        TODO: Citations needed for the following claims

        n_embed is equivalent to eqx.nn.MultiheadAttention's query_size and
        casts the value_size and key_size to the same value.

        n_embed is also equivalent to eqx.nn.MultiheadAttention's qk_size
        multiplied by num_heads. vo_size here is equal to qk_size
        """
        head_size = n_embed // num_heads
        if n_embed != head_size * num_heads:
            raise NumHeadsError

        self.num_heads = num_heads
        self.head_size = head_size
        self.context_size = context_size

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

        tril = jnp.tril(jnp.ones((context_size, context_size)))
        single_head_tril = jnp.expand_dims(tril, axis=0)
        self.multi_head_tril = jnp.tile(single_head_tril, reps=(num_heads, 1, 1))

        self.dropout = nn.Dropout(dropout)

        """
        Naive impl:

            self.heads = [
                AttentionHead(n_embed, head_size, context_size, jrand_seed)
                for _ in range(num_heads)
            ]
        """

    def __call__(self, x: Array, *, key: Optional[Array] = None) -> Array:
        """
        Naive impl:

            return jnp.concat([h(x) for h in self.heads], axis=-1)
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

        # apply dropout to block communication between tokens
        context_mat = self.dropout(context_mat, key=key)

        output = context_mat @ value_heads  # (num_heads, T, head_size)
        output = jnp.transpose(output, axes=(1, 0, 2)).reshape(
            self.context_size, self.num_heads * self.head_size
        )  # transpose to (T, num_heads, head_size), reshape to (T, num_heads * head_size)

        return self.dropout(output, key=key)  # apply dropout

    # same as equinox.nn._attention.MultiheadAttention._project
    def _project(self, proj: nn.Linear, x: Array) -> Array:
        seq_length, _ = x.shape  # (T, embed_size)
        projection = jax.vmap(proj)(x)  # feeds x through each attn head, (T, out_size)
        return projection.reshape(
            seq_length, self.num_heads, -1
        )  # reshapes output (T, num_heads, head_size)


class Block(equinox.Module):
    """
    Transformer block: communication followed by computation

    Implemented as a residual block with pre-norm formulation
    """

    sa_head: MultiHeadAttention  # self_attention_head
    ff_layer: nn.Linear  # feed_forward layer to process attn

    ln_sa: nn.LayerNorm
    ln_ff: nn.LayerNorm

    dropout: nn.Dropout

    def __init__(
        self,
        layer_prng_key: Array,
        num_heads: int = NUM_HEADS,
        n_embed: int = N_EMBED,  # C_embed
        context_size: int = CONTEXT_SIZE,  # T
        dropout: float = DROPOUT,
    ):
        self.ln_sa = nn.LayerNorm(n_embed)
        self.sa_head = MultiHeadAttention(
            num_heads=num_heads,
            layer_prng_key=layer_prng_key,
            n_embed=n_embed,
            context_size=context_size,
            dropout=dropout,
        )

        self.ln_ff = nn.LayerNorm(n_embed)
        layer_prng_key, subkey = jrand.split(layer_prng_key)
        self.ff_layer = nn.Linear(
            in_features=n_embed,
            out_features=n_embed,
            key=subkey,
        )

        self.dropout = nn.Dropout(dropout)

    def __call__(self, x: Array, *, key: Optional[Array] = None) -> Array:
        # residual self attention block (sum input to layer outputs)
        x = jax.vmap(self.ln_sa)(x)  # pre-norm formulation
        x = x + self.sa_head(x, key=key)

        x = jax.vmap(self.ln_ff)(x)  # pre-norm formulation
        x = jax.vmap(self.ff_layer)(x)
        x = x + jax.nn.relu(x)
        return self.dropout(x, key=key)


N_BLOCKS = 3


class Transformer(equinox.Module):
    token_embedding_table: nn.Embedding
    position_embedding_table: nn.Embedding
    attention_blocks: nn.Sequential
    lmh_pre_norm: nn.LayerNorm
    lm_head: nn.Linear  # language_model_head

    context_size: int  # T

    def __init__(
        self,
        vocab_size: int,  # C_vocab
        n_embed: int = N_EMBED,  # C_embed, head_size * num_heads
        context_size: int = CONTEXT_SIZE,  # T
        num_heads: int = 1,  # equivalent to single attention head
        num_blocks: int = N_BLOCKS,
        jrand_seed: int = JRAND_SEED,
        dropout: float = DROPOUT,
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

        blocks: list[equinox.Module] = []
        for _ in range(num_blocks):
            layer_prng_key, subkey = jrand.split(layer_prng_key)
            blocks.append(
                Block(
                    layer_prng_key=subkey,
                    num_heads=num_heads,
                    n_embed=n_embed,
                    context_size=context_size,
                    dropout=dropout,
                )
            )

        self.attention_blocks = nn.Sequential(blocks)

        self.lmh_pre_norm = nn.LayerNorm(n_embed)

        layer_prng_key, subkey = jrand.split(layer_prng_key)
        self.lm_head = nn.Linear(
            in_features=n_embed,
            out_features=vocab_size,
            key=subkey,
        )

        self.context_size = context_size

    def generate_token(self, x: Array, prng_key: Array) -> int:
        """
        @param x - model input context 1d array

        generates the next token index given the input context
        """
        logits = self.__call__(x)

        # equivalent to torch.multinomial(softmax(logits, axis=-1), num_samples = 1)
        return jax.random.categorical(prng_key, logits)

    def __call__(self, x: Array, *, key: Optional[Array] = None) -> Array:
        """
        @param x - model input context
        """
        # compute embedding
        tok_emb = jax.vmap(self.token_embedding_table)(x)  # (T, C_embed)
        pos_emb = jax.vmap(self.position_embedding_table)(
            jnp.arange(self.context_size)
        )  # (T, C_embed)
        emb = tok_emb + pos_emb

        # feed embedding through attention head, head_size * num_heads = C_embed
        emb = self.attention_blocks(emb, key=key)  # (T, C_embed)

        # applies layer normalization
        emb = jax.vmap(self.lmh_pre_norm)(emb)  # (T, C_embed)

        logits = jax.vmap(self.lm_head)(emb)  # (T, C_vocab)

        return logits


def compute_loss(
    model: equinox.Module, x: Array, y: Array, key: Optional[Array] = None
) -> float:
    out = jax.vmap(model)(x, key=key)  # vmap along batch dimension

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
    loss_fn: callable,
    key: Optional[Array] = None,
) -> tuple[float, equinox.Module, optax.OptState]:
    loss, grads = loss_fn(model, batch_input, batch_expected, key)
    updates, opt_state = opt_update(grads, opt_state)
    model = equinox.apply_updates(model, updates)
    return loss, model, opt_state


VAL_ITERS = 200  # validation loops


@equinox.filter_jit
def eval_model(
    model: equinox.Module,
    data_loader: TransformerDataLoader,
    prng_key: Array,
    val_iters: int = VAL_ITERS,
) -> float:
    model = nn.inference_mode(model)
    total_loss: float = 0

    def aggregate_loss(
        init_val: tuple[float, int, Array], _x: int
    ) -> tuple[float, int, Array]:
        total_loss, prng_key = init_val
        xv, yv, prng_key = data_loader.get_batch(train=False, prng_key=prng_key)
        loss = compute_loss(model, xv, yv)

        return (total_loss + loss, prng_key), _x

    (total_loss, _), _ = jax.lax.scan(
        aggregate_loss,
        (total_loss, prng_key),
        jnp.ones(val_iters),
    )

    model = nn.inference_mode(model, value=False)

    return total_loss / val_iters


NUM_NEW_TOKENS = 100


def generate_tokens(
    model: equinox.Module, num_new_tokens: int = NUM_NEW_TOKENS, seed: int = 0
) -> list[int]:
    model = nn.inference_mode(model)
    prng_key = jrand.PRNGKey(seed)
    generated_tokens = []
    next_token = jnp.array([0])
    for _ in range(num_new_tokens):
        prng_key, subkey = jrand.split(prng_key)
        next_token = model.generate_token(next_token, subkey)
        generated_tokens.append(next_token[0].item())

    model = nn.inference_mode(model, value=False)
    return generated_tokens


# ruff: noqa: PLR0913
def main(
    lr: int = 3e-4,
    num_steps: int = 50_000,
    epoch_size: int = 10_000,
    train: bool = False,
    batch_size: int = BATCH_SIZE,
    evaluate: bool = False,
    val_iters: int = VAL_ITERS,
    context_size: int = CONTEXT_SIZE,
    n_blocks: int = N_BLOCKS,
    n_embed: int = N_EMBED,
    num_heads: int = NUM_HEADS,
    dropout: float = DROPOUT,
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
    data_loader.tokenizer = tokenizer

    if not data_loader.load_encoding(encoding_save_file):
        print("Failed to load encoding")
        data_loader.generate_encoding(tokenizer)

    data_loader.batch_size = batch_size
    data_loader.context_size = context_size
    data_loader.refresh_data()

    transformer = Transformer(
        vocab_size=data_loader.vocab_size,
        n_embed=n_embed,
        context_size=context_size,
        num_heads=num_heads,
        num_blocks=n_blocks,
        jrand_seed=rand_seed,
        dropout=dropout,
    )

    if load_pytree:
        transformer = equinox.tree_deserialise_leaves(
            model_pytree_save_file, transformer
        )

    opt = optax.adabelief(lr)
    opt_state = opt.init(equinox.filter(transformer, equinox.is_inexact_array))

    if train:
        total_loss = 0

        start_time = time.time()
        end_time = time.time()

        prng_key = jrand.PRNGKey(rand_seed)
        dropout_keys = jnp.tile(prng_key, (batch_size, 1))

        get_train_data = equinox.filter_jit(data_loader.get_batch)

        loss_fn = equinox.filter_value_and_grad(compute_loss)

        for step in range(num_steps):
            batch_input, batch_expected, prng_key = get_train_data(prng_key=prng_key)

            loss, transformer, opt_state = make_step(
                model=transformer,
                batch_input=batch_input,
                batch_expected=batch_expected,
                opt_state=opt_state,
                opt_update=opt.update,
                loss_fn=loss_fn,
                key=dropout_keys,
            )

            total_loss += loss
            if ((step + 1) % epoch_size) == 0 or step == num_steps - 1:
                print(f"Epoch: {(step + 1) // epoch_size}")

                print(f"\tTraining loss: {total_loss / num_steps}")
                total_loss = 0

                if evaluate:
                    val_loss = eval_model(transformer, data_loader, prng_key, val_iters)
                    print(f"\tValidation loss: {val_loss}")

                end_time = time.time()
                elapsed_time = end_time - start_time
                print(f"\tElapsed time: {elapsed_time} seconds")
                start_time = end_time

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
    "-b",
    "--batch-size",
    type=int,
    help="Number of contexts per batch of training, default 4",
    default=BATCH_SIZE,
)
parser.add_argument(
    "-v",
    "--evaluate",
    action="store_true",
    help="If set will evaluate the network after each epoch during training",
    default=False,
)
parser.add_argument(
    "-vi",
    "--val-iters",
    type=int,
    help="Number of evaluation iterations to loop through, default 200",
    default=VAL_ITERS,
)
parser.add_argument(
    "-c",
    "--context-size",
    type=int,
    help="Number of tokens in each context block used for training, default 8",
    default=CONTEXT_SIZE,
)
parser.add_argument(
    "-nb",
    "--n-blocks",
    type=int,
    help="Number of attention blocks, default 3",
    default=N_BLOCKS,
)
parser.add_argument(
    "-ne",
    "--n-embed",
    type=int,
    help="Size of embedding vector, default 32",
    default=N_EMBED,
)
parser.add_argument(
    "-nh",
    "--num-heads",
    type=int,
    help="Number of attention heads, default 4",
    default=NUM_HEADS,
)
parser.add_argument(
    "-d",
    "--dropout",
    type=float,
    help="Dropout rate for training, default 0.2",
    default=DROPOUT,
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
    help="File to save & load pytree from. Overwrites file on save. Default is 'blm.eqx'",
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
        lr=args.learning_rate,
        num_steps=args.num_steps,
        epoch_size=args.epoch_size,
        train=args.train,
        batch_size=args.batch_size,
        evaluate=args.evaluate,
        val_iters=args.val_iters,
        context_size=args.context_size,
        n_blocks=args.n_blocks,
        n_embed=args.n_embed,
        num_heads=args.num_heads,
        dropout=args.dropout,
        data_file=args.data_file,
        encoding_save_file=args.encoding_save_file,
        tokenizer_save_file=args.tokenizer_save_file,
        load_pytree=args.load_pytree,
        model_pytree_save_file=args.model_pytree_save_file,
        rand_seed=args.rand_seed,
        num_new_tokens=args.num_new_tokens,
    )
