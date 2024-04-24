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
    JRAND_SEED,
    SAVE_FILE,
    TransformerDataLoader,
    get_bespoke_tokenizer,
)


class BigramLanguageModel(equinox.Module):
    token_embedding_table: nn.Embedding

    def __init__(self, vocab_size: int, layer_prng_key: Array):
        self.token_embedding_table = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_size=vocab_size,
            key=layer_prng_key,
        )

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
    data_file: str = "tiny_shakespeare.txt",
    encoding_save_file: str = "tiny_shakespeare_encoding.pkl",
    tokenizer_save_file: str = SAVE_FILE,
    load_pytree: bool = False,
    model_pytree_save_file: str = get_file_path("blm.eqx", __file__),
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

    prng_key = jrand.PRNGKey(rand_seed)
    prng_key, subkey = jrand.split(prng_key)
    blm = BigramLanguageModel(data_loader.vocab_size, subkey)

    if load_pytree:
        blm = equinox.tree_deserialise_leaves(model_pytree_save_file, blm)

    opt = optax.adabelief(lr)
    opt_state = opt.init(equinox.filter(blm, equinox.is_inexact_array))

    if train:
        total_loss = 0
        total_size = 0
        import math

        for step in range(num_steps):
            batch_input, batch_expected = data_loader.get_batch()

            loss, blm, opt_state = make_step(
                blm, batch_input, batch_expected, opt_state, opt.update
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
                continue

            total_loss += loss
            total_size += 1
            if (step % epoch_size) == 0 or step == num_steps - 1:
                print(f"Loss: {loss}, Total: {total_loss}, Size: {total_size}")
                print(f"Step={step} Loss={total_loss / total_size}")
                total_loss = 0
                total_size = 0

                equinox.tree_serialise_leaves(model_pytree_save_file, blm)

    generated_tokens = generate_tokens(
        model=blm,
        num_new_tokens=num_new_tokens,
        seed=rand_seed,
    )
    print(generated_tokens)
    print(tokenizer.decode(generated_tokens))


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
    default=get_file_path("blm.eqx", __file__),
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
        args.data_file,
        args.encoding_save_file,
        args.tokenizer_save_file,
        args.load_pytree,
        args.model_pytree_save_file,
        args.rand_seed,
        args.num_new_tokens,
    )
