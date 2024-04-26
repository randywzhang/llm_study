"""
This module contains the TransformerDataLoader class and doubles
as a script to train a tokenizer on an input file and encode the
file. 

Usage:

pants run src/transformer/data_loader.py -- input_file [--vocab-size] \
[--load-tokenizer <filename>] [--train-tokenizer] [--save-tokenizer <filename>] \
[--encoding-save-file <filename>]
"""

import pickle
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
import jax.random as jrand
from jax import Array

from tokens.regex_tokenizer import SAVE_FILE, RegexTokenizer
from util import get_file_path, load_text_from_file

TOKENIZER_VOCAB_SIZE = 500


def get_bespoke_tokenizer(
    raw_text: str,
    vocab_size: int = TOKENIZER_VOCAB_SIZE,
    load_from_file: bool = False,
    load_file: str = SAVE_FILE,
    train_further: bool = False,
    save_tokenizer: bool = False,
    save_file: str = SAVE_FILE,
    directory: str = __file__,
) -> RegexTokenizer:
    """
    Note: If you did a split for training/test data, you must
    ensure that the training data has the same set of unique
    characters that the test data does. Otherwise you will
    either error out or get garbage back.

    Also, if you are loading from file, you must make sure that
    the raw_text input is the same text that the saved tokenizer
    was trained with.
    """
    unique_chars = sorted(list(set(raw_text)))

    def base_tokenizer(text: str) -> list[int]:
        stoi = {ch: i for i, ch in enumerate(unique_chars)}
        return [stoi[ch] for ch in text]

    base_vocab = {i: ch.encode("utf-8") for i, ch in enumerate(unique_chars)}

    regex_tokenizer = RegexTokenizer(
        base_vocab=base_vocab,
        base_vocab_size=len(unique_chars),
        base_tokenizer=base_tokenizer,
    )

    if load_from_file:
        regex_tokenizer.load(load_file, directory)
        regex_tokenizer.base_tokenizer = base_tokenizer
        if not train_further:
            return regex_tokenizer

    regex_tokenizer.train(raw_text, vocab_size)

    if save_tokenizer:
        regex_tokenizer.save(save_file, directory)

    return regex_tokenizer


BATCH_SIZE = 4
CONTEXT_SIZE = 8

JRAND_SEED = 0


class TransformerDataLoader:
    data_file: str
    directory: str
    raw_text: str
    tokenizer: RegexTokenizer
    encoding: list[int]

    _data: Array
    _train_data: Array
    _val_data: Array

    batch_size: int
    context_size: int

    _prng_key: Array

    # ruff: noqa: PLR0913
    def __init__(
        self,
        data_file: str,
        directory: Optional[str] = __file__,
        tokenizer: RegexTokenizer = None,
        encoding_save_file: Optional[str] = None,
        batch_size: int = BATCH_SIZE,
        context_size: int = CONTEXT_SIZE,
        jrand_seed: int = JRAND_SEED,
    ):
        self.data_file = data_file
        self.directory = directory
        self.load_raw_text()

        self.tokenizer = tokenizer
        self.encoding = None
        if encoding_save_file:
            self.encoding_save_file = encoding_save_file
        else:
            self.encoding_save_file = "_".join(
                [data_file.split(".")[0], "encoding.pkl"]
            )

        self._data = None
        self._train_data = None
        self._val_data = None

        self.batch_size = batch_size
        self.context_size = context_size

        self._prng_key = jrand.PRNGKey(jrand_seed)

    def load_raw_text(self) -> None:
        self.raw_text = load_text_from_file(self.data_file, self.directory)

    def generate_encoding(self, tokenizer: RegexTokenizer = None) -> None:
        if tokenizer:
            self.tokenizer = tokenizer

        self.encoding = self.tokenizer.encode(self.raw_text)

    def save_encoding(
        self,
        encoding_save_file: Optional[str] = None,
        directory: str = __file__,
    ) -> None:
        if not encoding_save_file:
            encoding_save_file = self.encoding_save_file

        with Path.open(get_file_path(encoding_save_file, directory), "wb") as f:
            f.write(pickle.dumps(self.encoding))

    def load_encoding(
        self,
        encoding_save_file: Optional[str] = None,
        directory: str = __file__,
    ) -> bool:
        if not encoding_save_file:
            encoding_save_file = self.encoding_save_file

        try:
            with Path.open(get_file_path(encoding_save_file, directory), "rb") as f:
                # ruff: noqa: S301
                self.encoding = pickle.loads(f.read())
        except FileNotFoundError:
            return False

        return True

    def clear_data(self) -> None:
        self._data = None
        self._train_data = None
        self._val_data = None

    def refresh_data(self) -> tuple[Array, Array, Array]:
        return self.data, self.train_data, self.val_data

    def get_batch(
        self, train: bool = True, prng_key: Optional[Array] = None
    ) -> tuple[Array, Array, Array]:
        """
        In order to jit this function, a prng_key must be passed and
        refresh_data must be called
        """
        if prng_key is None:
            prng_key = self.prng_key

        data = self.train_data if train else self.val_data

        prng_key, subkey = jrand.split(prng_key)
        ix = jrand.randint(
            subkey,
            (self.batch_size,),
            0,
            len(data) - self.context_size,
        )
        iy = jnp.add(ix, 1)

        def slice_single(start: Array) -> Array:
            return jax.lax.dynamic_slice(data, (start,), (self.context_size,))

        x = jnp.stack(jax.vmap(slice_single)(ix))
        y = jnp.stack(jax.vmap(slice_single)(iy))
        # x = jnp.stack([data[i : i + self.context_size] for i in ix])
        # y = jnp.stack([data[i + 1 : i + self.context_size + 1] for i in ix])
        return x, y, prng_key

    @property
    def data(self) -> Array:
        if self._data is None:
            self._data = jnp.array(self.encoding)

        return self._data

    @property
    def train_data(self) -> Array:
        if self._train_data is None:
            self._train_data = self.data[: int(0.9 * self.data.shape[0])]

        return self._train_data

    @property
    def val_data(self) -> Array:
        if self._val_data is None:
            self._val_data = self.data[int(0.9 * self.data.shape[0]) :]

        return self._val_data

    @property
    def vocab_size(self) -> int:
        if self.encoding is None:
            return 0

        return len(set(self.encoding))

    @property
    def prng_key(self) -> Array:
        self._prng_key, subkey = jrand.split(self._prng_key)
        return subkey


parser = ArgumentParser()
parser.add_argument("input_file")
parser.add_argument("--vocab-size", default=500)
parser.add_argument(
    "--load-tokenizer",
    type=str,
    help="File to load tokenizer from, default tokenizer save location is regex_tokenizer.pkl",
)
parser.add_argument(
    "--train-tokenizer",
    action="store_true",
    default=False,
    help="If loading tokenizer, setting this flag trains the loaded tokenizer until its vocabulary reaches vocab_size",
)
parser.add_argument(
    "--save-tokenizer",
    default=SAVE_FILE,
    type=str,
    help="File to save tokenizer to, default is regex_tokenizer.pkl",
)
parser.add_argument(
    "--encoding-save-file",
    type=str,
    help="File to save encoding to. Default is <input_file>_encoding.pkl",
)


# ruff: noqa: T201
if __name__ == "__main__":
    args = parser.parse_args()

    data_loader = TransformerDataLoader(args.input_file)
    tokenizer = get_bespoke_tokenizer(
        raw_text=data_loader.raw_text,
        vocab_size=args.vocab_size,
        load_from_file=args.load_tokenizer is not None,
        load_file=args.load_tokenizer,
        train_further=args.train_tokenizer,
        save_tokenizer=args.save_tokenizer is not None,
        save_file=args.save_tokenizer,
    )

    print("encoding file...")
    data_loader.generate_encoding(tokenizer)

    print("saving encoding...")
    data_loader.save_encoding(args.encoding_save_file)
