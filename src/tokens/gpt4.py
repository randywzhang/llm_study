"""
Implementation of recover_merges follows Andrej Karpathy's minbpe exercise:
https://github.com/karpathy/minbpe/blob/master/exercise.md
"""

import tiktoken

"""
?i: - ignore case
'[sdmt]|ll|ve|re - matches 's 'd 'm 't 'll 've 're for contractions/possessive
[^\r\n\p{L}\p{N}] - NOT \r \n unicode letter, unicode number
?+ - one or more times
\p{L}+ - one or more unicode letters
\p{N}{1,3} - 1 to 3 unicode numbers
 ? - optional space
[^\s\p{L}\p{N}]++ - one or more characters that are NOT whitespace, letters, numbers
[\r\n]* - zero or more \r or \n
\s*[\r\n] - any whitespace followed by \r or \n
\s+(?!\S) - matches one or more whitespace characters up to but not including the space before a non-whitespace character (' x' would not match, '  x' would match ' ')
\s+ - matches one or more whitespace characters
"""
GPT_4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

GPT_4_CL100K_BASE_ENCODING = tiktoken.get_encoding("cl100k_base")
GPT_4_CL100K_SPECIAL_TOKENS = GPT_4_CL100K_BASE_ENCODING._special_tokens
GPT_4_CL100K_BASE_VOCAB_SIZE = GPT_4_CL100K_BASE_ENCODING.n_vocab - len(
    GPT_4_CL100K_SPECIAL_TOKENS
)


class BPERecoveryError(Exception):
    pass


def get_bpe(
    mergeable_ranks: dict[bytes, int], token: bytes, max_rank: int
) -> tuple[bytes, bytes]:
    token_parts = [bytes([b]) for b in token]

    """
    We need to reduce token_parts to a tuple to recover the pair
    that was merged to create the current rank, max_rank
    
    max_rank is named as such because the token cannot be created from
    merges with higher ranks

    The way we reduce the tokens must match the order in which the
    merges originally occurred. That is to say, we must merge in the
    order of rank.
    """
    while True:
        min_idx = None
        min_rank = None

        # find the byte pair (or pair of parts) with minimum rank
        for part_idx, pair in enumerate(zip(token_parts, token_parts[1:])):
            if rank := mergeable_ranks.get(pair[0] + pair[1]):
                # we have found a pair of parts that exists in mergeable ranks
                if min_rank is None or rank < min_rank:
                    # the rank of this pair is the lowest rank found
                    min_idx = part_idx
                    min_rank = rank

        if min_rank is None:
            # no more merges were found
            break

        if max_rank is not None and min_rank >= max_rank:
            # no valid merges were found (cannot merge with a higher rank,
            # during the training of the tokenizer, such a rank would not
            # yet exist)
            break

        # conduct the merge on the specified bytes
        token_parts = (
            token_parts[:min_idx]  # prefix parts
            + [token_parts[min_idx] + token_parts[min_idx + 1]]  # current merge
            + token_parts[min_idx + 2 :]  # suffix parts
        )

    if not len(token_parts) == 2:
        raise BPERecoveryError

    return token_parts[0], token_parts[1]


def recover_merges(mergeable_ranks: dict[bytes, int]) -> dict[tuple[int, int], int]:
    merges = {}
    for token, raw_rank in mergeable_ranks.items():
        rank = raw_rank

        if len(token) == 1:
            continue  # skip raw bytes

        pair = get_bpe(mergeable_ranks, token, max_rank=rank)

        # recover the integer ranks of the pair
        ix0 = mergeable_ranks[pair[0]]
        ix1 = mergeable_ranks[pair[1]]

        merges[(ix0, ix1)] = rank

    return merges


def gpt_4_base_str_tokenizer(text: str) -> list[int]:
    """
    The base encoding for utf-8 doesn't match the integer value
    in gpt4's _mergeable_ranks. This function remaps utf-8 bytes
    to gpt4's token ids.
    """

    """
    DEBUG_NOTES:
    
    This was a pita to get to work, tests were failing and
    I had no idea why, thought I messed up encoding/decoding
    but after double + triple ... checking the encode decode
    logic, it had to be something else. Looking into gpt4's
    _mergeable_ranks (rank 0 to 255) we can see that
    b'A' maps to 32, while int(b'A') == 65.


    for i, item in enumerate(GPT_4_CL100K_BASE_ENCODING._mergeable_ranks.items()):
        if i < 256:
            print(item)


    print(text)
    utf8 = text.encode(encoding="utf-8")
    print(utf8)
    utf8_list = list(text.encode(encoding="utf-8"))
    print(utf8_list)
    utf8_byte_list = [bytes([char]) for char in utf8_list]
    print(utf8_byte_list)
    """

    return [
        GPT_4_CL100K_BASE_ENCODING._mergeable_ranks.get(bytes([char]))
        for char in list(text.encode(encoding="utf-8"))
    ]
