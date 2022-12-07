##########################################
# File: tokenizer.py                     #
# Copyright Richard Stebbing 2022.       #
# Distributed under the MIT License.     #
# (See accompany file LICENSE or copy at #
#  http://opensource.org/licenses/MIT)   #
##########################################

import string
import typing as t
from functools import cached_property

import torch

from workshop.ext import pydantic as p
from workshop.ext.torch import validate_tensor


class TokenizerConfig(p.BaseModel):
    num_lowercase_letters: int = 3

    @p.validator("num_lowercase_letters")
    def validate_num_lowercase_letters(cls, num_lowercase_letters):
        if not (1 <= num_lowercase_letters <= 26):
            raise ValueError(f"num_lowercase_letters must be in closed interval [1, 26]\n{num_lowercase_letters = }")

        return num_lowercase_letters

    def __post_init__(self):
        assert len(self.special_tokens_) == len(self.special_token_ids_)

    @cached_property
    def cls_token_(self):
        return "CLS"

    @cached_property
    def pad_token_(self):
        return "PAD"

    @cached_property
    def special_tokens_(self):
        special_tokens = [self.cls_token_, self.pad_token_]

        return special_tokens

    @cached_property
    def vocabulary_size_(self):
        vocabulary_size = self.num_lowercase_letters + len(self.special_tokens_)

        return vocabulary_size

    @cached_property
    def cls_token_id_(self):
        return 0

    @cached_property
    def pad_token_id_(self):
        return 1

    @cached_property
    def special_token_ids_(self):
        special_token_ids = [self.cls_token_id_, self.pad_token_id_]

        return special_token_ids


class Tokenizer(p.BaseModel):
    config: TokenizerConfig

    def __post_init__(self):
        assert len(self.vocabulary_) == self.config.vocabulary_size_
        assert self.cls_token_id_ == self.config.cls_token_id_
        assert self.pad_token_id_ == self.config.pad_token_id_

    @property
    def cls_token_(self):
        return self.config.cls_token_

    @property
    def pad_token_(self):
        return self.config.pad_token_

    @property
    def special_tokens_(self):
        return self.config.special_tokens_

    @cached_property
    def ascii_lowercase_letters_(self):
        return string.ascii_lowercase[: self.config.num_lowercase_letters]

    @cached_property
    def vocabulary_(self):
        vocabulary = list(self.special_tokens_)
        vocabulary.extend(self.ascii_lowercase_letters_)

        return vocabulary

    @cached_property
    def token_to_token_id_(self):
        token_to_token_id = {t: i for i, t in enumerate(self.vocabulary_)}

        return token_to_token_id

    @cached_property
    def cls_token_id_(self):
        cls_token_id = self.token_to_token_id_[self.cls_token_]

        return cls_token_id

    @cached_property
    def pad_token_id_(self):
        pad_token_id = self.token_to_token_id_[self.pad_token_]

        return pad_token_id

    def encode(self, tokens: t.Collection[str]) -> t.List[int]:
        extended_tokens = [self.cls_token_]
        extended_tokens.extend(tokens)

        repeat_start = None
        token_ids = []
        for i, c in enumerate(extended_tokens, start=-1):
            token_id: t.Optional[int] = None

            if repeat_start is None:
                if c == "{":
                    # (`token_ids` should always be non-empty since the first entry in
                    # `extended_tokens` is `self.cls_token_`.)
                    assert token_ids
                    repeat_start = i
                    token_id = -1
                else:
                    token_id = self.token_to_token_id_.get(c)
            else:
                if c == "}":
                    repeat_str = "".join(extended_tokens[repeat_start + 2 : i + 1])
                    repeat = int(repeat_str)
                    if repeat < 1:
                        raise ValueError(f"repeat < 1\n{tokens = }\n{repeat = }")

                    token_id = token_ids[-1]
                    token_ids.extend([token_id] * (repeat - 1))

                    repeat_start = None

                token_id = -1

            if token_id is None:
                raise ValueError(f"unsupported token\n{tokens = }\n{i = }\n{c = }")

            if token_id >= 0:
                token_ids.append(token_id)

        if repeat_start is not None:
            raise ValueError(f"missing }}\n{tokens = }")

        return token_ids

    def decode(self, token_ids: t.Collection[int]) -> t.List[str]:
        if len(token_ids) < 1:
            raise ValueError(f"len(token_ids) < 1\n{len(token_ids) = }")

        tokens = []
        only_pad_tokens_remain = False
        for i, token_id in enumerate(token_ids):
            if i == 0:
                cls_token_id = self.cls_token_id_
                if token_id != cls_token_id:
                    raise ValueError(f"token_id != cls_token_id\n{token_id = }\n{cls_token_id = }")

                continue

            if not (0 <= token_id < len(self.vocabulary_)):
                raise ValueError(f"unsupported token_id\n{token_ids = }\n{i = }\n{token_id = }")

            if token_id == self.pad_token_id_:
                only_pad_tokens_remain = True
                continue

            if only_pad_tokens_remain:
                raise ValueError(f"non-pad token encountered after pad token\n{token_ids = }\n{i = }\n{token_id = }")

            token = self.vocabulary_[token_id]
            tokens.append(token)

        return tokens


def validate_token_ids(config: TokenizerConfig, token_ids):
    num_sequences = len(token_ids)

    unique_sequence_lengths = {len(token_ids_) for token_ids_ in token_ids}
    max_sequence_length = max(unique_sequence_lengths)

    if len(unique_sequence_lengths) > 1:
        padded_token_ids = []
        for token_ids_ in token_ids:
            padded_token_ids_ = token_ids_
            if (num_pad_tokens := max_sequence_length - len(padded_token_ids_)) > 0:
                padded_token_ids_ = list(padded_token_ids_)
                padded_token_ids_.extend([config.pad_token_id_] * num_pad_tokens)

            padded_token_ids.append(padded_token_ids_)

        token_ids = padded_token_ids

    token_ids = validate_tensor(token_ids, (num_sequences, max_sequence_length), torch.int64, name="token_ids")

    return token_ids
