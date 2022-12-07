##########################################
# File: data.py                          #
# Copyright Richard Stebbing 2022.       #
# Distributed under the MIT License.     #
# (See accompany file LICENSE or copy at #
#  http://opensource.org/licenses/MIT)   #
##########################################

import textwrap
import typing as t
from functools import cached_property

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset

from workshop.ext import pydantic as p

from .tokenizer import Tokenizer, TokenizerConfig, validate_token_ids


class DataConfig(p.BaseModel):
    positive_character_set: t.FrozenSet[str] = frozenset(["a", "b"])


class Item(p.BaseModel):
    token_ids: t.List[int]
    label: int


class Batch(p.BaseModel):
    token_ids: torch.Tensor
    labels: torch.Tensor


def collate_items(config: TokenizerConfig, items: t.List[Item]) -> Batch:
    token_ids = [i.token_ids for i in items]
    token_ids = validate_token_ids(config, token_ids)
    num_sequences = token_ids.shape[0]

    labels = torch.tensor([i.label for i in items], dtype=token_ids.dtype)
    assert labels.shape == (num_sequences,)

    batch = Batch(token_ids=token_ids, labels=labels)

    return batch


class AllFixedLengthSequences(p.BaseModel, TorchDataset):
    tokenizer_config: TokenizerConfig
    data_config: DataConfig

    sequence_length: p.PositiveInt

    @cached_property
    def tokenizer_(self):
        tokenizer = Tokenizer(config=self.tokenizer_config)

        return tokenizer

    def __getitem__(self, index) -> Item:
        num_lowercase_letters = self.tokenizer_config.num_lowercase_letters
        num_special_tokens = len(self.tokenizer_config.special_tokens_)

        token_ids_r = []
        for _ in range(self.sequence_length - 1):
            index, token_id_0 = divmod(index, num_lowercase_letters)
            token_ids_r.append(token_id_0 + num_special_tokens)

        token_ids = [self.tokenizer_.cls_token_id_]
        token_ids.extend(reversed(token_ids_r))

        tokens = self.tokenizer_.decode(token_ids)
        assert self.tokenizer_.encode(tokens) == token_ids

        label = int(all(token in tokens for token in self.data_config.positive_character_set))

        item = Item(token_ids=token_ids, label=label)

        return item

    def __len__(self):
        return self._len

    @cached_property
    def _len(self):
        len_ = self.tokenizer_config.num_lowercase_letters ** (self.sequence_length - 1)

        return len_


class VariableLengthSequences(p.BaseModel):
    tokenizer_config: TokenizerConfig
    data_config: DataConfig

    max_sequence_length: p.PositiveInt
    alpha: p.PositiveFloat
    batch_size: p.PositiveInt
    num_batches_per_epoch: p.PositiveInt
    seed: int

    train_batch_index: int = 0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if any(n <= 0 for n in self.configuration_budgets_):
            batch_size = self.batch_size
            num_configurations = self.num_configurations_
            data_config = self.data_config

            raise ValueError(
                textwrap.dedent(
                    f"""\
                    batch_size must be greater than or equal to num_configurations (from data_config.positive_character_set)
                    {batch_size = }
                    {num_configurations = }
                    {data_config = }"""
                )
            )

    @cached_property
    def tokenizer_(self):
        tokenizer = Tokenizer(config=self.tokenizer_config)

        return tokenizer

    @cached_property
    def positive_characters_(self):
        positive_characters = sorted(self.data_config.positive_character_set)

        return positive_characters

    @cached_property
    def other_characters_(self):
        other_characters = sorted(set(self.tokenizer_.ascii_lowercase_letters_).difference(self.positive_characters_))

        return other_characters

    @cached_property
    def num_configurations_(self):
        num_configurations = 2 ** len(self.positive_characters_)

        return num_configurations

    @cached_property
    def configuration_budgets_(self):
        effective_num_configurations = self.num_configurations_ + len(self.positive_characters_) + 1
        num_items_per_configuration, remainder = divmod(self.batch_size, effective_num_configurations)

        effective_configuration_budgets = [num_items_per_configuration] * effective_num_configurations
        for i in range(remainder):
            effective_configuration_budgets[i] += 1

        configuration_budgets = effective_configuration_budgets[: self.num_configurations_ - 1] + [
            sum(effective_configuration_budgets[self.num_configurations_ - 1 :])
        ]

        return configuration_budgets

    @cached_property
    def uniform_distributions_(self):
        uniform_distributions = {}
        for n in range(1, len(self.positive_characters_) + 1):
            p = np.ones(n)
            p /= n
            uniform_distributions[n] = p

        return uniform_distributions

    def __iter__(self):
        return self

    def __next__(self) -> Batch:
        if self.train_batch_index >= self.num_batches_per_epoch:
            self.train_batch_index = 0

            raise StopIteration

        random_state = np.random.RandomState(self.seed)
        configuration_budgets = list(self.configuration_budgets_)

        items = []
        for _ in range(self.batch_size):
            available_configurations = [i for i, r in enumerate(configuration_budgets) if r > 0]
            available_configuration_index = random_state.randint(len(available_configurations))
            configuration = available_configurations[available_configuration_index]
            configuration_budgets[configuration] -= 1

            positive_characters_subset = []
            for c in self.positive_characters_:
                configuration, include = divmod(configuration, 2)
                if include:
                    positive_characters_subset.append(c)

            min_sequence_length = max(len(positive_characters_subset), 1)
            sequence_length = random_state.randint(min_sequence_length, self.max_sequence_length + 1)
            assert min_sequence_length <= sequence_length <= self.max_sequence_length

            tokens = []

            if len(positive_characters_subset) > 0:
                num_positive_characters = random_state.randint(len(positive_characters_subset), sequence_length + 1)
                assert len(positive_characters_subset) <= num_positive_characters <= sequence_length

                positive_character_counts = np.ones(len(positive_characters_subset), dtype=np.int_)
                remaining_num_positive_characters = num_positive_characters - len(positive_characters_subset)
                uniform = self.uniform_distributions_[len(positive_characters_subset)]
                p = random_state.dirichlet(self.alpha * uniform)
                positive_character_counts += random_state.multinomial(remaining_num_positive_characters, p)
                assert positive_character_counts.sum() == num_positive_characters

                for c, n in zip(positive_characters_subset, positive_character_counts):
                    tokens.extend([c] * n)

                sequence_length -= num_positive_characters

            if sequence_length > 0:
                other_character_indices = random_state.randint(len(self.other_characters_), size=sequence_length)
                tokens.extend([self.other_characters_[i] for i in other_character_indices])

            tokens.sort()

            token_ids = self.tokenizer_.encode(tokens)

            label = int(all(token in tokens for token in self.positive_characters_))

            item = Item(token_ids=token_ids, label=label)
            items.append(item)

        batch = collate_items(self.tokenizer_config, items)

        next_seed = random_state.randint(2**32)
        self.seed = next_seed

        self.train_batch_index += 1

        return batch
