##########################################
# File: model.py                         #
# Copyright Richard Stebbing 2022.       #
# Distributed under the MIT License.     #
# (See accompany file LICENSE or copy at #
#  http://opensource.org/licenses/MIT)   #
##########################################

import abc
import typing as t
from enum import Enum
from functools import cached_property

import numpy as np
import torch
from torch import nn

from workshop.ext import pydantic as p
from workshop.ext.torch import RootMeanSquareLayerNorm, validate_tensor

from .tokenizer import TokenizerConfig, validate_token_ids


class LayerNorm(str, Enum):
    nn = "nn"
    rms = "rms"


LAYER_NORMS = {
    LayerNorm.nn: nn.LayerNorm,
    LayerNorm.rms: RootMeanSquareLayerNorm,
}

assert all(layer_norm in LAYER_NORMS for layer_norm in LayerNorm.__members__)


class EmbeddingsInitStrategy(str, Enum):
    default = "default"
    linear_like = "linear_like"


class AttentionInitStrategy(str, Enum):
    default = "default"
    output_fan_out = "output_fan_out"


class FeedForwardInitStrategy(str, Enum):
    default = "default"
    output_fan_out = "output_fan_out"


class NetworkConfig(p.BaseModel):
    hidden_size: p.PositiveInt = 2
    num_attention_heads: p.NonNegativeInt = 2
    no_attend_cls: bool = False
    layer_norm: LayerNorm = LayerNorm.rms
    attention_head_size: p.NonNegativeInt = 1
    feed_forward_size: p.NonNegativeInt = 2
    dropout_rate: p.NonNegativeFloat = 0.0
    layer_norm_eps: p.NonNegativeFloat = 1e-5
    embeddings_init_strategy: EmbeddingsInitStrategy = EmbeddingsInitStrategy.default
    attention_init_strategy: AttentionInitStrategy = AttentionInitStrategy.default
    feed_forward_init_strategy: FeedForwardInitStrategy = FeedForwardInitStrategy.default

    @cached_property
    def attention_size_(self) -> int:
        attention_size = self.num_attention_heads * self.attention_head_size

        return attention_size

    @cached_property
    def layer_norm_cls_(self):
        layer_norm_cls = LAYER_NORMS[self.layer_norm]

        return layer_norm_cls


class ModelConfig(p.BaseModel):
    tokenizer: TokenizerConfig = TokenizerConfig()
    network: NetworkConfig = NetworkConfig()


class AttentionResult(p.BaseModel):
    Q: torch.Tensor
    K: torch.Tensor
    V: torch.Tensor
    attention_values: torch.Tensor
    output: torch.Tensor


class SelfAttentionBlockResult(p.BaseModel):
    attention_normalized_hidden_states: t.Optional[torch.Tensor] = None
    attention_result: t.Optional[AttentionResult] = None
    attention_output_hidden_states: t.Optional[torch.Tensor] = None
    feed_forward_normalized_hidden_states: t.Optional[torch.Tensor] = None
    feed_forward_output: t.Optional[torch.Tensor] = None
    feed_forward_output_hidden_states: t.Optional[torch.Tensor] = None
    output_hidden_states: torch.Tensor


class ModelResult(p.BaseModel):
    token_ids: torch.Tensor
    pairwise_attention_mask: torch.Tensor
    embeddings: torch.Tensor
    self_attention_block_result: SelfAttentionBlockResult
    output_logits: torch.Tensor


class Module(nn.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def post_init_reset_parameters(self):
        raise NotImplementedError


class Model(Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config

        self.embeddings = Embeddings(config)
        self.self_attention_block = SelfAttentionBlock(config.network)
        self.binary_linear = BinaryLinear(config.network)

    def post_init_reset_parameters(self):
        self.embeddings.post_init_reset_parameters()
        self.self_attention_block.post_init_reset_parameters()

    @cached_property
    def special_token_ids_(self):
        special_token_ids = torch.tensor(self.config.tokenizer.special_token_ids_, dtype=torch.int64)

        return special_token_ids

    def forward(self, token_ids):
        c = self.config.network

        token_ids = validate_token_ids(self.config.tokenizer, token_ids)
        num_sequences, max_sequence_length = token_ids.shape
        assert max_sequence_length > 0

        embeddings: torch.Tensor = self.embeddings(token_ids)
        assert embeddings.shape == (num_sequences, max_sequence_length, c.hidden_size)

        is_special_token = (token_ids.unsqueeze(2) == self.special_token_ids_).any(dim=2)
        assert is_special_token.shape == (num_sequences, max_sequence_length)

        pairwise_attention_mask = torch.ones(
            (num_sequences, max_sequence_length, max_sequence_length), dtype=torch.bool
        )
        special_token_sequence_indices, special_token_token_indices = torch.nonzero(is_special_token).T
        pairwise_attention_mask[special_token_sequence_indices, :, special_token_token_indices] = False

        if not c.no_attend_cls:
            is_cls_token = token_ids == self.config.tokenizer.cls_token_id_
            assert is_cls_token.shape == (num_sequences, max_sequence_length)
            cls_token_sequence_indices, cls_token_token_indices = torch.nonzero(is_cls_token).T
            pairwise_attention_mask[cls_token_sequence_indices, :, cls_token_token_indices] = True

        self_attention_block_result: SelfAttentionBlockResult = self.self_attention_block(
            embeddings, pairwise_attention_mask
        )
        transformed_embeddings = self_attention_block_result.output_hidden_states
        assert transformed_embeddings.shape == (num_sequences, max_sequence_length, c.hidden_size)

        assert (token_ids[:, 0] == self.config.tokenizer.cls_token_id_).all()
        transformed_cls_embeddings = transformed_embeddings[:, 0]
        assert transformed_cls_embeddings.shape == (num_sequences, c.hidden_size)

        output_logits: torch.Tensor = self.binary_linear(transformed_cls_embeddings)
        assert output_logits.shape == (num_sequences,)

        result = ModelResult(
            token_ids=token_ids,
            pairwise_attention_mask=pairwise_attention_mask,
            embeddings=embeddings,
            self_attention_block_result=self_attention_block_result,
            output_logits=output_logits,
        )

        return result


class Embeddings(Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config

        c = config.network

        self.embeddings = nn.Embedding(config.tokenizer.vocabulary_size_, c.hidden_size)

        self.dropout = nn.Dropout(c.dropout_rate)

    def post_init_reset_parameters(self):
        config = self.config

        c = config.network

        if c.embeddings_init_strategy == EmbeddingsInitStrategy.linear_like:
            nn.init.kaiming_uniform_(self.embeddings.weight, a=np.sqrt(5))

        # (Set the "PAD" token to the origin for consistency.)
        with torch.no_grad():
            self.embeddings.weight[config.tokenizer.pad_token_id_] = 0.0

    def forward(self, token_ids):
        token_ids = validate_token_ids(self.config.tokenizer, token_ids)

        hidden_states = self.embeddings(token_ids)

        # This is different to Hugging Face's implementation of LayoutLM and the implementation of T5.
        # Specifically:
        # - Unlike the LayoutLM implementation, layer normalization is *not* applied within the
        #   `Embeddings` class. The T5 convention (where normalization is applied within blocks) is
        #   done instead.
        # - Unlike the T5 implementation, dropout is applied within this class instead of within
        #   the model that composes them with the line:
        #
        #     hidden_states = self.dropout(inputs_embeds)
        #
        # References:
        # transformers/models/t5/modeling_t5.py:T5Stack
        hidden_states = self.dropout(hidden_states)

        return hidden_states


class SelfAttentionBlock(Module):
    def __init__(self, config: NetworkConfig):
        super().__init__()

        self.config = config

        self.attention_layer_norm = (
            config.layer_norm_cls_(config.hidden_size, eps=config.layer_norm_eps)
            if config.attention_size_ > 0 and config.layer_norm_eps > 0.0
            else None
        )
        self.attention = Attention(config) if config.attention_size_ > 0 else None

        self.feed_forward_layer_norm = (
            config.layer_norm_cls_(config.hidden_size, eps=config.layer_norm_eps)
            if config.feed_forward_size > 0 and config.layer_norm_eps > 0.0
            else None
        )
        self.feed_forward = FeedForward(config) if config.feed_forward_size > 0 else None

        self.dropout = nn.Dropout(config.dropout_rate)

    def post_init_reset_parameters(self):
        if self.attention is not None:
            self.attention.post_init_reset_parameters()

        if self.feed_forward is not None:
            self.feed_forward.post_init_reset_parameters()

    def forward(self, hidden_states, pairwise_attention_mask=None) -> SelfAttentionBlockResult:
        c = self.config

        result = SelfAttentionBlockResult(output_hidden_states=hidden_states)

        hidden_states, pairwise_attention_mask = validate_hidden_states_and_pairwise_attention_mask(
            c, hidden_states, pairwise_attention_mask
        )
        num_sequences, max_sequence_length, _ = hidden_states.shape

        if self.attention is not None:
            # Attention is *always* done on `normalized_hidden_states`.
            # This is the case with LayoutLM in Hugging Face too, except the
            # normalization is pushed into the `LayoutLMEmbeddings` class.
            #
            # References:
            # transformers/models/t5/modeling_t5.py:T5LayerSelfAttention
            # transformers/models/layoutlm/modeling_layoutlm.py:LayoutLMEmbeddings
            #
            # Separately, if `self.attention_layer_norm` is a `LayerNorm`, then:
            #
            #   self.attention_layer_norm(hidden_states)
            #
            # is equivalent to:
            #
            #   means = hidden_states.mean(axis=-1, keepdims=True)
            #   vars_ = hidden_states.var(axis=-1, unbiased=False, keepdims=True)
            #   rsqrts = torch.rsqrt(vars_ + c.layer_norm_eps)
            #   scales = rsqrts
            #   biases = -rsqrts * means
            #   normalized_hidden_states = (
            #       hidden_states * scales + biases
            #   ) * self.attention_layer_norm.weight + self.attention_layer_norm.bias
            #
            # Reference:
            # aten/src/ATen/native/cpu/layer_norm_kernel.cpp
            normalized_hidden_states = (
                self.attention_layer_norm(hidden_states) if self.attention_layer_norm is not None else hidden_states
            )
            result.attention_normalized_hidden_states = normalized_hidden_states

            attention_result: AttentionResult = self.attention(normalized_hidden_states, pairwise_attention_mask)
            assert attention_result.output.shape == (num_sequences, max_sequence_length, c.hidden_size)
            result.attention_result = attention_result

            hidden_states = hidden_states + self.dropout(attention_result.output)
            result.attention_output_hidden_states = hidden_states

        if self.feed_forward is not None:
            normalized_hidden_states = (
                self.feed_forward_layer_norm(hidden_states)
                if self.feed_forward_layer_norm is not None
                else hidden_states
            )
            result.feed_forward_normalized_hidden_states = normalized_hidden_states

            output: torch.Tensor = self.feed_forward(normalized_hidden_states)
            assert output.shape == (num_sequences, max_sequence_length, c.hidden_size)
            result.feed_forward_output = output

            hidden_states = hidden_states + self.dropout(output)
            result.feed_forward_output_hidden_states = hidden_states

        result.output_hidden_states = hidden_states

        return result


class Attention(Module):
    def __init__(self, config: NetworkConfig):
        super().__init__()

        if config.attention_size_ <= 0:
            raise ValueError(f"config.attention_size_ <= 0\n{config.attention_size_ = }")

        self.config = config

        self.query = nn.Linear(config.hidden_size, config.attention_size_, bias=False)
        self.key = nn.Linear(config.hidden_size, config.attention_size_, bias=False)
        self.value = nn.Linear(config.hidden_size, config.attention_size_, bias=False)
        self.output = nn.Linear(config.attention_size_, config.hidden_size, bias=False)

        self.dropout = nn.Dropout(config.dropout_rate)

        rsqrt_attention_head_size = torch.rsqrt(torch.tensor(config.attention_head_size))
        self.register_buffer("rsqrt_attention_head_size", rsqrt_attention_head_size)

    def post_init_reset_parameters(self):
        config = self.config

        if config.attention_init_strategy == AttentionInitStrategy.output_fan_out:
            nn.init.kaiming_uniform_(self.output.weight, a=np.sqrt(5), mode="fan_out")

    def forward(self, normalized_hidden_states, pairwise_attention_mask=None):
        c = self.config

        normalized_hidden_states, pairwise_attention_mask = validate_hidden_states_and_pairwise_attention_mask(
            c, normalized_hidden_states, pairwise_attention_mask, name="normalized_hidden_states"
        )
        num_sequences, max_sequence_length, _ = normalized_hidden_states.shape

        def project_reshape_and_transpose(linear: nn.Module):
            X: torch.Tensor = linear(normalized_hidden_states)
            assert X.shape == (num_sequences, max_sequence_length, c.attention_size_)

            Y = X.reshape(num_sequences, max_sequence_length, c.num_attention_heads, c.attention_head_size).transpose(
                1, 2
            )
            assert Y.shape == (num_sequences, c.num_attention_heads, max_sequence_length, c.attention_head_size)

            return Y

        Q = project_reshape_and_transpose(self.query)
        K = project_reshape_and_transpose(self.key)
        V = project_reshape_and_transpose(self.value)

        attention_scores = (Q @ K.transpose(2, 3)) * self.rsqrt_attention_head_size
        assert attention_scores.shape == (
            num_sequences,
            c.num_attention_heads,
            max_sequence_length,
            max_sequence_length,
        )

        min_attention_score = torch.finfo(attention_scores.dtype).min
        additive_pairwise_exclude_mask = (min_attention_score * ~pairwise_attention_mask).unsqueeze(1)
        assert additive_pairwise_exclude_mask.shape == (num_sequences, 1, max_sequence_length, max_sequence_length)

        attention_scores = attention_scores + additive_pairwise_exclude_mask

        attention_weights = attention_scores.softmax(axis=-1)  # pyright: ignore
        assert attention_weights.shape == (
            num_sequences,
            c.num_attention_heads,
            max_sequence_length,
            max_sequence_length,
        )

        # References:
        # transformers/models/t5/modeling_t5.py:T5Attention
        # transformers/models/layoutlm/modeling_layoutlm.py:LayoutLMSelfAttention
        attention_weights = self.dropout(attention_weights)

        attention_values = attention_weights @ V
        assert attention_values.shape == (
            num_sequences,
            c.num_attention_heads,
            max_sequence_length,
            c.attention_head_size,
        )

        attention_values_ = attention_values.transpose(1, 2).reshape(
            num_sequences, max_sequence_length, c.attention_size_
        )
        assert attention_values_.shape == (num_sequences, max_sequence_length, c.attention_size_)

        output = self.output(attention_values_)
        assert output.shape == (num_sequences, max_sequence_length, c.hidden_size)

        result = AttentionResult(
            Q=Q,
            K=K,
            V=V,
            attention_values=attention_values_,
            output=output,
        )

        return result


class FeedForward(Module):
    def __init__(self, config: NetworkConfig):
        super().__init__()

        if config.feed_forward_size <= 0:
            raise ValueError(f"config.feed_forward_size <= 0\n{config.feed_forward_size = }")

        self.config = config

        self.input_ = nn.Linear(config.hidden_size, config.feed_forward_size, bias=False)
        self.output = nn.Linear(config.feed_forward_size, config.hidden_size, bias=False)
        self.activation = nn.GELU()

        self.dropout = nn.Dropout(config.dropout_rate)

    def post_init_reset_parameters(self):
        config = self.config

        if config.feed_forward_init_strategy == FeedForwardInitStrategy.output_fan_out:
            nn.init.kaiming_uniform_(self.output.weight, a=np.sqrt(5), mode="fan_out")

    def forward(self, normalized_hidden_states):
        c = self.config

        normalized_hidden_states = validate_hidden_states(c, normalized_hidden_states, name="normalized_hidden_states")
        num_sequences, max_sequence_length, _ = normalized_hidden_states.shape

        hidden_states = self.input_(normalized_hidden_states)
        assert hidden_states.shape == (num_sequences, max_sequence_length, c.feed_forward_size)

        hidden_states = self.activation(hidden_states)
        assert hidden_states.shape == (num_sequences, max_sequence_length, c.feed_forward_size)

        # Dropout is applied after the activation.
        #
        # References:
        # https://stackoverflow.com/a/40295999
        # transformers/models/t5/modeling_t5.py:T5DenseActDense
        hidden_states = self.dropout(hidden_states)

        output = self.output(hidden_states)
        assert output.shape == (num_sequences, max_sequence_length, c.hidden_size)

        return output


class BinaryLinear(Module):
    def __init__(self, config: NetworkConfig):
        super().__init__()

        self.config = config

        self.linear = nn.Linear(config.hidden_size, 1, bias=False)

        self.dropout = nn.Dropout(config.dropout_rate)

    def post_init_reset_parameters(self):
        pass

    def forward(self, hidden_states):
        c = self.config

        hidden_states = validate_hidden_states(c, hidden_states)
        num_sequences = hidden_states.shape[0]

        assert hidden_states.shape == (num_sequences, c.hidden_size)

        hidden_states = self.dropout(hidden_states)

        output_logits = self.linear(hidden_states)
        assert output_logits.shape == (num_sequences, 1)

        output_logits = output_logits[:, 0]

        return output_logits


def validate_hidden_states_and_pairwise_attention_mask(
    config: NetworkConfig, hidden_states, pairwise_attention_mask, *, name="hidden_states"
):
    hidden_states = validate_hidden_states(config, hidden_states, name=name)

    if hidden_states.ndim < 2:
        hidden_states = hidden_states.unsqueeze(0)

    assert hidden_states.ndim >= 2

    expected_pairwise_attention_mask_shape = tuple(hidden_states.shape[:-1]) + (hidden_states.shape[-2],)

    if pairwise_attention_mask is None:
        attention_mask = torch.ones(hidden_states.shape[:-1], dtype=torch.bool)
        pairwise_attention_mask = attention_mask.unsqueeze(2) & attention_mask.unsqueeze(1)

    pairwise_attention_mask = validate_tensor(
        pairwise_attention_mask, expected_pairwise_attention_mask_shape, torch.bool, name="pairwise_attention_mask"
    )

    return hidden_states, pairwise_attention_mask


def validate_hidden_states(config: NetworkConfig, hidden_states, *, name="hidden_states"):
    hidden_states = validate_tensor(hidden_states, config.hidden_size, torch.float32, name=name)

    return hidden_states
