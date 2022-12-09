# High-Level Walkthrough

Next: [`self_attention_block` Walkthrough ⧉](02-self-attention-block-walkthrough.md), Up: [Transformer Sequence Classification ⧉](.)

Table of Contents
1. [Introduction](#introduction)
2. [The `token_ids`](#the-token_ids)
3. [The `embeddings`](#the-embeddings)
4. [The `transformed_embeddings`](#the-transformed_embeddings)
5. [The `output_logits`](#the-output_logits)

## Introduction

To understand the model it is instructive to step through the sequence classifier considering two example strings: `"aac"` and `"baac"`.

All visualizations in this document can be reproduced from scratch by training and testing the model via the single command:
``` bash
python -m workshop.experiments.transformer_sequence_classification run walkthrough
```

Once complete, the figures should be available at:
``` bash
data/experiments/transformer_sequence_classification/walkthrough/01__no_attend_cls=True__layer_norm_eps=0.0/model_seed=5/figures
```

The visualizations can be run interactively too by running the `eval` command echoed to `stderr`:
``` bash
python -m workshop.experiments.transformer_sequence_classification.main \
  eval \
  --directory data/experiments/transformer_sequence_classification/walkthrough/01__no_attend_cls=True__layer_norm_eps=0.0/model_seed=5 \
  --max-num-incorrect-strings 10 \
  --visualize \
  aac baac
```
and additional strings can be evaluated by appending them to the command too. For example:
``` bash
python -m workshop.experiments.transformer_sequence_classification.main \
  eval \
  --directory data/experiments/transformer_sequence_classification/walkthrough/01__no_attend_cls=True__layer_norm_eps=0.0/model_seed=5 \
  --max-num-incorrect-strings 10 \
  --visualize \
  aac baac a b ab 'ac{100}'
```
additionally evaluates and visualizes `"a"`, `"b"`, `"ab"`, and `"ab{100}"` (which is supported notation and equivalent to `"a" + "b" * 100` in Python).

## The `token_ids`

Evaluation ([`Experiment.eval_` ⧉](https://github.com/rstebbing/workshop/blob/5b6dff53d8adc4c83707f8da80fc2c2b08f08c76/py/src/workshop/experiments/transformer_sequence_classification/experiment.py#L393)) begins by encoding each input string into a sequence of integers that are represented by a single [`token_ids` ⧉](https://github.com/rstebbing/workshop/blob/5b6dff53d8adc4c83707f8da80fc2c2b08f08c76/py/src/workshop/experiments/transformer_sequence_classification/experiment.py#L394-L410) tensor. In more detail:

1. Start with two `strings`:
   ``` python
   "aac"
   "baac"
   ```
2. In [`Tokenizer.encode` ⧉](https://github.com/rstebbing/workshop/blob/5b6dff53d8adc4c83707f8da80fc2c2b08f08c76/py/src/workshop/experiments/transformer_sequence_classification/tokenizer.py#L108) each string is split into individual characters (*tokens*) and an extra classification token (`"CLS"`) is prepended to each sequence:
   ``` python
   ["CLS", "a", "a", "c"]
   ["CLS", "b", "a", "a", "c"]
   ```
   (The role of the `"CLS"` token is explained later.)
3. Each token is replaced with an integer:
   ``` python
   [0, 2, 2, 4]
   [0, 3, 2, 2, 4]
   ```
4. In [`validate_token_ids` ⧉](https://github.com/rstebbing/workshop/blob/5b6dff53d8adc4c83707f8da80fc2c2b08f08c76/py/src/workshop/experiments/transformer_sequence_classification/tokenizer.py#L181) the two sequences are combined into a single tensor of shape
   ``` python
   (num_sequences, max_sequence_length) == (2, 5)
   ```
   by appending a padding token (`"PAD"`/`1`) to the first sequence:
   ``` python
   tensor([[0, 2, 2, 4, 1],
           [0, 3, 2, 2, 4]])
   ```

The `token_ids` are the one and only argument provided to the model.

## The `embeddings`

The first step in the forward pass ([`Model.forward` ⧉](https://github.com/rstebbing/workshop/blob/5b6dff53d8adc4c83707f8da80fc2c2b08f08c76/py/src/workshop/experiments/transformer_sequence_classification/model.py#L126)) is to build the [`embeddings` ⧉](https://github.com/rstebbing/workshop/blob/5b6dff53d8adc4c83707f8da80fc2c2b08f08c76/py/src/workshop/experiments/transformer_sequence_classification/model.py#L133-L134) tensor of shape

``` python
(num_sequences, max_sequence_length, hidden_size) == (2, 5, 2)
```
that the Transformer Block will operate on:
``` python
embeddings: torch.Tensor = self.embeddings(token_ids)
assert embeddings.shape == (num_sequences, max_sequence_length, c.hidden_size)
```

In short:
- Each token (`"CLS"`, `"PAD"`, `"a"`, `"b"`, and `"c"`) has an embedding of shape `(hidden_size,) == (2,)`:
  <p align="center">
    <img src="https://github.com/rstebbing/workshop/blob/main/experiments/transformer_sequence_classification/figures/walkthrough/8-1.png" width=80% alt="walkthrough embeddings"/>
  </p>
- `embeddings` is computed by replacing each element in `token_ids` with the corresponding embedding:
  ``` python
  tensor([[[-0.6023, -1.0448],
           [ 0.1197,  2.3601],
           [ 0.1197,  2.3601],
           [ 1.5460, -0.8536],
           [ 0.0000,  0.0000]],

          [[-0.6023, -1.0448],
           [-1.5085, -2.1599],
           [ 0.1197,  2.3601],
           [ 0.1197,  2.3601],
           [ 1.5460, -0.8536]]])
  ```

  For simplicity, positional information about where each character occurs in the sequence is *not* included. The embeddings for each `"a"` are therefore identical and will be transformed identically too. In practice, positional information is conveyed via positional embeddings of shape `(hidden_size,)` that are added to each token (e.g. [BERT ⧉](https://arxiv.org/abs/1810.04805), [LayoutLM ⧉](https://arxiv.org/abs/1912.13318)), or pushed deeper into the attention mechanism itself (e.g. [T5 ⧉](https://arxiv.org/abs/1910.10683)).

  The embeddings for each sequence are shown below. (The `"{2}"` suffix in `"a{2}"` indicates that `"a"` occurs twice in each sequence.)

  <p align="center">
    <img src="https://github.com/rstebbing/workshop/blob/main/experiments/transformer_sequence_classification/figures/walkthrough/8-2.png" width=80% alt="walkthrough embeddings: 'aac'"/>
    <img src="https://github.com/rstebbing/workshop/blob/main/experiments/transformer_sequence_classification/figures/walkthrough/8-3.png" width=80% alt="walkthrough embeddings: 'baac'"/>
  </p>

## The `transformed_embeddings`

The second step in the forward pass is to transform the `embeddings` tensor via the `self_attention_block` module (the Transformer Block) to produce the [`transformed_embeddings` ⧉](https://github.com/rstebbing/workshop/blob/5b6dff53d8adc4c83707f8da80fc2c2b08f08c76/py/src/workshop/experiments/transformer_sequence_classification/model.py#L145-L149) tensor of identical shape:
``` python
self_attention_block_result: SelfAttentionBlockResult = self.self_attention_block(
    embeddings, pairwise_attention_mask
)
transformed_embeddings = self_attention_block_result.output_hidden_states
assert transformed_embeddings.shape == (num_sequences, max_sequence_length, c.hidden_size)
```

The transformed embeddings for each token in each sequence are shown separately below. Importantly, the transformed `"CLS"` embeddings are *now* different for each sequence:
- For `"aac"` it is `tensor([-6.4319, -0.9393])` (down and to the left) and
- For `"baac"` it is `tensor([0.2414, 9.1150])` (up and to the right).

<p align="center">
  <img src="https://github.com/rstebbing/workshop/blob/main/experiments/transformer_sequence_classification/figures/walkthrough/8-17.png" width=80% alt="walkthrough transformed_embeddings: 'aac'"/>
  <img src="https://github.com/rstebbing/workshop/blob/main/experiments/transformer_sequence_classification/figures/walkthrough/8-18.png" width=80% alt="walkthrough transformed_embeddings: 'baac'"/>
</p>

## The `output_logits`

The third and final step in the forward pass is to compute a *logit* for each sequence. A positive value indicates that the sequence contains an `"a"` and a `"b"` (i.e. a final output of `1`), and a negative values indicates the opposite (i.e. `0`). The output logit for each sequence is defined simply as a linear projection of the transformed `"CLS"` embedding:
``` python
assert (token_ids[:, 0] == self.config.tokenizer.cls_token_id_).all()
transformed_cls_embeddings = transformed_embeddings[:, 0]
assert transformed_cls_embeddings.shape == (num_sequences, c.hidden_size)

output_logits: torch.Tensor = self.binary_linear(transformed_cls_embeddings)
assert output_logits.shape == (num_sequences,)
```

The `output_logits` for `"aac"` and `"baac"` respectively are:
``` python
tensor([-8.2945, 10.9587])
```

The transformed `"CLS"` embeddings for each sequence are shown with the `binary_linear` weights below. The output logit for `"baac"` is positive because its transformed `"CLS"` embedding is in the direction pointed to by the `binary_linear` weights (up and to the right), whereas the output logit for `"aac"` is negative because its transformed `"CLS"` embedding is in the *opposite* direction (down and to the left).

<p align="center">
  <img src="https://github.com/rstebbing/workshop/blob/main/experiments/transformer_sequence_classification/figures/walkthrough/8-15.png" width=80% alt="walkthrough cls_feed_forward_output_hidden_states"/>
</p>

In short, for each sequence:
1. The `embeddings` tensor is computed by replacing each token with its embedding.
2. The `transformed_embeddings` tensor is computed by transforming the `embeddings` via the `self_attention_block`.
3. The output logit is positive (indicating a final output of `1`) if its transformed `"CLS"` token is in the direction pointed to by the `binary_linear` classifier and negative (indicating a final output of `0`) otherwise.

So what *is* going on inside `self_attention_block`?
