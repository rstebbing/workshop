# `self_attention_block` Walkthrough

Next: [Generalization Errors ⧉](03-generalization-errors.md), Previous: [High-Level Walkthrough ⧉](01-high-level-walkthrough.md), Up: [Transformer Sequence Classification ⧉](.)

Table of Contents
1. [Introduction](#introduction)
2. [`attention`](#attention)
3. [`feed_forward`](#feed_forward)

## Introduction

The first step in `self_attention_block`, an instance of [`SelfAttentionBlock` ⧉](https://github.com/rstebbing/workshop/blob/5b6dff53d8adc4c83707f8da80fc2c2b08f08c76/py/src/workshop/experiments/transformer_sequence_classification/model.py#L230), is the application of attention to the `normalized_hidden_states` (i.e. the *normalized* `transformed_embeddings`) which is then added to the input `hidden_states` (`transformed_embeddings`):

``` python
attention_result: AttentionResult = self.attention(normalized_hidden_states, attention_mask)
assert attention_result.output.shape == (num_sequences, max_sequence_length, c.hidden_size)
result.attention_result = attention_result

hidden_states = hidden_states + self.dropout(attention_result.output)
```

To make this walkthrough simpler, no layer normalization is done so that `attention` is actually operating *directly* on `transformed_embeddings`.

## `attention`

1. The first step in `attention` is to compute the [`attention_weights` ⧉](https://github.com/rstebbing/workshop/blob/5b6dff53d8adc4c83707f8da80fc2c2b08f08c76/py/src/workshop/experiments/transformer_sequence_classification/model.py#L371-L395), a 4D tensor of shape
   ``` python
   (num_sequences, num_attention_heads, max_sequence_length, max_sequence_length) == (2, 2, 5, 5)
   ```
   where `attention_weights[i, j, k, l]` is the weight of the `l`th token toward the output value of the `k`th token for the `i`th sequence for the `j`th attention head.

   The `attention_weights` is computed by applying two *different* linear transformations to the input `normalized_hidden_states` (`transformed_embeddings`) to compute two tensors—`Q` (the queries) and `K` (the keys)—which are matrix multiplied together and passed through the softmax function:

   ``` python
   Q = project_reshape_and_transpose(self.query)
   K = project_reshape_and_transpose(self.key)
   ...
   attention_scores = (Q @ K.transpose(2, 3)) * self.rsqrt_attention_head_size
   ...
   attention_weights = attention_scores.softmax(axis=-1)
   ```

   `Q` and `K` both have shape
   ``` python
   (num_sequences, num_attention_heads, max_sequence_length, attention_head_size) == (2, 2, 5, 1)
   ```
   and the more positive the dot product of `Q[i, j, k]` and `K[i, j, l]` is, the closer `attention_weights[i, j, k, l]` is to one (and vice versa).

   In this example `num_attention_heads = 2` and `attention_head_size = 1`, making it possible to visualize the query for the `"CLS"` token and *all* keys as arrows on the x-axis in two plots (one for each head):

   <p align="center">
     <img src="https://github.com/rstebbing/workshop/blob/main/experiments/transformer_sequence_classification/figures/walkthrough/8-5.png" width=80% alt="walkthrough token_attention_keys_and_queries (head: 0)"/>
     <img src="https://github.com/rstebbing/workshop/blob/main/experiments/transformer_sequence_classification/figures/walkthrough/8-6.png" width=80% alt="walkthrough token_attention_keys_and_queries (head: 1)"/>
   </p>

   Considering *just* the first attention head, the `"CLS"` query attends to `"a"`s more than `"b"`s. This can be confirmed by inspecting `attention_weights[:, 0, 0]`, the attention weights for the first attention head for the `"CLS"` token in *both* sequences:
   ``` python
   tensor([[0.0000e+00, 4.9982e-01, 4.9982e-01, 3.5828e-04, 0.0000e+00],
           [0.0000e+00, 4.5179e-06, 4.9982e-01, 4.9982e-01, 3.5828e-04]])
   ```
   The majority of weight in both sequences is for the tokens corresponding to the two `"a"`s. There is zero weight given to the `"CLS"` tokens and `"PAD"` tokens so that neither can have non-zero attention weight and bias the `attention_values`.

   Considering *just* the second attention head, the `"CLS"` query attends to `"b"`s more than `"a"`s. This can be confirmed by inspecting `attention_weights[:, 1, 0]`:
   ``` python
   tensor([[0.0000e+00, 3.8669e-03, 3.8669e-03, 9.9227e-01, 0.0000e+00],
           [0.0000e+00, 9.9959e-01, 1.5737e-06, 1.5737e-06, 4.0383e-04]])
   ```
   The majority of weight in the second sequence (`"baac"`) is for the second token `"b"` (as expected), and the majority of weight in the first sequence is the `"c"` (because there is no `"b"` to attend to).

2. The second step in `attention` is to compute the `attention_values`. Toward this, a *third* linear transformation of the input `normalized_hidden_states` (`transformed_embeddings`) is evaluated: `V` (the values). It is combined with the attention weights via:
   ``` python
   attention_values = attention_weights @ V
   ```

   Like `Q` and `K`, the entries in `V` for each token can be visualized on the x-axis in two plots (one for each head):

   <p align="center">
     <img src="https://github.com/rstebbing/workshop/blob/main/experiments/transformer_sequence_classification/figures/walkthrough/8-7.png" width=80% alt="walkthrough token_attention_values (head: 0)"/>
     <img src="https://github.com/rstebbing/workshop/blob/main/experiments/transformer_sequence_classification/figures/walkthrough/8-8.png" width=80% alt="walkthrough token_attention_values (head: 1)"/>
   </p>

   For each attention head `"a"` and `"b"` have values of opposite sign. The attention values for the `"CLS"` token for each sequence (`attention_values[:, :, 0]`) can also be visualized which shows the combination of the `attention_weights` *and* `V`:

   <p align="center">
     <img src="https://github.com/rstebbing/workshop/blob/main/experiments/transformer_sequence_classification/figures/walkthrough/8-9.png" width=80% alt="walkthrough cls_attention_values (head: 0)"/>
     <img src="https://github.com/rstebbing/workshop/blob/main/experiments/transformer_sequence_classification/figures/walkthrough/8-10.png" width=80% alt="walkthrough cls_attention_values (head: 1)"/>
   </p>

   The attention values for `"CLS"` for `"aac"` and `"baac"` are identical for the first attention head because both sequences have `"a"`s. The attention values for the second head, however, are different:
   - For `"baac"` it is almost equal to the value for `"b"` because the `"b"` has weight very close to `1.0` (`9.9959e-01`).
   - For `"aac"` it is almost equal to the value for `"c"` becaues the `"c"` has weight very close to `1.0` (`9.9227e-01`).

3. The third and final step in `attention` is to transform the `attention_values` across all heads via another linear transformation:
   ``` python
   output = self.output(attention_values_)
   assert output.shape == (num_sequences, max_sequence_length, c.hidden_size)
   ```

   <p align="center">
     <img src="https://github.com/rstebbing/workshop/blob/main/experiments/transformer_sequence_classification/figures/walkthrough/8-11.png" width=80% alt="walkthrough cls_attention_output"/>
   </p>

   By construction, the `output` is in the same space as the input *unnormalized* `hidden_states` (`transformed_embeddings`) and is added directly to it. This is the first residual connection.

   <p align="center">
     <img src="https://github.com/rstebbing/workshop/blob/main/experiments/transformer_sequence_classification/figures/walkthrough/8-12.png" width=80% alt="walkthrough cls_attention_output_hidden_states"/>
   </p>

## `feed_forward`

The second step in `self_attention_block` is the application of a feed forward network to the (normalized) updated `hidden_states` that is (again) added back to the `hidden_states`:

``` python
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
```

The feed forward network is relatively standard: each token is projected to a vector of `feed_forward_size == 2` dimensions, an activation function is applied, and each token is then projected back to `hidden_size == 2` dimensions. The `output` of the feed forward network displaces either positive or negative sequences further from the origin:

<p align="center">
  <img src="https://github.com/rstebbing/workshop/blob/main/experiments/transformer_sequence_classification/figures/walkthrough/8-14.png" width=80% alt="walkthrough cls_feed_forward_outputs"/>
</p>

The `output` is added back to the updated `hidden_states` and gives the final transformed `"CLS"` embeddings shown already. (This is the second residual connection.)

<p align="center">
  <img src="https://github.com/rstebbing/workshop/blob/main/experiments/transformer_sequence_classification/figures/walkthrough/8-15.png" width=80% alt="walkthrough cls_feed_forward_output_hidden_states"/>
</p>

---

Next: [Generalization Errors ⧉](03-generalization-errors.md), Previous: [High-Level Walkthrough ⧉](01-high-level-walkthrough.md), Up: [Transformer Sequence Classification ⧉](.)
