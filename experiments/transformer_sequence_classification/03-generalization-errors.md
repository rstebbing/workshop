# Generalization Errors

Next: [Improving Generalization ⧉](04-improving-generalization.md), Previous: [`self_attention_block` Walkthrough ⧉](02-self-attention-block-walkthrough.md), Up: [Transformer Sequence Classification ⧉](.)

Table of Contents
1. [Introduction](#introduction)
2. [False Positives](#false-positives)
3. [False Negatives](#false-negatives)

## Introduction

The `model` demonstrated and examined in [High-Level Walkthrough ⧉](01-high-level-walkthrough.md) and [`self_attention_block` Walkthrough ⧉](02-self-attention-block-walkthrough.md) generalizes well to long strings—the test set contains strings up to 200 tokens in length. But this is not the case for every fitted model. In fact, for this particular experiment the accuracy is very dependent on the random seed used to initialize the various modules (e.g. `embeddings`) in the model. As will be shown later, other model hyperparameters like model dimensionality (`hidden_size`) impact this too.

A model that does *not* generalize well to longer sequences can be trained and tested via the single command:
``` bash
python -m workshop.experiments.transformer_sequence_classification run generalization_errors
```

Like before, the visualizations can be run interactively via too:
``` bash
python -m workshop.experiments.transformer_sequence_classification.main \
  eval \
  --directory data/experiments/transformer_sequence_classification/generalization_errors/01__layer_norm_eps=0.0/model_seed=2 \
  --max-num-incorrect-strings 10 \
  --visualize \
  a b c ab 'ab{100}c{55}' 'b{2}c{152}'
```

The transformed `"CLS"` embeddings for each sequence are shown below and the two errors—`"b{2}c{152}"` (false positive) and `"ab{100}c{55}"` (false negative)—are shown in purple and red respectively:

<p align="center">
  <img src="https://github.com/rstebbing/workshop/blob/main/experiments/transformer_sequence_classification/figures/generalization_errors/02-19.png" width=80% alt="generalization_errors cls_feed_forward_output_hidden_states"/>
</p>


## False Positives

The string `"b{2}c{152}"` is a false positive: the model outputs a positive logit when it should be negative. How is a false positive possible?

A false positive can occur when, for a given attention head, attention values for one or more tokens can be added together with positive weights to be *close* to the attention value of another token. The attention keys and `"CLS"` queries *and* attention values for both heads are shown below:

<p align="center">
  <img src="https://github.com/rstebbing/workshop/blob/main/experiments/transformer_sequence_classification/figures/generalization_errors/02-9.png" width=80% alt="generalization_errors token_attention_keys_and_queries (head: 0)"/>
  <img src="https://github.com/rstebbing/workshop/blob/main/experiments/transformer_sequence_classification/figures/generalization_errors/02-10.png" width=80% alt="generalization_errors token_attention_keys_and_queries (head: 1)"/>
  <img src="https://github.com/rstebbing/workshop/blob/main/experiments/transformer_sequence_classification/figures/generalization_errors/02-11.png" width=80% alt="generalization_errors token_attention_values (head: 0)"/>
  <img src="https://github.com/rstebbing/workshop/blob/main/experiments/transformer_sequence_classification/figures/generalization_errors/02-12.png" width=80% alt="generalization_errors token_attention_values (head: 1)"/>
</p>

The issue is in the first attention head (`(head: 0)`): it attends mostly to `"a"` and the token has a positive value of `2.0178`. But `"c"` *also* has a positive value (of `1.3575`) so that if there are a sufficiently large number of them:
- the `"CLS"` attention value for the sequence is more positive (i.e. further from `"b"`):
  <p align="center">
    <img src="https://github.com/rstebbing/workshop/blob/main/experiments/transformer_sequence_classification/figures/generalization_errors/02-13.png" width=80% alt="generalization_errors cls_attention_values (head: 0)"/>
  </p>
- the `"CLS"` attention output is closer to `"ab"`:
  <p align="center">
    <img src="https://github.com/rstebbing/workshop/blob/main/experiments/transformer_sequence_classification/figures/generalization_errors/02-15.png" width=80% alt="generalization_errors cls_attention_output"/>
  </p>
- and the transformed `"CLS"` embedding eventually projects positively onto the weight vector of the binary linear classifier.

The potential for false positives can be concealed by large differences in attention scores between the target token (e.g. `"a"`) and other tokens (e.g. `"c"`) for a given attention head. In this case, even if the attention values have the potential for false positives, if the target token (`"a"`) is present, it will take an impractically large number of other tokens (`"c"`) to move the `"CLS"` attention value.

## False Negatives

The string `"ab{100}c{55}"` is a false negative: the model outputs a negative logit when it should be positive. How is a false negative possible?

False negatives arise when too much attention is given to one or more undesired tokens. In this model the first attention head attends mostly to `"a"`s and then `"b"`s:

<p align="center">
  <img src="https://github.com/rstebbing/workshop/blob/main/experiments/transformer_sequence_classification/figures/generalization_errors/02-9.png" width=80% alt="generalization_errors token_attention_keys_and_queries (head: 0)"/>
</p>

If a sequence contains a large number of `"b"`s, then the `"CLS"` attention value for the sequence for the first head is more *negative*, shown by `"ab{100}c{55}"` being closer to `"b"`. In this case, the `"c"`s actually have no impact which can be verified by evaluating just `"ab{100}"`:
``` bash
python -m workshop.experiments.transformer_sequence_classification.main \
  eval \
  --directory data/experiments/transformer_sequence_classification/generalization_errors/01__layer_norm_eps=0.0/model_seed=2 \
  --max-num-incorrect-strings 10 \
  'ab{100}'
```
``` text
...
    list(zip(args.strings, output_probabilities)): [
        (
            'ab{100}',
            0.12186885625123978,
        ),
    ] (list) len=1
```

A model is suceptible to false negatives if the difference in attention scores between a target token (e.g. `"a"`) and other tokens (e.g. `"b"`) is too small so that large numbers of the latter can overwhelm the former.
