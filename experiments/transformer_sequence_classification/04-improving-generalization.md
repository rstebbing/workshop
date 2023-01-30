# Improving Generalization

Previous: [Generalization Errors ⧉](03-generalization-errors.md), Up: [Transformer Sequence Classification ⧉](.)

Table of Contents
1. [Introduction](#introduction)
2. [Attention](#attention)
3. [Dataset](#dataset)
4. [Initialization and Increased Dimensionality](#initialization-and-increased-dimensionality)
   1. [Increase `hidden_size`](#increase-hidden_size)
   2. [Increase `num_attention_heads`](#increase-num_attention_heads)
5. [Pruning](#pruning)

## Introduction

For this problem, false positives and false negatives are reduced if, for each attention head, the query for the `"CLS"` token projects *very* positively with the key for *one* token (`"a"` or `"b"`) and *very* negatively with the keys for the others.

How has this been achieved in the best models examined so far? How can this be improved too?

## Attention

The models examined so far have all been trained *without* attending to the `"CLS"` token: the attention value for every attention head is the weighted sum of attention values for *just* the tokens from the input string *without* any bias (which arises from including the `"CLS"` token). Eliminating the potential for the bias prevents models being fitted where one or more attention heads learn to attend to the *absence* of one or more tokens. This occurs rarely, but preventing it entirely simplifies the exposition. This *is* a bit unconventional and likely not something that should be done with multi-layer models in general.

An example model that achieves good test accuracy can be seen running:
``` bash
python -m workshop.experiments.transformer_sequence_classification run attend_cls
```

The query for the `"CLS"` token and the keys for all tokens *including* `"CLS"` too are shown below:

<p align="center">
  <img src="https://github.com/rstebbing/workshop/blob/main/experiments/transformer_sequence_classification/figures/attend_cls/4-5.png" width=80% alt="attend_cls token_attention_keys_and_queries (head: 0)"/>
  <img src="https://github.com/rstebbing/workshop/blob/main/experiments/transformer_sequence_classification/figures/attend_cls/4-6.png" width=80% alt="attend_cls token_attention_keys_and_queries (head: 1)"/>
</p>

## Dataset

The models examined so far have all been trained using instances of `VariableLengthSequences`, a class that synthetically generates random batches of labeled strings. The properties of this can be most easily understood by example:
``` python
from workshop.experiments.transformer_sequence_classification.data import DataConfig, VariableLengthSequences
from workshop.experiments.transformer_sequence_classification.tokenizer import Tokenizer, TokenizerConfig


tokenizer_config = TokenizerConfig()
tokenizer = Tokenizer(config=tokenizer_config)

data_config = DataConfig()
data_loader = VariableLengthSequences(
    tokenizer_config=tokenizer_config,
    data_config=data_config,
    max_sequence_length=10,
    alpha=1.0,
    batch_size=4,
    num_batches_per_epoch=2,
    seed=0,
)

for epoch in range(2):
    for batch_index, batch in enumerate(data_loader):
        strings = ["".join(tokenizer.decode(token_ids)) for token_ids in batch.token_ids]  # pyright: ignore
        print(f"Epoch {epoch}, batch {batch_index}:")
        for string, label in zip(strings, batch.labels):
            print(f"  {string!r} ({label.item()})")
```
``` text
Epoch 0, batch 0:
  'cccccc' (0)
  'aaaa' (0)
  'aabcc' (1)
  'bbbbbbbb' (0)
Epoch 0, batch 1:
  'bbb' (0)
  'aabbcccc' (1)
  'ccccccc' (0)
  'aaac' (0)
Epoch 1, batch 0:
  'aacccc' (0)
  'ccccc' (0)
  'abccccc' (1)
  'bbbccccccc' (0)
Epoch 1, batch 1:
  'cccccccccc' (0)
  'abbcc' (1)
  'bccc' (0)
  'accc' (0)
```

Each batch contains `batch_size == 4` items with strings up to `max_sequence_length == 10` in length. Each batch is guaranteed to contain:
- a positive example (a string with at least one `"a"` and `"b"`) and
- negative examples that include *either* `"a"` or  `"b"` (but not both) and *neither* (i.e. only `"c"`s).
`alpha == 1.0` is the concentration parameter for a Dirichlet distribution. Values less than `1.0` encourage more *imbalanced* strings like `"accccccccc"` over `"aaaaaccccc"`.

`VariableLengthSequences`, however, was *not* the first dataset attempted for this problem. A simpler dataset comprising of all possible strings for a given length was tried first. This dataset is implemented by the `AllFixedLengthSequences` and can be examined by example:
``` python
from collections import Counter

from workshop.experiments.transformer_sequence_classification.data import AllFixedLengthSequences, DataConfig
from workshop.experiments.transformer_sequence_classification.tokenizer import Tokenizer, TokenizerConfig


tokenizer_config = TokenizerConfig()
data_config = DataConfig()

tokenizer = Tokenizer(config=tokenizer_config)
dataset = AllFixedLengthSequences(
    tokenizer_config=tokenizer_config,
    data_config=data_config,
    sequence_length=10,
)

print(f"{len(dataset) = }")

num_items = 10
print(f"Items [0, {num_items}):")
for i in range(num_items):
    item = dataset[i]
    string = "".join(tokenizer.decode(item.token_ids))
    print(f"  {string!r} ({item.label})")

print("Label counts:")
label_counts = Counter(dataset[i].label for i in range(len(dataset)))
for label, count in sorted(label_counts.items()):
    print(f"  {label}: {count}")
```
``` text
len(dataset) = 19683
Items [0, 10):
  'aaaaaaaaa' (0)
  'aaaaaaaab' (1)
  'aaaaaaaac' (0)
  'aaaaaaaba' (1)
  'aaaaaaabb' (1)
  'aaaaaaabc' (1)
  'aaaaaaaca' (0)
  'aaaaaaacb' (1)
  'aaaaaaacc' (0)
  'aaaaaabaa' (1)
Label counts:
  0: 1023
  1: 18660
```

Notably: `AllFixedLengthSequences` has a much lower proportion of negative examples than `VariableLengthSequences`—5.2% (`1023 / (1023 + 18660)`) versus 75%. This difference matters a lot, and the models trained with `AllFixedLengthSequences` generalize worse to longer sequences (exhibiting a larger number of false negatives). This can be seen from the confusion matrices below:
``` bash
python -m workshop.experiments.transformer_sequence_classification run all_fixed_length_sequences
```
``` text
...
    test_confusion_matrices: [
        (
            'data/experiments/transformer_sequence_classification/all_fixed_length_sequences/01__layer_norm_eps=0.0__train_dataset=all_fixed_length_sequences',
            [
                (
                    'model_seed=0',
                    [
                        [4329, 0],
                        [
                            340,
                            5315,
                        ],
                    ],
                ),
                (
                    'model_seed=1',
                    [
                        [4329, 0],
                        [
                            322,
                            5333,
                        ],
                    ],
                ),
            ],
        ),
    ] (list) len=1
```

To summarize: negative examples are important so that the model can learn what *not* to pay attention to!

## Initialization and Increased Dimensionality

The only difference between the model examined in [High-Level Walkthrough ⧉](01-high-level-walkthrough.md)—which generalizes well—and the model examined in [Generalization Errors ⧉](03-generalization-errors.md)—which does *not* generalize well—is that they were initialized with different random seeds (i.e. different values for `model_seed`). Running:
``` bash
python -m workshop.experiments.transformer_sequence_classification run all_initializations
```
it can be seen that only *one* initialiation results in zero false positives and zero false negatives. The rest have high numbers of false negatives (in particular):
``` text
...
    test_confusion_matrices: [
        (
            'data/experiments/transformer_sequence_classification/all_initializations/01__layer_norm_eps=0.0',
            [
                (
                    'model_seed=0',
                    [
                        [4329, 0],
                        [
                            1378,
                            4277,
                        ],
                    ],
                ),
                ...
                (
                    'model_seed=7',
                    [
                        [4329, 0],
                        [
                            214,
                            5441,
                        ],
                    ],
                ),
            ],
        ),
    ] (list) len=1

```

### Increase `hidden_size`

Re-running training with different seeds *is* a strategy in of itself, but another approach is to change the problem so that the optimization algorithm (the AdamW algorithm) has a higher chance of achieving a low validation loss irrespective of the initialization. A starting point is to increase the model dimensionality `hidden_size` from `2` to `16`. Running:
``` bash
python -m workshop.experiments.transformer_sequence_classification run increased_dimensionality
```
it can be seen that all initializations result in zero false positives and zero false negatives:
``` text
...
   test_confusion_matrices: [
        (
            'data/experiments/transformer_sequence_classification/increased_dimensionality/01__hidden_size=16__num_attention_heads=16__layer_norm_eps=0.0',
            [
                (
                    'model_seed=0',
                    [
                        [4329, 0],
                        [0, 5655],
                    ],
                ),
                ...
                (
                    'model_seed=7',
                    [
                        [4329, 0],
                        [0, 5655],
                    ],
                ),
            ],
        ),
    ] (list) len=1
```

Ignoring `"PAD"` there are only four tokens (`"a"`, `"b"`, `"c"` and `"CLS"`) so that the embeddings can be visualized in three dimensions without loss of information too:

   <p align="center">
     <img src="https://github.com/rstebbing/workshop/blob/main/experiments/transformer_sequence_classification/figures/increased_dimensionality/5-1.png" width=80% alt="increased_dimensionality embeddings"/>
   </p>

Increasing `hidden_size` results in improved accuracy for at least a couple of reasons:

1. For high model accuracy, the embeddings should have discriminative 1D projections (e.g. that separate `"a"` from `"b"` and `"c"`, or separate `"b"` from `"a"` and `"c"`) and by increasing the dimensionality of the embeddings (a) the are more variables to manipulate to make this possible and (b) there is a higher chance of close enough projections being possible in the first place. This is demonstrated by the low validation errors achieved after just a 1-2 epochs. E.g. for `model_seed == 0` the validation loss after two epochs is `0.00686`.

   > The paper [The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks ⧉](https://arxiv.org/abs/1803.03635) examines thoroughly the connection between neural network size and chance of high-quality *sub-network* initializations, *and* that the sub-networks can be trained to full accuracy in isolation too. Other relevants papers include [Linear Mode Connectivity and the Lottery Ticket Hypothesis ⧉](https://arxiv.org/abs/1912.05671) and [Lottery Tickets on a Data Diet: Finding Initializations with Sparse Trainable Networks ⧉](https://arxiv.org/abs/2206.01278).

2. The implemented `Model` relies on the default initialization strategy for a PyTorch `Embedding` layer, which has implications as `Linear` layers. The default initialization strategy is to sample each weight from the standard normal distribution *irrespective* of `hidden_size`. The square magnitude of each initialized embeddings follows a chi-squared distribution with `hidden_size` degrees of freedom (and a mean of `hidden_size` as a result). Therefore: models with higher `hidden_size` have initial embeddings with lager magnitudes, making them less susceptible to false negatives because the 1D projections that constitute the attention scores have larger magnitudes too. This can be seen by directly comparing the model weights of the very first model in [High-Level Walkthrough ⧉](01-high-level-walkthrough.md) against the model weights from `model_seed == 0` above too:

   <p align="center">
     <img src="https://github.com/rstebbing/workshop/blob/main/experiments/transformer_sequence_classification/figures/walkthrough/8-16.png" width=80% alt="walkthrough parameter_magnitudes"/>
     <img src="https://github.com/rstebbing/workshop/blob/main/experiments/transformer_sequence_classification/figures/increased_dimensionality/5-18.png" width=80% alt="increased_dimensionality parameter_magnitudes"/>
   </p>

   The `output` layers in the `SelfAttentionBlock` and `FeedForward` blocks are both `Linear` layers, and the default initialization strategy has implications as `hidden_size` increases in isolation too. The default strategy is to sample each weight uniformly between `-1/sqrt(in_features)` and `1/sqrt(in_features)`, where `in_features` is `config.attention_size == 2` or `config.feed_forward_size == 2`. Like the embeddings, this is constant in `hidden_size`. The end result is larger magnitude outputs from the `SelfAttentionBlock` and more positive or more negative logits overall.

The impact of *just* the large magnitude embedding initializations can be demonstrated by first removing all biases toward larger magnitude attention scores and outputs by initializing all three aforementioned layers differently:
``` bash
python -m workshop.experiments.transformer_sequence_classification run lower_magnitude_initializations
```

``` text
...
   test_confusion_matrices: [
        (
            'data/experiments/transformer_sequence_classification/lower_magnitude_initializations/01__hidden_size=16__layer_norm_eps=0.0__embeddings_init_strategy=linear_like__attention_init_strategy=output_fan_out__feed_forward_init_strategy=output_fan_out',
            [
                (
                    'model_seed=0',
                    [
                        [4329, 0],
                        [0, 5655],
                    ],
                ),
                (
                    'model_seed=1',
                    [
                        [4329, 0],
                        [
                            273,
                            5382,
                        ],
                    ],
                ),
                (
                    'model_seed=2',
                    [
                        [
                            2886,
                            1443,
                        ],
                        [0, 5655],
                    ],
                ),
                ...
                (
                    'model_seed=6',
                    [
                        [4329, 0],
                        [
                            328,
                            5327,
                        ],
                    ],
                ),
                (
                    'model_seed=7',
                    [
                        [4329, 0],
                        [
                            240,
                            5415,
                        ],
                    ],
                ),
            ],
        ),
    ] (list) len=1
```

Second, initializing *just* the `Embeddings` with the default standard normalizations initialization shows the false negatives eliminated (again):
``` bash
python -m workshop.experiments.transformer_sequence_classification run default_embeddings_initialization
```

``` text
...
    test_confusion_matrices: [
        (
            'data/experiments/transformer_sequence_classification/default_embeddings_initialization/01__hidden_size=16__layer_norm_eps=0.0__attention_init_strategy=output_fan_out__feed_forward_init_strategy=output_fan_out',
            [
                (
                    'model_seed=0',
                    [
                        [4329, 0],
                        [0, 5655],
                    ],
                ),
                ...
                (
                    'model_seed=7',
                    [
                        [4329, 0],
                        [0, 5655],
                    ],
                ),
            ],
        ),
    ] (list) len=1

```

### Increase `num_attention_heads`

The `lower_magnitude_initializations` experiment has a few initializations (e.g. `model_seed == 2`) that have a high number of false positives. Inspection of the attention queries and keys shows that this is because the two attention heads are attending to the same token (`"b"`):

<p align="center">
  <img src="https://github.com/rstebbing/workshop/blob/main/experiments/transformer_sequence_classification/figures/lower_magnitude_initializations/17-9.png" width=80% alt="lower_magnitude_initializations token_attention_keys_and_queries (head: 0)"/>
  <img src="https://github.com/rstebbing/workshop/blob/main/experiments/transformer_sequence_classification/figures/lower_magnitude_initializations/17-10.png" width=80% alt="lower_magnitude_initializations token_attention_keys_and_queries (head: 1)"/>
</p>

To have a higher chance of discriminative attention heads, the `num_attention_heads` can be increased too (e.g. from `2` to `16`). Running:
``` bash
python -m workshop.experiments.transformer_sequence_classification run increased_num_attention_heads
```
shows the removal of false positives for any initialization:
``` text
...
    test_confusion_matrices: [
        (
            'data/experiments/transformer_sequence_classification/increased_num_attention_heads/01__hidden_size=16__num_attention_heads=16__layer_norm_eps=0.0__embeddings_init_strategy=linear_like__attention_init_strategy=output_fan_out__feed_forward_init_strategy=output_fan_out',
            [
                (
                    'model_seed=0',
                    [
                        [4329, 0],
                        [0, 5655],
                    ],
                ),
                ...
                (
                    'model_seed=7',
                    [
                        [4329, 0],
                        [0, 5655],
                    ],
                ),
            ],
        ),
    ] (list) len=1
```

False negatives are *also* reduced. This is due to a different implicit bias that occurs due to how the multiple attention heads combine in the `SelfAttentionBlock`.

To summarize: increasing `hidden_size` and `num_attention_heads` results in more initializations converging to low validation error in fewer epochs. Generalization error is improved too, but reduction in false negatives is due (in part) to the implicit biases arising from default initialization strategies for some of the layers (e.g. `Embeddings`). The impact of this bias might *not* always be positive!

## Pruning

Increasing the number of parameters can help training converge in fewer epochs to a more-general model ([Initialization and Dimensionality](#initialization-and-dimensionality)) but the down-side is a larger model that is more expensive to evaluate. Assuming this extra model capacity is *not* wanted, what can be done about this?

*Pruning* is the procedure of reducing the number of parameters in a neural network, during and/or after training. The high-level idea is to remove redundant parts of the network, like redundant dimensions in embeddings—as purposefully introduced *and* observed in in [Increase `hidden_size`](#increase-hidden_size)—or redundant attention heads—as introduced in [Increase `num_attention_heads`](#increase-num_attention_heads). This experiment will be extended in future to demonstrate pruning techniques, including identifying and pruning redundant heads as explained in [Are Sixteen Heads Really Better than One? ⧉](https://arxiv.org/abs/1905.10650) and demonstrated on BERT in [run_bertology.py ⧉](https://github.com/huggingface/transformers/blob/main/examples/research_projects/bertology/run_bertology.py).

---

Previous: [Generalization Errors ⧉](03-generalization-errors.md), Up: [Transformer Sequence Classification ⧉](.)
