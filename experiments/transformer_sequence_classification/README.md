# Transformer Sequence Classification

Table of Contents
1. [Introduction](#introduction)
2. [Problem Statement](#problem-statement)
3. [Report](#report)
   1. [High-Level Walkthrough ⧉](01-high-level-walkthrough.md)
   2. [`self_attention_block` Walkthrough ⧉](02-self-attention-block-walkthrough.md)
   3. [Generalization Errors ⧉](03-generalization-errors.md)
   4. [Improving Generalization ⧉](04-improving-generalization.md)
4. [Conclusions](#conclusions)

## Introduction

The *Transformer* is a neural network architecture that was proposed in the paper [Attention is All You Need ⧉](https://arxiv.org/abs/1706.03762) and has been massively successful in problems ranging from [image classification ⧉](https://arxiv.org/abs/2010.11929) to [machine translation ⧉](https://arxiv.org/abs/1910.10683) and [text generation ⧉](https://arxiv.org/abs/2005.14165).

There are many variations of this architecture, but common among them is the use of a large number of *Transformer Blocks* applied one after another.

A single Transformer Block is a neural network comprising of:
- a *Multi-Headed Attention Block* and
- a *Feed Forward Block*

both of which operate on (layer) normalized inputs and are combined with residual connections.

Large Transformer models are incredibly exciting, but the high model dimensionality, large number of layers, and complexity of preprocessing (e.g. tokenization algorithms) and postprocessing make them non-ideal for gaining simple intuitions about the internals of the Transformer Block.

The goal of the set of experiments in this repository is to help build intuitions about the internals of the Transformer Block by probing a simple single-layer sequence classifier. The report is not intended to be an introduction to attention etc., but *is* intended to be a complement to introductory guides like Jay Alammar's excellent [The Illustrated Transformer ⧉](https://jalammar.github.io/illustrated-transformer/) and other expositions with toy problems like Anthropic's [Toy Models of Superposition ⧉](https://transformer-circuits.pub/2022/toy_model/index.html).

## Problem Statement

Build a simple sequence classifier with a single layer to solve the following toy problem:

> Given a string (e.g. `"aac"` or `"baac"`) output `1` if the string contains at least one `"a"` *and* at least one `"b"`, or `0` otherwise.

This problem is useful because it is amenable to a small number of attention heads (e.g. `2`) and a small number of model dimensions (`2`), making it straightforward to visualize all internal states without additional projections.

## Report

This report is split across multiple documents which are best read consecutively. The documents are:

1. [High-Level Walkthrough ⧉](01-high-level-walkthrough.md) details what occurs when two example strings `"aac"` and `"baac"` are processed by the sequence classifier. This document details how the input strings are tokenized, how the tokens are mapped to embeddings, how the embeddings are transformed by the Transformer Block, and how the transformed embeddings are processed to generate output logits that indicate whether or not the sequence contains `"a"` *and* `"b"`.
2. [`self_attention_block` Walkthrough ⧉](02-self-attention-block-walkthrough.md) details *how* the embeddings are transformed within the instance of the [`SelfAttentionBlock` ⧉](https://github.com/rstebbing/workshop/blob/5b6dff53d8adc4c83707f8da80fc2c2b08f08c76/py/src/workshop/experiments/transformer_sequence_classification/model.py#L230). Specific attention (pun intended) is given to how the attention keys, queries, and values are constructed for the example strings.
3. [Generalization Errors ⧉](03-generalization-errors.md) examines example errors—both false positives and false negatives—and details why they occur within the Transformer Block.
4. [Improving Generalization ⧉](04-improving-generalization.md) examines strategies for improving generalization of the model. This document focuses primarily on dataset construction, initialization strategies, and model dimensionality.

## Reproducibility

Commands to reproduce the included figures are included inline within each document. All experiments can be run via the single command:
``` bash
python -m workshop.experiments.transformer_sequence_classification all --no-visualize > /dev/null
```
This takes approximately 10 minutes on an M1 Max. All figures can be generated too by omitting `--no-visualize`.

The model training, evaluation, and visualization steps are all cached to disk too, so that it re-running the above command a second time is fast. Adding a new experiment can be done by appending a new `RunSpec` entry to `RUN_SPECS` in [main.py ⧉](https://github.com/rstebbing/workshop/blob/5b6dff53d8adc4c83707f8da80fc2c2b08f08c76/py/src/workshop/experiments/transformer_sequence_classification/main.py#L56).

## Conclusions

To summarize this experiment and some potentially useful observations and/or reminders:

1. A simple Transformer-based sequence classifier works by (a) transforming the embedding for an introduced `"CLS"` token based on the tokens in the sequence and (b) projecting the transformed embedding to produce the output classification logit ([High-Level Walkthrough ⧉](01-high-level-walkthrough.md)).
2. The kernel of the model is the Multi-Headed Attention Block: for each attention head, the *keys* for each token are projected onto the `"CLS"` *query* to determine their respective scores and attention weights. The output attention value for `"CLS"` is the weighted sum of attention values from each token, and a weighted sum across all heads determines the transformed embedding ([`self_attention_block` Walkthrough: `attention` ⧉](02-self-attention-block-walkthrough.md#attention)).
3. The absolute magnitude *and* relative differences of the attention scores within each attention head dictate generalizability to larger and/or more imbalanced sequences, but it is the attention values that govern whether or not false positives are possible ([Generalization Errors: False Positives ⧉](03-generalization-errors.md#false-positives)).
4. Balanced and diverse datasets and batches—compared to simple and exhaustive datasets and batches—lead to models that generalize better to longer sequences, even when the length of the training sequences are identical ([Generalization Errors: Dataset ⧉](04-improving-generalization.md#dataset)).
5. Biases in model initialization strategies should be well-understood so that model dimensionality changes do *not* unexpectedly change model sensitivity by biasing toward larger (or smaller) vectors. For example: the default initialization for PyTorch `Embeddings` is the standard normal distribution which results in the average magnitude of an embedding increasing by a factor of `√2` for every doubling of the number of dimensions ([Generalization Errors: Initialization and Increased Dimensionality ⧉](04-improving-generalization.md#initialization-and-increased-dimensionality)).
