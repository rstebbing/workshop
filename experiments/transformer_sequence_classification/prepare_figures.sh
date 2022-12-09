#!/bin/bash

set -euxo pipefail


function copy_files {
  mkdir -p "${output_dir}"

  for f in "${files[@]}"; do
      cp "${input_dir}/${f}" "${output_dir}/${f}"
  done
}


cd "${SRC_ROOT}"


input_dir=data/experiments/transformer_sequence_classification/walkthrough/01__no_attend_cls=True__layer_norm_eps=0.0/model_seed=5/figures/min_validation_loss_epoch
output_dir=experiments/transformer_sequence_classification/figures/walkthrough
files=(
  8-1.png
  8-2.png
  8-3.png
  8-17.png
  8-18.png
  8-15.png
  8-5.png
  8-6.png
  8-7.png
  8-8.png
  8-9.png
  8-10.png
  8-11.png
  8-12.png
  8-14.png
  8-16.png
)
copy_files


input_dir=data/experiments/transformer_sequence_classification/generalization_errors/01__no_attend_cls=True__layer_norm_eps=0.0/model_seed=2/figures/min_validation_loss_epoch
output_dir=experiments/transformer_sequence_classification/figures/generalization_errors
files=(
  02-19.png
  02-9.png
  02-10.png
  02-11.png
  02-12.png
  02-13.png
  02-15.png
)
copy_files


input_dir=data/experiments/transformer_sequence_classification/attend_cls/01__layer_norm_eps=0.0/model_seed=8/figures/min_validation_loss_epoch
output_dir=experiments/transformer_sequence_classification/figures/attend_cls
files=(
  4-5.png
  4-6.png
)
copy_files


input_dir=data/experiments/transformer_sequence_classification/increased_dimensionality/01__hidden_size=16__no_attend_cls=True__layer_norm_eps=0.0/model_seed=0/figures/min_validation_loss_epoch
output_dir=experiments/transformer_sequence_classification/figures/increased_dimensionality
files=(
  5-1.png
  5-18.png
)
copy_files


input_dir=data/experiments/transformer_sequence_classification/lower_magnitude_initializations/01__hidden_size=16__no_attend_cls=True__layer_norm_eps=0.0__embeddings_init_strategy=linear_like__attention_init_strategy=output_fan_out__feed_forward_init_strategy=output_fan_out/model_seed=2/figures/min_validation_loss_epoch
output_dir=experiments/transformer_sequence_classification/figures/lower_magnitude_initializations
files=(
  17-9.png
  17-10.png
)
copy_files
