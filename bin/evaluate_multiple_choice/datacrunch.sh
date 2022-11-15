#!/usr/bin/env bash

set -exu


for model in  /fruitbasket/models/facebook/opt-175b \
              /fruitbasket/models/bigscience/bloom

do
  python src/evaluate_multiple_choice.py \
            -f  multiple_choice-dataset/xsum/factcc/* \
                multiple_choice-dataset/xsum/mfma/* \
                multiple_choice-dataset/xsum/gold_distractors/* \
                multiple_choice-dataset/xsum/random_distractors/* \
                multiple_choice-dataset/xsum/factual_distractors_from_other_models/* \
                multiple_choice-dataset/xsum/distractors_from_other_models/* \
                multiple_choice-dataset/cnn_dm/factcc/* \
                multiple_choice-dataset/cnn_dm/mfma/* \
                multiple_choice-dataset/cnn_dm/gold_distractors/* \
                multiple_choice-dataset/cnn_dm/random_distractors/* \
                multiple_choice-dataset/cnn_dm/factual_distractors_from_other_models/* \
                multiple_choice-dataset/cnn_dm/distractors_from_other_models/* \
            -m $model \
            -b 1 \
            -p -1 \
            --use_bitsandbytes  \
            --compute_choices_iteratively
done
