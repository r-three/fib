#!/usr/bin/env bash
set -exu

for model in /fruitbasket/models/facebook/opt-13b \
             /fruitbasket/models/EleutherAI/gpt-neox-20b
do
  python src/evaluate_mulChoice.py \
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
            --use_hugFace_parallelism \
            --compute_choices_iteratively
done

for model in /fruitbasket/models/bigscience/T0 \
             /fruitbasket/models/google/t5-xxl-lm-adapt
do
  python src/evaluate_mulChoice.py \
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
            --use_hugFace_parallelism
done

python src/evaluate_mulChoice.py \
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
          -m /fruitbasket/models/google/flan-t5-xxl \
          -b 1 \
          -p 0 \
          --use_hugFace_parallelism

python src/evaluate_mulChoice.py \
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
          -m /fruitbasket/models/facebook/opt-30b \
          -b 1 \
          -p -1 \
          --use_bitsandbytes  \
          --compute_choices_iteratively
