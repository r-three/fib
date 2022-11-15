#!/usr/bin/env bash
set -exu

for model in /fruitbasket/models/bigscience/T0_3B \
             /fruitbasket/models/google/t5-xl-lm-adapt \
             /fruitbasket/models/bigscience/bloom-1b1 \
             /fruitbasket/models/bigscience/bloom-1b7 \
             /fruitbasket/models/bigscience/bloom-3b \
             /fruitbasket/models/bigscience/bloom-7b1 \
             /fruitbasket/models/EleutherAI/gpt-neo-1.3B \
             /fruitbasket/models/EleutherAI/gpt-neo-2.7B \
             /fruitbasket/models/EleutherAI/gpt-j-6B \
             /fruitbasket/models/facebook/opt-1.3b \
             /fruitbasket/models/facebook/opt-2.7b \
             /fruitbasket/models/facebook/opt-6.7b \
             /fruitbasket/models/gpt2-xl
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
            -p -1
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
          -m /fruitbasket/models/google/flan-t5-xl \
          -b 1 \
          -p 0
