#!/usr/bin/env bash
set -exu

for model in mrm8488/bloom-560m-finetuned-news-summarization-xsum \
              VictorSanh/bart-base-finetuned-xsum \
              sshleifer/distill-pegasus-xsum-16-8 \
              facebook/bart-large-xsum \
              google/pegasus-xsum \
              sshleifer/distilbart-xsum-12-6
do
  python src/evaluate_mulChoice.py \
            -f  multiple_choice-dataset/xsum/factcc/* \
                multiple_choice-dataset/xsum/mfma/* \
                multiple_choice-dataset/xsum/gold_distractors/* \
                multiple_choice-dataset/xsum/random_distractors/* \
                multiple_choice-dataset/xsum/factual_distractors_from_other_models/* \
                multiple_choice-dataset/xsum/distractors_from_other_models/* \
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
          -m sysresearch101/t5-large-finetuned-xsum \
          -b 1 \
          -p 0