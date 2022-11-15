#!/bin/bash

wget "https://huggingface.co/bigscience/bloom/raw/main/config.json"
wget "https://huggingface.co/bigscience/bloom/raw/main/pytorch_model.bin.index.json"
wget "https://huggingface.co/bigscience/bloom/raw/main/special_tokens_map.json"
wget "https://huggingface.co/bigscience/bloom/resolve/main/tokenizer.json"
wget "https://huggingface.co/bigscience/bloom/raw/main/tokenizer_config.json"


for i in {1..9}
do
   echo "$i"
   wget "https://huggingface.co/bigscience/bloom/resolve/main/pytorch_model_0000$i-of-00072.bin"
done

for i in {10..72}
do
   echo "$i"
   wget "https://huggingface.co/bigscience/bloom/resolve/main/pytorch_model_000$i-of-00072.bin"
done