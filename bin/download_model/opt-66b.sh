#!/bin/bash

wget "https://huggingface.co/facebook/opt-66b/raw/main/config.json"
wget "https://huggingface.co/facebook/opt-66b/raw/main/merges.txt"
wget "https://huggingface.co/facebook/opt-66b/raw/main/pytorch_model.bin.index.json"
wget "https://huggingface.co/facebook/opt-66b/raw/main/special_tokens_map.json"
wget "https://huggingface.co/facebook/opt-66b/raw/main/tokenizer_config.json"
wget "https://huggingface.co/facebook/opt-66b/raw/main/vocab.json"


for i in {1..9}
do
   echo "$i"
   wget "https://huggingface.co/facebook/opt-66b/resolve/main/pytorch_model-0000$i-of-00014.bin"
done

for i in {10..14}
do
   echo "$i"
   wget "https://huggingface.co/facebook/opt-66b/resolve/main/pytorch_model-000$i-of-00014.bin"
done