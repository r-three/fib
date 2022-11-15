#!/bin/bash

wget "https://huggingface.co/facebook/opt-30b/raw/main/config.json"
wget "https://huggingface.co/facebook/opt-30b/raw/main/merges.txt"
wget "https://huggingface.co/facebook/opt-30b/raw/main/pytorch_model.bin.index.json"
wget "https://huggingface.co/facebook/opt-30b/raw/main/special_tokens_map.json"
wget "https://huggingface.co/facebook/opt-30b/raw/main/tokenizer_config.json"
wget "https://huggingface.co/facebook/opt-30b/raw/main/vocab.json"


for i in {1..7}
do
   echo "$i"
   wget "https://huggingface.co/facebook/opt-30b/resolve/main/pytorch_model-0000$i-of-00007.bin"
done
