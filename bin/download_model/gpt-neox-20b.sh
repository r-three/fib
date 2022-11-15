#!/bin/bash

for i in {1..9}
do
   echo "$i"
   wget "https://huggingface.co/EleutherAI/gpt-neox-20b/resolve/main/pytorch_model-0000$i-of-00046.bin"
done

for i in {10..46}
do
   echo "$i"
   wget "https://huggingface.co/EleutherAI/gpt-neox-20b/resolve/main/pytorch_model-000$i-of-00046.bin"
done