#!/bin/bash

# Script for downloading pretrained word embeddings.
# Note that we are NOT DISTRIBUTING these models with our code due to licensing reasons.
# Skip downloading a model by commenting out the corresponding commands.


# mBERT
mkdir bert-base-multilingual-cased
curl https://huggingface.co/bert-base-multilingual-cased/resolve/main/config.json -o bert-base-multilingual-cased/config.json --create-dirs
curl https://huggingface.co/bert-base-multilingual-cased/resolve/main/vocab.txt -o bert-base-multilingual-cased/vocab.txt
curl https://cdn-lfs.huggingface.co/bert-base-multilingual-cased/3496a508a9a3511c8a55e4d0e6f471c70c68c2a8c4784b3b2b5dc16ffb87d238 -o bert-base-multilingual-cased/pytorch_model.bin

# XLM-R (large)
mkdir xlm-roberta-large
curl https://huggingface.co/xlm-roberta-large/resolve/main/config.json -o xlm-roberta-large/config.json
curl https://huggingface.co/xlm-roberta-large/resolve/main/sentencepiece.bpe.model -o xlm-roberta-large/sentencepiece.bpe.model
curl https://cdn-lfs.huggingface.co/xlm-roberta-large/01e55aa45dbb9164fee19aef60007a1c91d175051c01be1fb15056cfa60f3e53 -o xlm-roberta-large/pytorch_model.bin
