#!/bin/bash

# Script for downloading corpus files as well as creating delexicalized versions of the train and dev sets.
# Note that we are NOT DISTRIBUTING these corpus files with our code due to licensing reasons.

# English EWT
# train
curl https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master/en_ewt-ud-train.conllu -o en_ewt/en_ewt-ud-train.conllu
python delexicalize_corpus.py en_ewt/en_ewt-ud-train.conllu > en_ewt/en_ewt-ud-train.delex.conllu

# dev
curl https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master/en_ewt-ud-dev.conllu -o en_ewt/en_ewt-ud-dev.conllu
python delexicalize_corpus.py en_ewt/en_ewt-ud-dev.conllu > en_ewt/en_ewt-ud-dev.delex.conllu


# test
curl https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master/en_ewt-ud-test.conllu -o en_ewt/en_ewt-ud-test.conllu


# Latvian LVTB
# train
curl https://raw.githubusercontent.com/UniversalDependencies/UD_Latvian-LVTB/master/lv_lvtb-ud-train.conllu -o lv_lvtb/lv_lvtb-ud-train.conllu
python delexicalize_corpus.py lv_lvtb/lv_lvtb-ud-train.conllu > lv_lvtb/lv_lvtb-ud-train.delex.conllu

# dev
curl https://raw.githubusercontent.com/UniversalDependencies/UD_Latvian-LVTB/master/lv_lvtb-ud-dev.conllu -o lv_lvtb/lv_lvtb-ud-dev.conllu
python delexicalize_corpus.py lv_lvtb/lv_lvtb-ud-dev.conllu > lv_lvtb/lv_lvtb-ud-dev.delex.conllu

# test
curl https://raw.githubusercontent.com/UniversalDependencies/UD_Latvian-LVTB/master/lv_lvtb-ud-test.conllu -o lv_lvtb/lv_lvtb-ud-test.conllu

