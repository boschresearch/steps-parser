#  Copyright (c) 2020 Robert Bosch GmbH
#  All rights reserved.
#
#  This source code is licensed under the AGPL v3 license found in the
#  LICENSE file in the root directory of this source tree.
#
#  Author: Stefan Grünewald

from torch.utils.data import Dataset

from math import inf

from data_handling.dependency_matrix import DependencyMatrix
from data_handling.tag_sequence import TagSequence
from data_handling.vocab import BasicVocab
from data_handling.annotated_sentence import AnnotatedSentence


class CustomCoNLLDataset(Dataset):
    """An object of this class represents a (map-style) dataset of annotated sentences in a CoNLL-like format.
    The individual objects contained within the dataset are of type AnnotatedSentence.
    """
    def __init__(self):
        self.sentences = list()

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, item):
        return self.sentences[item]

    def append_sentence(self, sent):
        """Append one sentence to the dataset.

        Args:
            sent: AnnotatedSentence object to add to the dataset.
        """
        self.sentences.append(sent)

    @staticmethod
    def from_corpus_file(corpus_filename, annotation_layers, max_sent_len=inf, keep_traces=False):
        """Read in a dataset from a corpus file in CoNLL format.

        Args:
            corpus_filename: Path to the corpus file to read from.
            annotation_layers: Dictionary mapping annotation IDs to annotation type and CoNLL column to read data from.
            max_sent_len: The maximum length of any given sentence. Sentences with a greater length are ignored.
            keep_traces: Whether to keep empty nodes as tokens (used in enhanced UD; default: False).

        Returns:
            A CustomCoNLLDataset object containing the sentences in the input corpus file, with the specified annotation
            layers.
        """
        dataset = CustomCoNLLDataset()

        for raw_conll_sent in _iter_conll_sentences(corpus_filename):
            processed_sent = AnnotatedSentence.from_conll(raw_conll_sent, annotation_layers, keep_traces=keep_traces)
            if len(processed_sent) <= max_sent_len:
                dataset.append_sentence(processed_sent)

        return dataset

    @staticmethod
    def extract_label_vocab(*conllu_datasets, annotation_id):
        """Extract a vocabulary of labels from one or more CONLL-U datasets.

        Args:
            *conllu_datasets: One or more CustomCoNLLDataset objects to extract the label.
            annotation_id: Identifier of the annotation layer to extract labels for.
        """
        vocab = BasicVocab()

        for dataset in conllu_datasets:
            for sentence in dataset:
                if isinstance(sentence[annotation_id], DependencyMatrix):
                    for label in [lbl for head_row in sentence[annotation_id].data for lbl in head_row]:
                        vocab.add(label)
                elif isinstance(sentence[annotation_id], TagSequence):
                    for label in sentence[annotation_id].data:
                        vocab.add(label)
                else:
                    raise Exception("Unknown annotation type")

        assert vocab.is_consistent()
        return vocab


def _iter_conll_sentences(conll_file):
    """Helper function to iterate over the CoNLL sentence data in the given file.
        Args:
            conll_file: The custom CoNLL file to parse.
        Yields:
            An iterator over the raw CoNLL lines for each sentence.
    """
    # CoNLL parsing code adapted from https://github.com/pyconll/pyconll/blob/master/pyconll/_parser.py
    opened_file = False
    if isinstance(conll_file, str):
        conll_file = open(conll_file, "r")
        opened_file = True

    sent_lines = []
    for line in conll_file:
        line = line.strip()

        if line:
            sent_lines.append(line)
        else:
            if sent_lines:
                yield sent_lines
                sent_lines = []

    if sent_lines:
        yield sent_lines

    if opened_file:
        conll_file.close()
