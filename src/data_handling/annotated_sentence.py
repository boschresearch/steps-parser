#  Copyright (c) 2020 Robert Bosch GmbH
#  All rights reserved.
#
#  This source code is licensed under the AGPL v3 license found in the
#  LICENSE file in the root directory of this source tree.
#
#  Author: Stefan Grünewald

import torch

from .tag_sequence import TagSequence
from .dependency_matrix import DependencyMatrix


class AnnotatedSentence:
    """An object of this class represents a sequence of tokens and one or more named annotation layers on top of these
       tokens, which can be accessed dictionary-style. Each of these layers is of one of two types:

         * `TagSequence`, for representing tags for each token (e.g. POS tags);
         * `DependencyMatrix`, for representing dependency edges between tokens (e.g. syntactic dependencies).

       Note that a "[root]" token is prepended to each sentence to ensure consistency, i.e., if the "raw" sentence
       contains n tokens, len(self.tokens) == n+1.

       Note on traces: While the class is able to read in sentences with traces (= tokens with IDs like "8.1"),
       it treats them completely identically to normal tokens and consequently forgets about their special ID.
    """

    def __init__(self, tokens, annotation_data, multiword_tokens=None):
        """
        Args:
            tokens: List of strings containing the tokens of the sentence.
            annotation_data: Dictionary (`annotation_id` -> `TagSequence` or `DependencyMatrix`) mapping named annotation
                layers to the actual annotation data.
            multiword_tokens: Optional; Dictionary mapping sentence indices to multiword tokens occurring at these
                indices, specified as tuples `(token_id, token_form)`. Multiword tokens are printed out at the correct
                positions when outputting the sentence in CoNLL format, but do not affect anything else.
        """
        self.tokens = tokens
        assert self.tokens[0] == "[root]"

        self.annotation_data = annotation_data
        for annotation_id in self.annotation_data:
            curr_annotation_data = self.annotation_data[annotation_id]
            if isinstance(curr_annotation_data, TagSequence):
                assert len(curr_annotation_data) == len(self.tokens)
            elif isinstance(curr_annotation_data, DependencyMatrix):
                assert len(curr_annotation_data) == len(self.tokens)
                assert all(len(row) == len(self.tokens) for row in curr_annotation_data)
            else:
                assert False

        if multiword_tokens is None:
            self.multiword_tokens = dict()
        else:
            self.multiword_tokens = multiword_tokens

    def __len__(self):
        """Return the number of tokens in this sentence, including `[root]`."""
        return len(self.tokens)

    def __str__(self):
        """Return the (raw) tokens of this sentence as a whitespace-tokenized string."""
        return "AnnotatedSentence(\"{}\")".format(" ".join(self.tokens[1:]))

    def __getitem__(self, item):
        """Return the annotation data for the given annotation ID."""
        return self.annotation_data[item]

    @staticmethod
    def from_conll(conll_lines, annotation_layers, keep_traces=False):
        """Create an AnnotatedSentence from a list of CoNLL lines. A `[root]` token will be added to the
        beginning of the sentence.

        Args:
            conll_lines: Iterable of lines (strings) representing the sentence, in CoNLL format.
            annotation_layers: Dictionary mapping annotation IDs to annotation type and CoNLL column to read data from.
            keep_traces: Optional; Whether to keep empty nodes as tokens (used in enhanced UD).

        Returns:
            An AnnotatedSentence object representing the sentence specified in the CoNLL data.
        """
        # Create token list and ID->Index dictionary
        tokens = ['[root]']
        id_to_ix = dict({'0': 0})
        ix = 1
        multiword_tokens = dict()

        # Extract relevant CoNLL lines as well as corresponding tokens
        filtered_conll_lines = list()
        for line in conll_lines:
            assert line  # Lines must not be empty
            if line.startswith("#"):
                continue  # Ignore metadata

            line = line.strip()
            elements = line.split('\t')

            token_id = elements[0]
            token_form = elements[1]

            if "-" in token_id:
                multiword_tokens[ix] = (token_id, token_form)
                continue
            if "." in token_id and not keep_traces:
                continue

            filtered_conll_lines.append(line)
            tokens.append(token_form)
            id_to_ix[token_id] = ix
            ix += 1

        # Create annotation layers
        annotation_data = dict()
        for annotation_id in annotation_layers:
            annotation_type = annotation_layers[annotation_id]["type"]
            annotation_column = annotation_layers[annotation_id]["source_column"]

            if "args" in annotation_layers[annotation_id]:
                annotation_kwargs = annotation_layers[annotation_id]["args"]
            else:
                annotation_kwargs = {}

            if annotation_type == "TagSequence":
                annotation_data[annotation_id] = TagSequence.from_conll(filtered_conll_lines, annotation_column, **annotation_kwargs)
            elif annotation_type == "DependencyMatrix":
                annotation_data[annotation_id] = DependencyMatrix.from_conll(filtered_conll_lines, annotation_column, id_to_ix, **annotation_kwargs)
            else:
                assert False

        return AnnotatedSentence(tokens, annotation_data, multiword_tokens)

    @staticmethod
    def from_tensors(tokens, label_tensors, label_vocabs, annotation_types, multiword_tokens=None):
        """Create an AnnotatedSentence from tensors containing label indices for the different annotation layers.

        Args:
            tokens: The tokens of the sentence (list of strings).
            label_tensors: Dictionary mapping output IDs to the label tensors.
            label_vocabs: Dictionary mapping output IDs to label vocabularies.
            annotation_types: Dictionary mapping output IDs to the type of annotation to be created.
            multiword_tokens: Optional; Dictionary mapping sentence indices to multiword tokens occurring at these
                indices, specified as tuples `(token_id, token_form)`.

        Returns:
            An AnnotatedSentence object representing the sequence of tokens and annotation data specified in the
            input tensors, converted to actual labels via the specified label vocabularies.
        """
        assert tokens[0] == "[root]"
        assert annotation_types.keys() == label_tensors.keys() == label_vocabs.keys()

        annotation_data = dict()
        for annotation_id in label_tensors:
            assert annotation_types[annotation_id] in {DependencyMatrix, TagSequence}
            # Call the static "from_tensor" method for the correct annotation class
            annotation_data[annotation_id] = annotation_types[annotation_id].from_tensor(tokens,
                                                                                         label_tensors[annotation_id],
                                                                                         label_vocabs[annotation_id])

        return AnnotatedSentence(tokens, annotation_data, multiword_tokens=multiword_tokens)

    def tokens_no_root(self):
        """Return the "raw" tokens of this sentences, i.e. everything except the [root] token at the start."""
        assert self.tokens[0] == "[root]"
        return self.tokens[1:]

    def to_conll(self, column_mapping, num_cols=10):
        """Output a string that contains this annotated sentence in custom CoNLL format.

        Args:
            column_mapping: Mapping from the annotation ID to the column in which this layer should be displayed.
            num_cols: Overall number of columns. Columns which were not specified in column_mapping are filled with
                underscores (_).

        Returns:
            A string containing CoNLL lines representing the sentence and its annotations.
        """
        token_ids = [str(ix) for ix in range(1, len(self.tokens) - 1 + 1)]  # Remember that there is a [root] token
        token_forms = self.tokens_no_root()                                 # (which we omit here)

        assert len(token_ids) == len(token_forms)

        conll_columns = [["_"] * len(token_ids)] * num_cols
        conll_columns[0] = token_ids
        conll_columns[1] = token_forms
        for annotation_id, column_ix in column_mapping.items():
            if isinstance(column_ix, tuple) or isinstance(column_ix, list):  # Special case for splitting dependencies into head+label
                assert len(column_ix) == 2
                assert isinstance(self[annotation_id], DependencyMatrix)
                heads_column, labels_column = self.annotation_data[annotation_id].to_conll(split_heads_labels=True)
                conll_columns[column_ix[0]] = heads_column
                conll_columns[column_ix[1]] = labels_column
            else:
                if column_ix == 0 or column_ix == 1:
                    raise Exception("Columns 0 and 1 are reserved for token IDs and token forms!")
                conll_columns[column_ix] = self.annotation_data[annotation_id].to_conll()

        conllu_lines = ["\t".join(row) for row in zip(*conll_columns)]

        # Insert multiword tokens at the proper indices
        for ix in sorted(self.multiword_tokens.keys(), reverse=True):
            mwt_id, mwt_form = self.multiword_tokens[ix]
            conllu_lines.insert(ix - 1, "\t".join([mwt_id, mwt_form] + ["_"] * (num_cols - 2)))

        return "\n".join(conllu_lines) + "\n"

    @staticmethod
    def get_tensorized_annotations(sentences, label_vocabs):
        """For an iterable of AnnotatedSentences and each annotation layer, get annotations as a batched tensor,
        using the provided dictionary of label vocabularies for label->index conversion of each annotation layer.

        Args:
            sentences: The sentences whose annotations to convert to tensors (iterable of AnnotatedSentence).
            label_vocabs: Dictionary that maps annotation IDs to vocabularies for label->index conversion.

        Returns:
            A dictionary that maps annotation IDs to tensors containing label indices. For annotation layers of type
            TagSequence, tensors are of shape `(num_sentences, max_sent_length)`. For annotation layers of type
            DependencyMatrix, tensors are of shape `(num_sentences, max_sent_length**2)`.

            """
        target_tensors = dict()

        annotation_ids = set(label_vocabs.keys())
        for annotation_id in annotation_ids:
            # Make sure that all sentences have the same annotation layers
            assert all(isinstance(sent[annotation_id], DependencyMatrix) for sent in sentences) or \
                   all(isinstance(sent[annotation_id], TagSequence) for sent in sentences)

            if isinstance(sentences[0][annotation_id], DependencyMatrix):
                target_tensors[annotation_id] = AnnotatedSentence._get_tensorized_dependencies(sentences, annotation_id, label_vocabs[annotation_id])
            elif isinstance(sentences[0][annotation_id], TagSequence):
                target_tensors[annotation_id] = AnnotatedSentence._get_tensorized_tags(sentences, annotation_id, label_vocabs[annotation_id])

        return target_tensors

    @staticmethod
    def _get_tensorized_dependencies(sentences, annotation_id, label_vocab):
        """For an iterable of AnnotatedSentences, create a batched label index matrix tensor for the given (dependency)
        annotation layer. Shape is (num_sentences, max_sent_length**2).
        """
        assert all(isinstance(sent[annotation_id], DependencyMatrix) for sent in sentences)

        dep_matrices = [sent[annotation_id].as_index_matrix(label_vocab) for sent in sentences]
        max_len_dep = max(len(dep_matrix) for dep_matrix in dep_matrices)
        for dep_matrix in dep_matrices:
            dep_matrix.tensorize(padded_length=max_len_dep)
        dep_matrix_batch = torch.stack([dep_matrix.data for dep_matrix in dep_matrices])
        return dep_matrix_batch

    @staticmethod
    def _get_tensorized_tags(sentences, annotation_id, label_vocab):
        """For an iterable of AnnotatedSentences, create a batched tag index tensor for the given (tag) annotation layer.
        Shape is (num_sentences, max_sent_length).
        """
        assert all(isinstance(sent[annotation_id], TagSequence) for sent in sentences)

        tag_indices = [[label_vocab.token2ix(tag_label) for tag_label in sent[annotation_id]] for sent in sentences]
        max_len_tags = max(len(tag_ixs) for tag_ixs in tag_indices)
        for tag_ixs in tag_indices:
            padding_length = max_len_tags - len(tag_ixs)
            for _ in range(padding_length):
                tag_ixs.append(-1)

        tags_batch = torch.stack([torch.tensor(tag_ixs) for tag_ixs in tag_indices])

        return tags_batch

    @staticmethod
    def get_annotation_counts(gold, predicted):
        """Given a pair of AnnotatedSentence objects over the same sequence of tokens (one representing the
        gold-standard annotation, the other the system output), compute per-label annotation counts (predicted, gold,
        correct) for each annotation layer.

        Args:
            gold: AnnotatedSentence representing the gold-standard annotation.
            predicted: AnnotatedSentence representing the system output.

        Returns:
            A nested dictionary (annotation_id -> label -> "predicted"/"gold"/"correct") of counts.
        """
        assert gold.tokens == predicted.tokens

        counts = dict()
        for annotation_id in gold.annotation_data:
            if annotation_id in predicted.annotation_data:
                assert type(gold[annotation_id]) == type(predicted[annotation_id])
                annotation_type = type(gold[annotation_id])
                counts[annotation_id] = annotation_type.get_annotation_counts(gold[annotation_id],
                                                                              predicted[annotation_id])

        return counts
