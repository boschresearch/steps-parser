#  Copyright (c) 2020 Robert Bosch GmbH
#  All rights reserved.
#
#  This source code is licensed under the AGPL v3 license found in the
#  LICENSE file in the root directory of this source tree.
#
#  Author: Stefan Grünewald

from collections import defaultdict


class TagSequence:
    """An object of this class represents a sequence of tags over a sentence (as part of an AnnotatedSentence object),
    e.g. part-of-speech tags.
    """

    def __init__(self, data):
        """
        Args:
            data: List of tags (strings) to initialize this TagSequence with. The first tag should be `ROOT`, in line
              with the "[root]" token that is prepended to each AnnotatedSentence.
        """
        self.data = data

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def to_conll(self):
        """Convert this TagSequence object into CoNLL annotations.

        Returns:
            A list containing the entries of a CoNLL annotation column representing this tag sequence. The list simply
            contains the tags within the sequence, except for the first one (which should be the dummy `ROOT` tag).
        """
        return self.data[1:]

    @staticmethod
    def from_conll(conll_lines, annotation_column, ignore_root=False):
        """Create a TagSequence from a CoNLL-annotated sentence.

        Args:
            conll_lines: Iterable of lines (strings) representing the sentence, in CoNLL format.
            annotation_column: Which column to extract tags from.
            ignore_root: If true, the first tag in the sequence is set to `__IGNORE__` instead of `ROOT`.

        Returns:
            A TagSequence object containing the tags read from the CoNLL data.
        """
        data = ["__IGNORE__"] if ignore_root else ["ROOT"]
        for line in conll_lines:
            assert line  # Lines must not be empty
            if line.startswith("#"):
                continue  # Ignore metadata

            line = line.strip()
            elements = line.split('\t')
            token_id = elements[0]
            if "-" in token_id:
                continue

            tag = elements[annotation_column]
            data.append(tag)

        return TagSequence(data)

    @staticmethod
    def from_tensor(tokens, label_tensor, label_vocab):
        """Create a TagSequence from a tensor containing label indices using the specified label vocabulary.

        Args:
            tokens: The tokens of the sentence associated with the tag label indices in the tensor.
            label_tensor: The tensor to read label indices from. Should be 1-dimensional and have at least
              `len(tokens)` entries.
            label_vocab: Label vocabulary to translate label indices into actual tag labels.

        Returns:
            A TagSequence object containing the tags read from the label index tensor.
        """
        tags = list()
        for lbl_ix in label_tensor:
            lbl_ix = int(lbl_ix)
            tags.append(label_vocab.ix2token(lbl_ix))
        tags = tags[0:len(tokens)]

        return TagSequence(tags)

    @staticmethod
    def get_annotation_counts(gold, predicted):
        """Compare a system-created TagSequence with the corresponding gold-standard TagSequence.
        For each tag type, return the counts for

          * how often this tag occurred in the gold standard ("gold")
          * how often the system predicted this tag ("predicted")
          * how often the gold tag and predicted tag were identical ("correct")

        The above metrics will also be calculated for an artificial tag ("TOTAL") that represents the sum of the
        metrics over all tag types.

        Note that the artificial "ROOT" tag (which is prepended to each TagSequence) is ignored in this evaluation.

        Args:
            gold: A TagSequence containing gold dependencies.
            predicted: A TagSequence containing predicted dependencies.

        Returns:
            A nested dictionary (label -> "predicted"/"gold"/"correct") of counts.
        """
        assert len(predicted) == len(gold)
        assert gold[0] == "ROOT" or gold[0] == "__IGNORE__"

        counts = defaultdict(lambda: {"predicted": 0, "gold": 0, "correct": 0})

        for gold_tag, predicted_tag in zip(gold.data[1:], predicted.data[1:]):  # Skip tag for the artificial root
            if gold_tag == "__IGNORE__":
                continue

            counts[gold_tag]["gold"] += 1
            counts["TOTAL"]["gold"] += 1

            counts[predicted_tag]["predicted"] += 1
            counts["TOTAL"]["predicted"] += 1

            if predicted_tag == gold_tag:
                counts[predicted_tag]["correct"] += 1
                counts["TOTAL"]["correct"] += 1

        return counts
