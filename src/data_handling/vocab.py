#  Copyright (c) 2020 Robert Bosch GmbH
#  All rights reserved.
#
#  This source code is licensed under the AGPL v3 license found in the
#  LICENSE file in the root directory of this source tree.
#
#  Author: Stefan Grünewald

class BasicVocab:
    """Class for mapping labels/tokens to indices and vice versa."""

    def __init__(self, vocab_filename=None, ignore_label="__IGNORE__", ignore_index=-1):
        """A vocabulary is read from a file in which each label constitutes one line. The index associated with each
        label is the index of the line that label occurred in (counting from 0).

        In addition, a special label and index (`ignore_label` and `ignore_index`) are added to signify content which
        should be ignored in parsing/tagging tasks.

        Args:
            vocab_filename: Name of the file to read the vocabulary from.
            ignore_label: Special label signifying ignored content. Default: `__IGNORE__`.
            ignore_index: Special index signifying ignored content. Should be negative to avoid collisions with "true"
              indices. Default: `-1`.
        """
        self.ix2token_data = dict()
        self.token2ix_data = dict()

        self.vocab_filename = vocab_filename

        if self.vocab_filename is not None:
            with open(vocab_filename) as vocab_file:
                for ix, line in enumerate(vocab_file):
                    token = line.strip()

                    self.ix2token_data[ix] = token
                    self.token2ix_data[token] = ix

        self.ignore_label = ignore_label
        self.ignore_index = ignore_index

        self.ix2token_data[ignore_index] = ignore_label
        self.token2ix_data[ignore_label] = ignore_index

        assert self.is_consistent()

    def __len__(self):
        return len(self.ix2token_data) - 1  # Do not count built-in "ignore" label

    def __str__(self):
        # Do not consider built-in "ignore" label
        return "\n".join(self.ix2token_data[ix] for ix in sorted(self.ix2token_data.keys()) if ix >= 0)

    def ix2token(self, ix):
        """Get the token associated with index `ix`."""
        return self.ix2token_data[ix]

    def token2ix(self, token):
        """Get the index associated with token `token`."""
        return self.token2ix_data[token]

    def add(self, token):
        """Adds a token to the vocabulary if it does not already exist."""
        if token not in self.token2ix_data:
            new_ix = len(self)

            self.token2ix_data[token] = new_ix
            self.ix2token_data[new_ix] = token

    def to_file(self, vocab_filename):
        """Write vocabulary to a file."""
        with open(vocab_filename, "w") as vocab_file:
            vocab_file.write(str(self))

    def is_consistent(self):
        """Checks if all index mappings match up. Used for debugging."""
        if len(self.ix2token_data) != len(self.token2ix_data):
            return False

        try:
            for token, ix in self.token2ix_data.items():
                if self.ix2token_data[ix] != token:
                    return False
        except IndexError:
            return False

        if "[null]" in self.token2ix_data:
            assert self.token2ix_data["[null]"] == 0

        return True


class IntegerVocab:
    """Class for mapping strings representing non-negative integers to the actual numbers, and vice versa. (Used for
    prediction of head indices in dependency parsing.)

    A special label and index (`ignore_label` and `ignore_index`) are added to signify content which should be ignored
    in parsing/tagging tasks.
    """

    def __init__(self, ignore_label="__IGNORE__", ignore_index=-1):
        """
        Args:
            ignore_label: Special label signifying ignored content. Default: `__IGNORE__`.
            ignore_index: Special index signifying ignored content. Should be negative to avoid collisions with "true"
              indices. Default: `-1`.
        """
        self.ignore_label = ignore_label
        self.ignore_index = ignore_index

    def __len__(self):
        raise Exception("Length of IntegerVocab is undefined.")

    def __str__(self):
        raise Exception("Cannot convert IntegerVocab to str.")

    def ix2token(self, ix):
        """Get the token associated with index `ix`."""
        assert isinstance(ix, int)

        if ix == self.ignore_index:
            return self.ignore_label
        else:
            assert ix >= 0  # Head indices should not be negative
            return str(ix)

    def token2ix(self, token):
        """Get the index associated with token `token`."""
        assert isinstance(token, str)

        if token == self.ignore_label:
            return self.ignore_index
        else:
            assert int(token) >= 0  # Head indices should not be negative
            return int(token)
