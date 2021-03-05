#  Copyright (c) 2020 Robert Bosch GmbH
#  All rights reserved.
#
#  This source code is licensed under the AGPL v3 license found in the
#  LICENSE file in the root directory of this source tree.
#
#  Author: Stefan Grünewald

import torch
import torch.nn.functional as F

from torch import nn

from models.outputs.mlp import MLP


class SequenceTagger(nn.Module):
    """Module for tagging a sequence of tokens using a simple feedforward network architecture."""
    def __init__(self, input_dim, vocab, hidden_size, input_dropout=0.0, hidden_dropout=0.0):
        """
        Args:
            input_dim: Dimensions of input vectors.
            vocab: Vocabulary of output labels.
            hidden_size: Dimensions of hidden layer.
            input_dropout: Dropout ratio to apply to input. Default: 0.0.
            hidden_dropout: Dropout ratio to apply to hidden layer. Default: 0.0.
        """
        super(SequenceTagger, self).__init__()
        self.vocab = vocab
        self.mlp = MLP(n_in=input_dim, n_hidden=hidden_size, n_out=len(self.vocab), hidden_activation=F.relu,
                       input_dropout=input_dropout, hidden_dropout=hidden_dropout)

    def forward(self, embeddings_batch, true_seq_lengths):
        """Take a batch of embedding sequences and feed them to the MLP to obtain logits for tag labels.

        Args:
            embeddings_batch: Tensor (shape: batch_size * max_seq_len * embeddings_dim) containing input embeddings.
            true_seq_lengths: Tensor (shape: batch_size) containing the true (=non-padded) lengths of the sentences.

        Returns: A tuple consisting of (a) a tensor containing the output logits of the tagger
        (shape: batch_size * max_seq_len * vocab_size); (b) a tensor containing the actual
          predictions, in the form of label indices (shape: batch_size * seq_len).
        """
        logits = self.mlp(embeddings_batch)    # Shape: (batch_size, seq_len, len(output_vocab))
        labels = torch.argmax(logits, dim=-1)  # Shape: (batch_size, seq_len)

        return logits, labels
