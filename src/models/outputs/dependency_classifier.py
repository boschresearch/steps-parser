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

import models.outputs.biaffine as scorer_module


class DependencyClassifier(nn.Module):
    """Module for classifying (syntactic/semantic) dependencies between pairs of tokens. Every token pair is mapped
    onto a logits vector with the dimensionality of the specified output label vocabulary. Label indices are then
    extracted from these logits.
    """
    def __init__(self, input_dim, vocab, scorer_class, hidden_size, dropout=0.0):
        """
        Args:
            input_dim: Dimensionality of input vectors.
            vocab: Vocabulary of output labels.
            scorer_class: Which class to use for scoring arcs, e.g. DeepBiaffineScorer.
            hidden_size: Hidden size of the scorer module.
            dropout: Dropout ratio of the scorer module.
        """
        super(DependencyClassifier, self).__init__()
        self.vocab = vocab
        self.scorer = getattr(scorer_module, scorer_class)(input1_size=input_dim, input2_size=input_dim,
                                                           hidden_size=hidden_size, output_size=len(self.vocab),
                                                           hidden_func=F.relu, dropout=dropout)

    def forward(self, embeddings_batch, true_seq_lengths):
        """Take a batch of embedding sequences and feed them to the classifier to obtain logits for dependency labels
        for each pair of tokens.

        Args:
            embeddings_batch: Tensor (shape: batch_size * max_seq_len * embeddings_dim) containing input embeddings.
            true_seq_lengths: Tensor (shape: batch_size) containing the true (=non-padded) lengths of the sentences.

        Returns: A tuple consisting of (a) a tensor containing the output logits of the dependency classifier
          ("flattened"; shape: `batch_size * (max_seq_len**2) * vocab_size`); (b) a tensor containing the actual
          predictions, in the form of label indices ("flattened"; shape: `batch_size * (max_seq_len**2)`).
        """
        batch_size = embeddings_batch.shape[0]
        seq_len = embeddings_batch.shape[1]

        assert len(true_seq_lengths) == batch_size

        # Run the scorer on all the token pairs
        logits = self.scorer(embeddings_batch, embeddings_batch)
        labels = torch.argmax(logits, dim=-1)

        # Return "flattened" version of the logits and labels.
        return logits.view(batch_size, seq_len * seq_len, -1),\
               labels.view(batch_size, seq_len * seq_len)
