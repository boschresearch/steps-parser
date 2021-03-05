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


class ArcScorer(nn.Module):
    """Module for assigning a score to every ordered pair of tokens. Very similar to DependencyClassifier, but every
    pair of tokens only receives a single scalar value instead of a distribution over some label vocabulary.

    These scores can then be used for single head prediction (head_mode=="single-head"; use with softmax loss) or
    for multi-head prediction (head_mode=="multi-head"; use with sigmoid loss).

    NOTE: You have to be careful here with how heads vs. dependents are encoded. When using the single-head setting,
    rows contain scores for possible heads, meaning each row represents a dependent. When using the multi-head
    setting with a DependencyAnnotatedMatrix(edge_existence==True), rows contain scores for possible dependents, meaning
    each row represents a head. Since every ArcScorer has its own biaffine classifier, this should not be a problem, but
    be careful when combining their outputs in a factorized system.
    """
    def __init__(self, input_dim, vocab, scorer_class, hidden_size, dropout=0.0, head_mode="single_head", minus_inf=-1e30):
        """
        Args:
            input_dim: Dimensionality of input vectors.
            vocab: Vocabulary of output labels.
            scorer_class: Which class to use for scoring arcs, e.g. DeepBiaffineScorer.
            hidden_size: Hidden size of the scorer module.
            dropout: Dropout ratio of the scorer module.
            head_mode: Whether to predict only a single head for each token ("single_head"; uses softmax loss) or
              an arbitrary number of heads ("multi-head"; uses sigmoid loss). Default: "single_head".
            minus_inf: Value to use for masking padding tokens during head extraction. Default: -1e30.
        """
        super(ArcScorer, self).__init__()
        self.scorer = getattr(scorer_module, scorer_class)(input1_size=input_dim, input2_size=input_dim,
                                                           hidden_size=hidden_size, output_size=1,
                                                           hidden_func=F.relu, dropout=dropout)
        self.vocab = vocab

        assert head_mode in {"single_head", "multi_head"}
        self.head_mode = head_mode
        self.minus_inf = torch.FloatTensor([minus_inf])

    def forward(self, embeddings_batch, true_seq_lengths):
        """Take a batch of embedding sequences and feed them to the classifier to obtain a score for each pair of
        tokens.

        Args:
            embeddings_batch: Tensor (shape: batch_size * max_seq_len * embeddings_dim) containing input embeddings.
            true_seq_lengths: Tensor (shape: batch_size) containing the true (=non-padded) lengths of the sentences.

        Returns: A tuple consisting of (a) a tensor containing the output logits of the arc scorer
          ("flattened"; shape: `batch_size * (max_seq_len**2)`); (b) a tensor containing the actual predictions, in the
          form of label indices ("flattened"; shape: `batch_size * (max_seq_len**2)`).
        """
        batch_size = embeddings_batch.shape[0]
        seq_len = embeddings_batch.shape[1]

        assert len(true_seq_lengths) == batch_size

        # Run the scorer on all the token pairs
        logits = self.scorer(embeddings_batch, embeddings_batch)

        # Remove last dimension (which is always of size 1 b/c we're outputting scalar scores)
        logits = logits.squeeze(dim=-1)

        # Set scores for indices which correspond to padding tokens to "-inf" (a very very small value)
        logits = self._mask_illegal_heads(logits, true_seq_lengths)

        # Compute actual labels (== head indices)
        labels = self._extract_labels(logits)

        # Return logits and labels
        if self.head_mode == "single_head":
            return logits, labels
        elif self.head_mode == "multi_head":
            # "Flatten" the output in multi-head setting
            return logits.view(batch_size, seq_len * seq_len),\
                   labels.view(batch_size, seq_len * seq_len)

    def _mask_illegal_heads(self, logits, true_seq_lengths):
        """Set scores for indices which correspond to padding tokens to "-inf" (a very very small value)"""
        batch_size = logits.shape[0]
        raw_seq_len = logits.shape[1]

        if self.minus_inf.device != logits.device:
            self.minus_inf = self.minus_inf.to(logits.device)

        ixs = torch.arange(logits.numel(), device=logits.device).view(logits.shape)
        row_ixs = ixs[:, :, 0].unsqueeze(2).expand(ixs.shape)

        true_seq_len_expanded = true_seq_lengths.unsqueeze(1).unsqueeze(2).expand(logits.shape)

        mask = (ixs >= row_ixs + true_seq_len_expanded).int()
        filler = self.minus_inf.unsqueeze(0).unsqueeze(1).expand(mask.shape)

        logits += mask*filler

        return logits

    def _extract_labels(self, logits):
        if self.head_mode == "single_head":
            labels = torch.argmax(logits, dim=-1)
        elif self.head_mode == "multi_head":
            labels = torch.where(logits > 0, torch.tensor(1, device=logits.device), torch.tensor(0, device=logits.device))
        else:
            raise Exception("Unknown head mode {}!".format(self.head_mode))

        if self.head_mode == "single_head":
            assert len(logits.shape) == 3
            assert len(labels.shape) == 2
            assert logits.shape[0:2] == labels.shape
        elif self.head_mode == "multi_head":
            assert logits.shape == labels.shape

        return labels
