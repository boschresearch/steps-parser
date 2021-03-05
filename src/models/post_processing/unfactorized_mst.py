#  Copyright (c) 2020 Robert Bosch GmbH
#  All rights reserved.
#
#  This source code is licensed under the AGPL v3 license found in the
#  LICENSE file in the root directory of this source tree.
#
#  Author: Stefan Grünewald

import torch

from torch.nn.functional import softmax

from models.post_processing.post_processor import PostProcessor
from util.chuliu_edmonds import chuliu_edmonds_one_root


class UnfactorizedMSTPostProcessor(PostProcessor):
    """Class for heuristically post-processing the predicted unfactorized dependencies of a parsed sentence in order to
    extract a spanning tree from them. This entails the following steps:

      * Compute the edge weight for each pair of tokens: log(sum(non-[null] label probabilities))
      * Run the Chu-Liu/Edmonds on the edge weights to extract a maximum spanning tree
      * Retrieve the labeled tree by choosing the highest-scoring non-[null] label for each extracted edge
    """
    def __init__(self, annotation_ids, vocabs):
        """
        Args:
            annotation_ids: Must be a single-element list containing the annotation ID of the dependency label matrix.
            vocabs: Dictionary mapping annotation IDs (in this case, the single annotation ID) to label vocabularies.
        """
        super(UnfactorizedMSTPostProcessor, self).__init__(annotation_ids, vocabs)
        assert len(self.annotation_ids) == 1
        self.deps_id, = self.annotation_ids

    def post_process(self, sentence, logits):
        dependencies = sentence[self.deps_id]
        logits = logits[self.deps_id]

        probs = softmax(logits.view((len(dependencies), len(dependencies), -1)), dim=2)
        edge_weights = self.compute_edge_weights_from_label_probs(probs)
        head_indices = chuliu_edmonds_one_root(edge_weights)

        self.retrieve_labeled_dependency_tree(dependencies, head_indices, probs)

    def compute_edge_weights_from_label_probs(self, probs):
        null_probs = probs[:, :, 0]
        non_null_logprobs = torch.log(1 - null_probs)
        non_null_logprobs_transposed = torch.transpose(non_null_logprobs, 0, 1)

        return non_null_logprobs_transposed.detach().cpu().numpy()

    def retrieve_labeled_dependency_tree(self, dependencies, head_indices, probs):
        vocab = self.vocabs[self.deps_id]  # Dependency label vocab

        # Erase all dependencies from the matrix to start with a "clean slate"
        for i in range(len(dependencies)):
            for j in range(len(dependencies)):
                dependencies[i][j] = "[null]"

        for dependent_ix, head_ix in list(enumerate(head_indices))[1:]:
            probs[head_ix, dependent_ix, 0] = 0  # Make [null] label impossible

            if head_ix == 0:
                relation = "root"
            else:
                relation_ix = torch.argmax(probs[head_ix, dependent_ix])
                relation = vocab.ix2token(int(relation_ix))

            dependencies[head_ix][dependent_ix] = relation

