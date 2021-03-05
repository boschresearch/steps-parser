#  Copyright (c) 2020 Robert Bosch GmbH
#  All rights reserved.
#
#  This source code is licensed under the AGPL v3 license found in the
#  LICENSE file in the root directory of this source tree.
#
#  Author: Stefan Grünewald

from torch.nn.functional import log_softmax

from models.post_processing.post_processor import PostProcessor
from util.chuliu_edmonds import chuliu_edmonds_one_root


class FactorizedMSTPostProcessor(PostProcessor):
    """Post-processor to assemble a basic dependency tree from a logits tensor of arc scores (output of ArcScorer) by
    using the Chu-Liu/Edmonds MST algorithm and only keeping those entries in the sentence's DependencyMatrix which
    correspond to tree edges.

    (This is the "usual" method for graph-based parsing of syntactic dependency trees.)
    """

    def __init__(self, annotation_ids, vocabs):
        """
        Args:
            annotation_ids: Must be a list containing two elements: (1) the annotation ID of the unlabeled arc matrix;
              (2) the annotation ID of the dependency label matrix.
            vocabs: Dictionary mapping annotation IDs to label vocabularies.
        """
        super(FactorizedMSTPostProcessor, self).__init__(annotation_ids, vocabs)
        assert len(self.annotation_ids) == 2
        self.heads_id, self.labels_id = self.annotation_ids

    def post_process(self, sentence, logits):
        head_logits = logits[self.heads_id]

        assert len(head_logits.shape) == 2
        assert head_logits.shape[0] == head_logits.shape[1]  # Logits must be a square matrix

        head_logprobs = log_softmax(head_logits, dim=1).detach().cpu().numpy()
        head_indices = chuliu_edmonds_one_root(head_logprobs)

        self.retrieve_labeled_dependency_tree(sentence, head_indices)

    def retrieve_labeled_dependency_tree(self, sentence, head_indices):
        heads = sentence[self.heads_id]
        labels = sentence[self.labels_id]

        # Assign heads computed by MST algorithm
        heads.data[:] = [self.vocabs[self.heads_id].ix2token(int(head_ix)) for head_ix in head_indices]

        # Set the first column of the dependency matrix to [null] (root does not have a head)
        for i in range(len(labels)):
            labels[i][0] = "[null]"

        # Set all dependency labels which are not part of the extracted tree to [null]
        for dependent_ix in range(1, len(labels)):
            true_head_ix = head_indices[dependent_ix]
            for head_ix in range(len(labels)):
                if head_ix != true_head_ix:
                    labels[head_ix][dependent_ix] = "[null]"
                elif head_ix == true_head_ix == 0:
                    labels[head_ix][dependent_ix] = "root"
