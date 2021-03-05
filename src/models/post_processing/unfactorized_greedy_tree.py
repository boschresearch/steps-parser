#  Copyright (c) 2020 Robert Bosch GmbH
#  All rights reserved.
#
#  This source code is licensed under the AGPL v3 license found in the
#  LICENSE file in the root directory of this source tree.
#
#  Author: Stefan Grünewald

import torch

from torch.nn.functional import softmax

from data_handling.dependency_matrix import dependents
from models.post_processing.unfactorized_greedy import UnfactorizedGreedyPostProcessor


class UnfactorizedGreedyTreePostProcessor(UnfactorizedGreedyPostProcessor):
    """Class for greedily/heuristically post-processing the predicted unfactorized dependencies of a parsed sentence in
    order to ensure that they form a valid tree.

    This class overrides the has_superfluous_heads and set_head methods from UnfactorizedGreedyPostProcessor in order
    to produce dependency trees as output."""

    def has_superfluous_heads(self, head_relations):
        """In a tree, having more than one head is verboten."""
        return len(head_relations) > 1

    def set_head(self, dependencies, dependent_ix, head_ix, relation):
        """In a tree, we must delete the existing head before assigning a new head."""
        for i in range(len(dependencies)):
            dependencies[i][dependent_ix] = "[null]"
        dependencies[head_ix][dependent_ix] = relation
