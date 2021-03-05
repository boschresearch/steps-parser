#  Copyright (c) 2020 Robert Bosch GmbH
#  All rights reserved.
#
#  This source code is licensed under the AGPL v3 license found in the
#  LICENSE file in the root directory of this source tree.
#
#  Author: Stefan Grünewald

from models.post_processing.unfactorized_greedy import UnfactorizedGreedyPostProcessor

from util.lexicalize import lexicalize


class UnfactorizedGreedyGraphPostProcessor(UnfactorizedGreedyPostProcessor):
    """Class for greedily/heuristically post-processing the predicted unfactorized dependencies of a parsed sentence in
    order to ensure that they form a valid graph.

    This class overrides the has_superfluous_heads and set_head methods from UnfactorizedGreedyPostProcessor in order
    to produce dependency graphs as output. Additionally, it performs label lexicalization.
    """
    def post_process(self, sentence, logits):
        super(UnfactorizedGreedyGraphPostProcessor, self).post_process(sentence, logits)

        dependencies = sentence[self.deps_id]
        lexicalize(dependencies, sentence.tokens)

    def has_superfluous_heads(self, head_relations):
        """Check if the given set of head relations is inconsistent (e.g. more than one punct relation)."""
        if len(head_relations) <= 1:
            return False

        # If the token has more than one head, something fishy is going on if it is attached via one of the following
        # relations. (Note that this was determined empirically and may be language-dependent!)
        unitary_relations = {"fixed", "flat", "goeswith", "punct", "cc"}
        if set(head_relations) & unitary_relations:
            return True
        else:
            return False

    def set_head(self, dependencies, dependent_ix, head_ix, relation):
        dependencies[head_ix][dependent_ix] = relation
