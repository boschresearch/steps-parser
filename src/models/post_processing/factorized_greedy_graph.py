#  Copyright (c) 2020 Robert Bosch GmbH
#  All rights reserved.
#
#  This source code is licensed under the AGPL v3 license found in the
#  LICENSE file in the root directory of this source tree.
#
#  Author: Stefan Grünewald

import torch
import re

from copy import deepcopy

from data_handling.dependency_matrix import heads, dependents
from models.post_processing.post_processor import PostProcessor

from util.lexicalize import lexicalize


class FactorizedGreedyGraphPostProcessor(PostProcessor):
    """An object of this class post-processes the (factorized) enhanced dependencies of a parsed sentence to form a
    valid dependency graph. This entails the following steps:

      * Extract the labeled dependency graph from heads and labels
      * Remove self-loops
      * Remove superfluous heads of tokens which should only have one head (e.g. punctuation)
      * Ensure that there is exactly one root
      * Greedily connect nodes which cannot be reached from the root
      * Add lexical information to placeholder labels (e.g. obl:[case] -> obl:in)
    """
    def __init__(self, annotation_ids, vocabs):
        """
        Args:
            annotation_ids: Must be a list containing two elements: (1) the annotation ID of the unlabeled arc matrix;
              (2) the annotation ID of the dependency label matrix.
            vocabs: Dictionary mapping annotation IDs to label vocabularies.
        """
        super(FactorizedGreedyGraphPostProcessor, self).__init__(annotation_ids, vocabs)
        assert len(self.annotation_ids) == 2
        self.heads_id, self.labels_id = self.annotation_ids
        self.vocabs = vocabs

    def post_process(self, sentence, logits):
        raw_labels = deepcopy(sentence[self.labels_id])  # Remember all the predicted labels for later

        self.extract_labeled_dependency_graph(sentence)

        dependencies = sentence[self.labels_id]
        for j in range(1, len(dependencies)):
            head_indices = list()
            head_relations = list()
            for i in range(len(dependencies)):
                if i == j:
                    dependencies[i][j] = "[null]"  # Remove self-loops

                head_relation = dependencies[i][j]
                if head_relation != "[null]":
                    head_indices.append(i)
                    head_relations.append(head_relation)

            if self.inconsistent_heads(head_relations):  # Detect & remove inconsistent head relations
                self.remove_superfluous_heads(dependencies, head_indices, j, logits[self.heads_id])

        # Ensure that the sentence has exactly one root
        sent_roots = self.get_sentence_roots(dependencies)
        if 0 in sent_roots or len(sent_roots) != 1:
            self.enforce_singular_root(dependencies, logits[self.heads_id])

        # Check if every node is reachable from the root; if not, greedily connect to the rest of the graph
        reachable_from_root, not_reachable_from_root = self.get_reachable_from_root(dependencies)
        if not_reachable_from_root:
            self.connect_graph(dependencies, reachable_from_root, not_reachable_from_root, logits[self.heads_id], raw_labels)

        # Lexicalize relations
        lexicalize(dependencies, sentence.tokens)

    def extract_labeled_dependency_graph(self, sentence):
        """Keep only those dependency labels where an arc was predicted."""
        arcs = sentence[self.heads_id]
        labels = sentence[self.labels_id]

        for i in range(len(arcs)):
            for j in range(len(arcs)):
                if arcs[i][j] == "[null]":
                    labels[i][j] = "[null]"
                else:
                    assert arcs[i][j] == "[edge]"

    def inconsistent_heads(self, head_relations):
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

    def remove_superfluous_heads(self, dependencies, head_indices, j, arc_logits):
        """Given a sentence and a token j, remove all head relations except the most confidently predicted one."""
        # Extract the relevant logits for the j'th token and its heads and compute sigmoid
        head_indices = torch.tensor(head_indices)
        relevant_logits = arc_logits.view((len(dependencies), len(dependencies)))[head_indices, j]

        # Best head is the one with highest-scoring arc
        best_head_ix = torch.argmax(relevant_logits)
        best_head = head_indices[best_head_ix]

        # Remove all head relations except the best one
        for i in range(len(dependencies)):
            if i != best_head:
                dependencies[i][j] = "[null]"

    def get_sentence_roots(self, dependencies):
        roots = list()
        for dependent_ix, lbl in enumerate(dependencies[0]):
            if lbl == "root":
                roots.append(dependent_ix)

        return roots

    def enforce_singular_root(self, dependencies, arc_logits):
        best_root_ix = self.find_best_root(dependencies, arc_logits)

        for i in range(len(dependencies)):
            dependencies[0][i] = "root" if i == best_root_ix else "[null]"
            dependencies[i][best_root_ix] = "root" if i == 0 else "[null]"

    def find_best_root(self, dependencies, arc_logits):
        root_scores = arc_logits.view((len(dependencies), len(dependencies)))[0, 1:]

        return int(torch.argmax(root_scores)) + 1

    def get_reachable_from_root(self, dependencies):
        reachable_from_root = self.get_reachable_from(dependencies, 0, set())
        not_reachable_from_root = set(range(len(dependencies))) - reachable_from_root

        return reachable_from_root, not_reachable_from_root

    def get_reachable_from(self, dependencies, node_ix, encountered_nodes):
        encountered_nodes.add(node_ix)

        for dependent_id, _ in dependents(dependencies, node_ix):
            if dependent_id not in encountered_nodes:
                encountered_nodes |= self.get_reachable_from(dependencies, dependent_id, encountered_nodes)

        return encountered_nodes

    def connect_graph(self, dependencies, reachable_from_root, not_reachable_from_root, arc_logits, raw_labels):
        arc_scores = arc_logits.view((len(dependencies), len(dependencies)))

        reachable_ix = torch.tensor(list(reachable_from_root))
        unreachable_ix = torch.tensor(list(not_reachable_from_root))

        # We are only interested in non-[null] edges from reachable nodes (except [root]) to unreachable nodes:
        arc_scores[unreachable_ix, :] = -1e30  # Set scores of all arcs originating from unreachable nodes to -inf
        arc_scores[:, reachable_ix] = -1e30    # Set scores of all arcs ending at reachable nodes to -inf
        arc_scores[0, :] = -1e30               # Set probabilities of all edges originating at [root] to -inf

        best = int(torch.argmax(arc_scores))

        # Because argmax flattens dimensions, we need to do some arithmetic here:
        head_ix = best // len(dependencies)      # Determine row (for head index)
        dependent_ix = best % len(dependencies)  # Determine column (for dependent index)

        relation = raw_labels[head_ix][dependent_ix]  # Look up label

        # Add the edge between the determined head and the determined dependent
        dependencies[head_ix][dependent_ix] = relation

        # Check if there are still unreachable nodes. If so, repeat the above procedure until the graph is connected.
        reachable_from_root, not_reachable_from_root = self.get_reachable_from_root(dependencies)
        if not_reachable_from_root:
            self.connect_graph(dependencies, reachable_from_root, not_reachable_from_root, arc_logits, raw_labels)
