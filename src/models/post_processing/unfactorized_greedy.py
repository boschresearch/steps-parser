#  Copyright (c) 2020 Robert Bosch GmbH
#  All rights reserved.
#
#  This source code is licensed under the AGPL v3 license found in the
#  LICENSE file in the root directory of this source tree.
#
#  Author: Stefan Grünewald

import torch

from abc import abstractmethod
from torch.nn.functional import softmax

from models.post_processing.post_processor import PostProcessor
from data_handling.dependency_matrix import dependents


class UnfactorizedGreedyPostProcessor(PostProcessor):
    """Abstract base class for greedily/heuristically post-processing the predicted unfactorized dependencies of a
    parsed sentence in order to ensure that they form a valid graph/tree. This entails the following steps:

      * Remove self-loops
      * Remove superfluous heads of tokens which should only have one head (e.g. punctuation)
      * Ensure that there is exactly one root
      * Check if every node is reachable from the root
        * If not: Greedily connect unreachable nodes to the rest of the graph.
    """
    def __init__(self, annotation_ids, vocabs):
        """
        Args:
            annotation_ids: Must be a single-element list containing the annotation ID of the dependency label matrix.
            vocabs: Dictionary mapping annotation IDs (in this case, the single annotation ID) to label vocabularies.
        """
        super(UnfactorizedGreedyPostProcessor, self).__init__(annotation_ids, vocabs)
        assert len(annotation_ids) == 1
        self.deps_id, = self.annotation_ids

    def post_process(self, sentence, logits):
        dependencies = sentence[self.deps_id]
        logits = logits[self.deps_id]

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

            if self.has_superfluous_heads(head_relations):
                self.remove_superfluous_heads(dependencies, head_indices, j, logits)

        # Ensure that the sentence has one root
        sent_roots = self.get_sentence_roots(dependencies)
        if 0 in sent_roots or len(sent_roots) != 1:
            self.enforce_singular_root(dependencies, logits)

        # Check if every node is reachable from the root; if not, greedily connect to the rest of the graph
        reachable_from_root, not_reachable_from_root = self.get_reachable_from_root(dependencies)
        if not_reachable_from_root:
            self.connect(dependencies, reachable_from_root, not_reachable_from_root, logits)

    @abstractmethod
    def has_superfluous_heads(self, head_relations):
        """Abstract method that determines under which circumstances a token is judged to have too many heads."""
        pass

    def remove_superfluous_heads(self, dependencies, head_indices, j, logits):
        """Given a sentence and a token j, remove all head relations except the most confidently predicted one."""
        vocab = self.vocabs[self.deps_id]  # Dependency label vocab

        # Extract the relevant logits for the j'th token and its heads and compute softmax
        head_indices = torch.tensor(head_indices)
        relevant_logits = logits.view((len(dependencies), len(dependencies), -1))[head_indices, j, :]
        probs = softmax(relevant_logits, dim=1)
        assert probs.shape == (len(head_indices), len(vocab))

        # Best relation is the one with the highest top probability
        top_probs = torch.max(probs, dim=1)[0]
        best_top_prob_ix = torch.argmax(top_probs)
        best_head = head_indices[best_top_prob_ix]

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

    def enforce_singular_root(self, dependencies, logits):
        best_root_ix = self.find_best_root(dependencies, logits)

        for i in range(len(dependencies)):
            dependencies[0][i] = "root" if i == best_root_ix else "[null]"
            dependencies[i][best_root_ix] = "root" if i == 0 else "[null]"

    def find_best_root(self, dependencies, logits):
        root_label_ix = self.vocabs[self.deps_id].token2ix("root")

        probs = softmax(logits.view((len(dependencies), len(dependencies), -1)), dim=2)
        root_probs = probs[0, 1:, root_label_ix]

        return int(torch.argmax(root_probs)) + 1

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

    def connect(self, dependencies, reachable_from_root, not_reachable_from_root, logits):
        vocab = self.vocabs[self.deps_id]

        probs = softmax(logits.view((len(dependencies), len(dependencies), -1)), dim=2)
        assert probs.shape == (len(dependencies), len(dependencies), len(vocab))

        reachable_ix = torch.tensor(list(reachable_from_root))
        unreachable_ix = torch.tensor(list(not_reachable_from_root))

        # We are only interested in non-[null] edges from reachable nodes (except [root]) to unreachable nodes:
        probs[:, :, 0] = 0            # Set probabilities of [null] edges to zero
        probs[unreachable_ix, :] = 0  # Set probabilities of all edges originating from unreachable nodes to zero
        probs[:, reachable_ix] = 0    # Set probabilities of all edges ending at reachable nodes to zero
        probs[0, :] = 0               # Set probabilities of all edges originating at [root] to zero

        best = int(torch.argmax(probs))

        # Because argmax flattens dimensions, we need to do some arithmetic here:
        head_ix = best // len(vocab) // len(dependencies)      # Determine row (for head index)
        dependent_ix = best // len(vocab) % len(dependencies)  # Determine column (for dependent index)
        relation_ix = best % len(vocab)                        # Determine label index (for dependency label)
        relation = vocab.ix2token(relation_ix)                 # Look up actual label

        # Set the head of the determined dependent to be the determined head
        self.set_head(dependencies, dependent_ix, head_ix, relation)

        # Check if there are still unreachable nodes. If so, repeat the above procedure until the graph is connected.
        reachable_from_root, not_reachable_from_root = self.get_reachable_from_root(dependencies)
        if not_reachable_from_root:
            self.connect(dependencies, reachable_from_root, not_reachable_from_root, logits)

    @abstractmethod
    def set_head(self, dependencies, dependent_ix, head_ix, relation):
        """Abstract class which determines how to set the head of a token."""
        pass
