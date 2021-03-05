#  Copyright (c) 2020 Robert Bosch GmbH
#  All rights reserved.
#
#  This source code is licensed under the AGPL v3 license found in the
#  LICENSE file in the root directory of this source tree.
#
#  Author: Stefan Grünewald

from collections import defaultdict
from itertools import chain

from data_handling.label_index_matrix import LabelIndexMatrix


class DependencyMatrix:
    """An object of this class represents a matrix of dependency edges between the tokens of a sentence (as part of an
       AnnotatedSentence object), e.g. syntactic dependencies.

       In the matrix, rows represent heads and columns represent dependents. Each cell contains the relation holding
       between the head and the dependent, or a special symbol (`[null]`) in the case of no relation.

       Note that the matrix always contains a row and column for the root of a sentence, in line with the `[root]` token
       that is prepended to each AnnotatedSentence. This means that if the "raw" sentence contains n tokens, the
       dependency matrix will have `(n+1)**2` entries.
    """

    def __init__(self, data):
        """
        Args:
            data: A matrix of dependency relations (represented as a list of lists of dependency labels).
        """
        self.data = data

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def as_index_matrix(self, label_vocab):
        """Convert this DependencyMatrix into a LabelIndexMatrix.

        Args:
            label_vocab: The vocabulary object to use to convert dependency labels to label indices.

        Returns:
            A LabelIndexMatrix object containing the label indices for this dependency matrix.
        """
        return LabelIndexMatrix.from_label_matrix(self.data, label_vocab)

    def to_conll(self, split_heads_labels=False):
        """Convert this DependencyMatrix object into CoNLL annotations.

        Args:
            split_heads_labels: Whether to put dependency head indicies and dependency labels in two separate CoNLL
              columns (default: False).

        Returns:
            If `split_heads_labels==False`, returns a single list containing the entries of a CoNLL annotation column
            representing this dependency matrix. If `split_heads_labels==True`, returns a tuple of two such lists: One
            for dependency head indicies and one for dependency labels.
        """
        if split_heads_labels:
            return self._to_conll_split()
        else:
            return self._to_conll_joint()

    def _to_conll_split(self):
        heads_column = list()
        labels_column = list()

        for dependent_ix in range(1, len(self) - 1 + 1):  # Remember that there is a [root] token which we omit here
            head_ix = None
            dep_label = None

            column = [row[dependent_ix] for row in self]
            for curr_head_ix, relation in enumerate(column):
                if relation not in {"[null]", "__IGNORE__"}:
                    assert head_ix is None and dep_label is None, "Each token must have exactly 1 head when parsing into basic representation"
                    head_ix = curr_head_ix
                    dep_label = relation

            assert head_ix is not None and dep_label is not None, "Each token must have exactly 1 head when parsing into basic representation"

            heads_column.append(str(head_ix))
            labels_column.append(dep_label)

        return heads_column, labels_column

    def _to_conll_joint(self):
        conll_column = list()

        for dependent_ix in range(1, len(self) - 1 + 1):  # Remember that there is a [root] token which we omit here
            deps = list()
            column = [row[dependent_ix] for row in self]
            for head_ix, relation in enumerate(column):
                if relation not in {"[null]", "__IGNORE__"}:
                    deps.append((head_ix, relation))
            conll_column.append("|".join("{}:{}".format(head_ix, relation) for (head_ix, relation) in deps))

        return conll_column

    def pretty_print(self, tokens):
        """Display this dependency matrix as a nicely formatted table.

        Args:
            tokens: The tokens of the sentence.
        """
        assert len(tokens) == len(self)

        # Determine required column width for printing
        col_width = 0
        for token in tokens:
            col_width = max(col_width, len(token))
        for i in range(len(self)):
            for j in range(len(self[i])):
                if len(self[i][j]) > col_width:
                    col_width = max(col_width, len(self[i][j]))
        col_width += 3

        # Print dependency matrix
        print()
        print("".join(token.rjust(col_width) for token in [""] + tokens))
        print()
        for head_ix in range(len(tokens)):
            print(tokens[head_ix].rjust(col_width), end="")
            for dependent_ix in range(len(tokens)):
                print(self[head_ix][dependent_ix].rjust(col_width), end="")
            print()
            print()

    @staticmethod
    def from_conll(conll_lines, annotation_column, id_to_ix, ignore_non_relations=False, ignore_root_column=False,
                   ignore_diagonal=False, ignore_below_diagonal=False, ignore_above_diagonal=False,
                   edge_existence_only=False):
        """
        Create a DependencyMatrix from a CoNLL-annotated sentence.

        Args:
            conll_lines: Iterable of lines (strings) representing the sentence, in CoNLL format.
            annotation_column: Which column(s) to extract dependency information from. Use a tuple of column indices for
              basic representation (where dependency heads and dependency labels are represented in two different
              columns).
            id_to_ix: Dictionary that maps IDs of tokens (such as `8.1`) to their actual position in the sentence.
            ignore_non_relations: If true, non-dependency entries in the matrix are set to `__IGNORE__`. Otherwise,
              they are set to `[null]`. Default: False.
            ignore_root_column: If true, the first column of the matrix is set to `__IGNORE__`. Default: False.
            ignore_diagonal: If true, the diagonal of the matrix is set to `__IGNORE__`. Default: False.
            ignore_below_diagonal: If true, all entries below the diagonal of the matrix are set to `__IGNORE__`.
              Default: False.
            ignore_above_diagonal: If true, all entries above the diagonal of the matrix are set to `__IGNORE__`.
              Default: False.
            edge_existence_only: If true, only store in the matrix whether an edge exists or not (`[edge]` vs.
              `[null]`), discarding dependency labels. Default: False.

        Returns:
            A DependencyMatrix object containing the dependencies read from the CoNLL data.
        """
        filler = "__IGNORE__" if ignore_non_relations else "[null]"
        data = [[filler for i in range(len(id_to_ix))] for j in range(len(id_to_ix))]

        for line in conll_lines:
            line = line.strip()
            elements = line.split('\t')

            dependent_id = elements[0]
            dependent_ix = id_to_ix[dependent_id]

            if isinstance(annotation_column, tuple) or isinstance(annotation_column, list):  # Basic representation
                assert len(annotation_column) == 2
                head_column, label_column = annotation_column
                head, label = elements[head_column], elements[label_column]
                incoming_edges = ["{}:{}".format(head, label)]
            else:  # Enhanced representation
                incoming_edges = elements[annotation_column].split("|")

            if incoming_edges == ["_"]:  # Underscores mean no relation is specified here
                continue

            for incoming_edge in incoming_edges:
                head_id, dependency_type = incoming_edge.split(":", 1)
                head_ix = id_to_ix[head_id]
                data[head_ix][dependent_ix] = dependency_type

        if edge_existence_only:
            data = [["[edge]" if data[i][j] not in {"[null]", "__IGNORE__"} else data[i][j] for j in range(len(data))] for i in range(len(data))]

        if ignore_root_column:
            data = [["__IGNORE__" if j == 0 else data[i][j] for j in range(len(data))] for i in range(len(data))]
        if ignore_diagonal:
            data = [["__IGNORE__" if i == j else data[i][j] for j in range(len(data))] for i in range(len(data))]
        if ignore_below_diagonal:
            data = [["__IGNORE__" if i > j else data[i][j] for j in range(len(data))] for i in range(len(data))]
        if ignore_above_diagonal:
            data = [["__IGNORE__" if j > i else data[i][j] for j in range(len(data))] for i in range(len(data))]

        return DependencyMatrix(data)

    @staticmethod
    def from_tensor(tokens, label_tensor, label_vocab):
        """Create a DependencyMatrix from a tensor containing label indices using the specified label vocabulary.

        Args:
            tokens: The tokens of the sentence associated with the dependency label indices in the tensor.
            label_tensor: The tensor to read label indices from. Should be 1-dimensional and have at least
              `len(tokens)**2` entries.
            label_vocab: Label vocabulary to translate label indices into actual dependency labels.

        Returns:
            A DependencyMatrix object containing the dependencies read from the label index tensor.
        """
        label_index_matrix = LabelIndexMatrix.from_tensor(label_tensor, len(tokens))
        dependencies = list()
        for orig_row in label_index_matrix:
            new_row = list()
            for orig_cell in orig_row:
                new_cell = label_vocab.ix2token(orig_cell)
                new_row.append(new_cell)
            dependencies.append(new_row)

        return DependencyMatrix(dependencies)

    @staticmethod
    def get_annotation_counts(gold, predicted):
        """Compare a system-created DependencyMatrix with a corresponding gold-standard DependencyMatrix.
        For each dependency label type, return the counts for

          * how often this label occurred in the gold standard ("gold")
          * how often the system predicted this label ("predicted")
          * how often the gold label and predicted label were identical ("correct")

        The above counts will also be calculated for an artificial label ("TOTAL") that represents the sum of the
        counts over all label types.

        Args:
            gold: A DependencyMatrix containing gold dependencies.
            predicted: A DependencyMatrix containing predicted dependencies.

        Returns:
            A nested dictionary (label -> "predicted"/"gold"/"correct") of counts.
        """
        assert len(predicted) == len(gold)

        counts = defaultdict(lambda: {"predicted": 0, "gold": 0, "correct": 0})

        for i in range(len(predicted)):
            for j in range(len(predicted)):
                predicted_label = predicted[i][j]
                gold_label = gold[i][j]

                if gold_label == "__IGNORE__":
                    continue

                if gold_label != "[null]":
                    counts[gold_label]["gold"] += 1
                    counts["TOTAL"]["gold"] += 1
                if predicted_label != "[null]":
                    counts[predicted_label]["predicted"] += 1
                    counts["TOTAL"]["predicted"] += 1
                    if predicted_label == gold_label:
                        counts[predicted_label]["correct"] += 1
                        counts["TOTAL"]["correct"] += 1

        return counts


def heads(dependencies, token_ix):
    """For a given token in a DependencyMatrix (specified via its index), generate all of its dependency heads, together
    with the relations by which they are attached.

    The order is going outwards from the specified token, first to the left, then to the right.

    Args:
        dependencies: The DependencyMatrix object to get dependencies from.
        token_ix: Index of the token whose dependency heads will be generated.

    Yields:
        The heads of the specified token, given as tuples `(ix, deprel)`.
    """
    for i in chain(range(token_ix - 1, 0, -1), range(token_ix + 1, len(dependencies))):  # Iterate to the left from word, then to the right
        deprel = dependencies[i][token_ix]
        if deprel != "[null]":
            yield i, deprel


def dependents(dependencies, token_ix):
    """For a given token in a DependencyMatrix (specified via its index), generate all of its dependents, together
    with the relations by which they are attached.

    The order is going outwards from the specified token, first to the left, then to the right.

    Args:
        dependencies: The DependencyMatrix object to get dependencies from.
        token_ix: Index of the token whose dependents will be generated.

    Yields:
        The dependents of the specified token, given as tuples `(ix, deprel)`.
    """
    for j in chain(range(token_ix - 1, 0, -1), range(token_ix + 1, len(dependencies))):
        deprel = dependencies[token_ix][j]
        if deprel != "[null]":
            yield j, deprel
