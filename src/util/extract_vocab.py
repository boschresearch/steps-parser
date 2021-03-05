#  Copyright (c) 2020 Robert Bosch GmbH
#  All rights reserved.
#
#  This source code is licensed under the AGPL v3 license found in the
#  LICENSE file in the root directory of this source tree.
#
#  Author: Stefan Grünewald
"""Script to extract a label vocabulary from one or more CoNLL-formatted corpus file(s)."""

import argparse

from data_handling.custom_conll_dataset import CustomCoNLLDataset

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description='Extract a label vocabulary from one or more CoNLL-formatted corpus file(s)')
    argparser.add_argument('type', type=str, help='type of annotation (tags vs. dependencies) (required)')
    argparser.add_argument('column', type=int, help='column to read the labels from (indexing starts at 0) (required)')
    argparser.add_argument('files', nargs='+', type=str, help='CoNLL corpus files')

    args = argparser.parse_args()
    if args.type.lower() in {"dep", "deps", "dependency", "dependencies", "dependencymatrix"}:
        args.type = "DependencyMatrix"
    elif args.type.lower() in {"tag", "tags", "tagsequence"}:
        args.type = "TagSequence"
    else:
        raise Exception("Annotation type must be \"dependencies\" or \"tags\"!")

    annotation_layer = {"label_extraction": {"type": args.type, "source_column": args.column}}

    conll_datasets = [CustomCoNLLDataset.from_corpus_file(corpus_filename, annotation_layer) for corpus_filename in args.files]
    vocab = CustomCoNLLDataset.extract_label_vocab(*conll_datasets, annotation_id="label_extraction")

    print(vocab)
