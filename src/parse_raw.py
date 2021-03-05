"""Script to parse raw text corpora. Evaluation against a reference corpus may optionally be performed after parsing."""

#  Copyright (c) 2020 Robert Bosch GmbH
#  All rights reserved.
#
#  This source code is licensed under the AGPL v3 license found in the
#  LICENSE file in the root directory of this source tree.
#
#  Author: Stefan Grünewald

import argparse
import stanza

from io import StringIO
from stanza.utils.conll import CoNLL

from init_config import ConfigParser
from parse_corpus import parse_corpus, get_config_modification, create_output, reset_file, run_evaluation


def preprocess_to_stream(corpus_filename, lang):
    """Pre-process (tokenize, segment) the specified raw text corpus using the Stanford's Stanza library.

    Args:
        corpus_filename: Filename of the raw text corpus to pre-process.
        lang: Language of Stanza model to use for pre-processing.

    Returns:
        A stream containing the pre-processed text in CoNLL format.
    """
    stanza_pipeline = stanza.Pipeline(lang=lang, processors='tokenize,mwt', use_gpu=False)
    with open(corpus_filename, "r") as corpus_file:
        doc = stanza_pipeline(corpus_file.read())

    conll = CoNLL.convert_dict(doc.to_dict())
    conll_stream = StringIO()
    for sent in conll:
        for token in sent:
            print("\t".join(token), file=conll_stream)
        print(file=conll_stream)

    conll_stream.seek(0)

    return conll_stream


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Graph-based UD parser (raw text parsing mode)')

    # Required arguments
    argparser.add_argument('model_dir', type=str, help='path to model directory (required)')
    argparser.add_argument('language', type=str, help='language code (for tokenization and segmentation).')
    argparser.add_argument('corpus_filename', type=str, help='path to corpus file (required).')

    # Optional arguments
    argparser.add_argument('-o', '--output-filename', type=str, default="", help='output filename. If none is provided,'
                                                                                 'output will not be saved to disk.')
    argparser.add_argument('-r', '--reference-corpus', type=str, default="", help='Reference corpus to evaluate against.')
    argparser.add_argument('-e', '--eval', type=str, default="basic", help='Evaluation type (basic/enhanced).'
                                                                           'Default: basic')
    args = argparser.parse_args()
    config = ConfigParser.from_args(args, modification=get_config_modification(args))
    conll_stream = preprocess_to_stream(args.corpus_filename, args.language)
    output_file = create_output(args.output_filename)

    parse_corpus(config, conll_stream, output_file)

    # Run evaluation
    if args.reference_corpus != "":
        output_file = reset_file(output_file, args.output_filename)
        with open(args.reference_corpus, "r") as reference_corpus:
            run_evaluation(reference_corpus, output_file, args.eval.lower())

    conll_stream.close()
