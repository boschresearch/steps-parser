#  Copyright (c) 2020 Robert Bosch GmbH
#  All rights reserved.
#
#  This source code is licensed under the AGPL v3 license found in the
#  LICENSE file in the root directory of this source tree.
#
#  Author: Stefan Grünewald

"""Script to parse corpora in CoNLL format. Evaluation against the input corpus may optionally be performed after
parsing.
"""

import argparse
import io

from pathlib import Path
from collections import defaultdict

from init_config import ConfigParser
from data_handling.custom_conll_dataset import CustomCoNLLDataset

from util.conll18_ud_eval import evaluate as evaluate_basic, load_conllu as load_conllu_basic
from util.iwpt20_xud_eval import evaluate as evaluate_enhanced, load_conllu as load_conllu_enhanced

BASIC_UD_EVAL_METRICS = ["Lemmas", "UPOS", "XPOS", "UFeats", "UAS", "LAS", "CLAS", "MLAS", "BLEX"]
ENHANCED_UD_EVAL_METRICS = ["Lemmas", "UPOS", "XPOS", "UFeats", "EULAS", "ELAS"]
DEFAULT_TREEBANK_TYPE = defaultdict(lambda: 0)


def parse_corpus(config, corpus_file, output, parser=None, keep_columns=None):
    """Parse each sentence of a CoNLL-format input corpus, writing to the specified output file/stream. Can pass either
    a config (in which the parser to be evaluated will be initialized from this config) or a MultiParser object
    directly.

    Args:
        config: Config to initialize model from. (Will be ignored if parser argument is provided.)
        corpus_file: Path to CoNLL corpus to parse from.
        output: Output file/stream to write to.
        parser: If provided, this parser will be used for parsing and config will be ignored. Default: None.
        keep_columns: If provided (as iterable of ints), these CoNLL annotation columns will be copied over from the
          input corpus.
    """
    if parser is None:
        model = config.init_model()
        trainer = config.init_trainer(model, None, None)  # Inelegant, but need to do this because trainer handles checkpoint loading
        parser = trainer.parser

    annotation_layers = config["data_loaders"]["args"]["annotation_layers"]
    if keep_columns is not None:
        for col in keep_columns:
            annotation_layers[col] = {"type": "TagSequence", "source_column": col}
    column_mapping = {annotation_id: annotation_layer["source_column"] for annotation_id, annotation_layer in annotation_layers.items()}

    dataset = CustomCoNLLDataset.from_corpus_file(corpus_file, annotation_layers)
    for sentence in dataset:
        parsed_sentence = parser.parse(sentence)
        for col in keep_columns or []:  # Copy over columns to keep from input corpus
            parsed_sentence.annotation_data[col] = sentence[col]
        print(parsed_sentence.to_conll(column_mapping), file=output)


def get_config_modification(args, lstm=False):
    """Modify config for parsing/evaluation purposes. Specifically, this entails loading the best checkpoint and not
    redundantly saving the model again after parsing.
    """
    modification = dict()

    modification["saving"] = False  # Do not save config

    # Overwrite vocab file paths with saved vocab files in model config directory
    model_dir = Path(args.model_dir)
    for vocab_path in model_dir.glob("*.vocab"):
        outp_id = vocab_path.stem
        modification[f"model.args.outputs.{outp_id}.args.vocab.args.vocab_filename"] = str(vocab_path)

    # Overwrite transformer model configuration file with stored config; do not load weights
    if lstm:
        modification["model.args.embeddings_processor.args.embeddings_wrapper.args.model_path"] = model_dir / "transformer.json"
        modification["model.args.embeddings_processor.args.embeddings_wrapper.args.tokenizer_path"] = model_dir / "tokenizer"
        modification["model.args.embeddings_processor.args.embeddings_wrapper.args.config_only"] = True
    else:
        modification["model.args.embeddings_processor.args.model_path"] = model_dir / "transformer.json"
        modification["model.args.embeddings_processor.args.tokenizer_path"] = model_dir / "tokenizer"
        modification["model.args.embeddings_processor.args.config_only"] = True

    return modification


def create_output(output_filename):
    """Create the output that parsed sentences are written to. If output_filename is the empty string, output will be
    written to a StringIO instead of a file on disk.
    """
    if not output_filename:
        return io.StringIO()
    else:
        return open(output_filename, "w")


def reset_file(output_file, output_filename):
    """Reset output in read mode. (For use in post-parsing evaluation.)"""
    if not output_filename:
        assert isinstance(output_file, io.StringIO)
        output_file.seek(0)
        return output_file
    else:
        assert not isinstance(output_file, io.StringIO)
        output_file.close()
        output_file = open(output_filename, "r")
        return output_file


def run_evaluation(corpus_file, system_file, mode):
    if mode == "basic":
        gold_ud = load_conllu_basic(corpus_file)
        system_ud = load_conllu_basic(system_file)
        eval_results = evaluate_basic(gold_ud, system_ud)
        print_eval_results(eval_results, BASIC_UD_EVAL_METRICS)
    elif mode == "enhanced":
        gold_ud = load_conllu_enhanced(corpus_file, DEFAULT_TREEBANK_TYPE)
        system_ud = load_conllu_enhanced(system_file, DEFAULT_TREEBANK_TYPE)
        eval_results = evaluate_enhanced(gold_ud, system_ud)
        print_eval_results(eval_results, ENHANCED_UD_EVAL_METRICS)
    else:
        raise Exception(f"Unknown evaluation mode {mode}.")

    return eval_results


def print_eval_results(eval_results, metrics):
    print("Metric     | Precision |    Recall |  F1 Score")
    print("-----------+-----------+-----------+----------")
    for metric in metrics:
        p, r, f = eval_results[metric].precision, eval_results[metric].recall, eval_results[metric].f1
        print(f"{metric:11}|{100*p:10.2f} |{100*r:10.2f} |{100*f:10.2f}")


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Graph-based UD parser (CoNLL corpus parsing mode)')

    # Required arguments
    argparser.add_argument('model_dir', type=str, help='path to model directory (required)')
    argparser.add_argument('corpus_filename', type=str, help='path to corpus file (required).')

    # Optional arguments
    argparser.add_argument('-o', '--output-filename', type=str, default="", help='output filename. If none is provided,'
                                                                                 'output will not be saved to disk.')
    argparser.add_argument('-e', '--eval', type=str, default="none", help='Evaluation type (basic/enhanced/none).'
                                                                          'Default: none')
    argparser.add_argument('-k', '--keep-columns', nargs='+', type=int, help='Indices of columns to retain from input'
                                                                             'corpus')
    argparser.add_argument('--lstm', action='store_true', help='Use this flag if model has an LSTM')

    args = argparser.parse_args()
    config = ConfigParser.from_args(args, modification=get_config_modification(args, lstm=args.lstm))
    output_file = create_output(args.output_filename)

    with open(args.corpus_filename, "r") as corpus_file:
        parse_corpus(config, corpus_file, output_file, keep_columns=args.keep_columns)

    # Run evaluation
    if args.eval.lower() not in {"", "none"}:
        output_file = reset_file(output_file, args.output_filename)
        with open(args.corpus_filename, "r") as corpus_file:
            run_evaluation(corpus_file, output_file, args.eval.lower())

    output_file.close()
