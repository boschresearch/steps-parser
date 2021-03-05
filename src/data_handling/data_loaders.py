#  Copyright (c) 2020 Robert Bosch GmbH
#  All rights reserved.
#
#  This source code is licensed under the AGPL v3 license found in the
#  LICENSE file in the root directory of this source tree.
#
#  Author: Stefan Grünewald

from torch.utils.data import DataLoader

from math import inf

from data_handling.custom_conll_dataset import CustomCoNLLDataset
from data_handling.bucket_sampler import BucketBatchSampler
from data_handling.annotated_sentence import AnnotatedSentence


class StandardCONLLLoader(DataLoader):
    """Pytorch DataLoader class for loading batches of sentences from a corpus file in CONLL-U format."""
    def __init__(self, corpus_path, output_vocabs, annotation_layers, keep_traces=False,
                 batch_size=10, max_sent_len=inf, shuffle=True, num_workers=1):
        """
        Args:
            corpus_path: Path of the corpus file to load sentence batches from.
            output_vocabs: Dictionary mapping output IDs to label vocabularies.
            annotation_layers: Dictionary mapping annotation IDs to annotation type and CoNLL column to read data from.
            keep_traces: Whether to keep empty nodes as tokens (used in enhanced UD; default: False).
            batch_size: Size of sentence batches yielded by this data loader (default: 10).
            max_sent_len: The maximum length of any given sentence. Sentences with a greater length are ignored
              (default: inf).
            shuffle: Whether to shuffle the instances before creating batches.
            num_workers: Wow many subprocesses to use for data loading.
        """
        assert output_vocabs.keys() == annotation_layers.keys()  # Annotation layers and vocabularies must match

        self.conll_dataset = CustomCoNLLDataset.from_corpus_file(corpus_path, annotation_layers,
                                                                 max_sent_len=max_sent_len, keep_traces=keep_traces)
        self.output_vocabs = output_vocabs

        super().__init__(self.conll_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                         collate_fn=lambda x: _batchify(x, self.output_vocabs))


class BucketedCONLLLoader(DataLoader):
    """Pytorch DataLoader class for loading batches of sentences from a corpus file in CONLL-U format while performing
    bucketing. For a more detailed description of the bucketing procedure, see documentation of the BucketBatchSampler
    class."""
    def __init__(self, corpus_path, output_vocabs, annotation_layers, batch_size, bucket_size,
                 size_fn=lambda x: len(x) ** 2, max_sent_len=inf, max_tokens_per_batch=None, keep_traces=False, num_workers=1):
        """
        Args:
            corpus_path: Path of the corpus file to load sentence batches from.
            output_vocabs: Dictionary mapping output IDs to label vocabularies.
            annotation_layers: Dictionary mapping annotation IDs to annotation type and CoNLL column to read data from.
            batch_size: Size of sentence batches yielded by this data loader (default: 10).
            bucket_size: Size of the buckets to distribute instances into. Smaller buckets result in more heterogeneous
              instance sizes within batches.
            size_fn: Function to compute the size of an instance with (default: `len(inst)**2`)
            max_sent_len: The maximum length of any given sentence. Sentences with a greater length are ignored
              (default: inf).
            max_tokens_per_batch: The maximum cumulative size that a batch is allowed to have.
            keep_traces: Whether to keep empty nodes as tokens (used in enhanced UD; default: False).
            num_workers: Wow many subprocesses to use for data loading.
        """

        assert output_vocabs.keys() == annotation_layers.keys()  # Annotation layers and vocabularies must match

        self.conll_dataset = CustomCoNLLDataset.from_corpus_file(corpus_path, annotation_layers,
                                                                 max_sent_len=max_sent_len, keep_traces=keep_traces)
        self.bucket_sampler = BucketBatchSampler(self.conll_dataset, batch_size, bucket_size,
                                                 size_fn=size_fn, max_cumsize_per_batch=max_tokens_per_batch)
        self.output_vocabs = output_vocabs

        super().__init__(self.conll_dataset, batch_sampler=self.bucket_sampler, num_workers=num_workers,
                         collate_fn=lambda x: _batchify(x, self.output_vocabs))


def _batchify(sentences, output_vocabs):
    """Helper function to create model input / gold output from a bunch of DependencyAnnotatedSentences.

    Output: A tuple whose first element is the list of AnnotatedSentences and whose second element is a dictionary
    containing the batched target tensor for each annotation ID.
    """
    return sentences, AnnotatedSentence.get_tensorized_annotations(sentences, output_vocabs)
