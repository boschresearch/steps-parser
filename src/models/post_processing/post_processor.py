#  Copyright (c) 2020 Robert Bosch GmbH
#  All rights reserved.
#
#  This source code is licensed under the AGPL v3 license found in the
#  LICENSE file in the root directory of this source tree.
#
#  Author: Stefan Grünewald

from abc import ABC, abstractmethod


class PostProcessor(ABC):
    """Base class for sentence post-processing."""
    def __init__(self, annotation_ids, vocabs):
        """
        Args:
            annotation_ids: List of annotation IDs this post-processor operates on.
            vocabs: Dictionary mapping annotation IDs to label vocabularies.
        """
        super().__init__()
        self.annotation_ids = annotation_ids
        self.vocabs = vocabs

    @abstractmethod
    def post_process(self, sentence, logits):
        """
        Args:
            sentence: AnnotatedSentence object to perform post-processing on.
            logits: Dictionary mapping annotation IDs to logits tensors returned by the underlying system.
        """
        pass
