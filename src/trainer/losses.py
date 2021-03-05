#  Copyright (c) 2020 Robert Bosch GmbH
#  All rights reserved.
#
#  This source code is licensed under the AGPL v3 license found in the
#  LICENSE file in the root directory of this source tree.
#
#  Author: Stefan Grünewald

import torch

from torch.nn.modules.loss import _Loss
from torch.nn import CrossEntropyLoss
from torch.nn import BCEWithLogitsLoss


class BCEWithLogitsLossWithIgnore(_Loss):
    """Custom BCEWithLogitsLoss that ignores indices where target tensor is negative.
    Useful when working with padding.

    Additionally, makes BCE loss work with integer targets.
    """

    def __init__(self):
        super(BCEWithLogitsLossWithIgnore, self).__init__()
        self.bce_with_logits_loss = BCEWithLogitsLoss()

    def forward(self, input, target):
        assert input.shape == target.shape

        input_non_ignored = input[target >= 0]
        target_non_ignored = target[target >= 0]

        assert input_non_ignored.shape == target_non_ignored.shape

        if target_non_ignored.dtype != torch.float32:
            target_non_ignored = target_non_ignored.float()

        return self.bce_with_logits_loss(input_non_ignored, target_non_ignored)
