#  Copyright (c) 2020 Robert Bosch GmbH
#  All rights reserved.
#
#  This source code is licensed under the AGPL v3 license found in the
#  LICENSE file in the root directory of this source tree.
#
#  Author: Stefan Grünewald

from collections import defaultdict


class LossScaler:
    """Class for scaling the loss of outputs according to some scheme.
    For each output, the scaling scheme is a function mapping the epoch number to a scaling factor. By default, this
    is a function that always returns 1.
    """

    def __init__(self, scaling_fn):
        """
        Args:
            scaling_fn: A dictionary mapping output IDs to strings evaluating to lambda expressions that map an epoch
              number to a scaling factor.
              Example:
              `scaling_fn["upos"] == "0 if epoch <= 10 else 0.5 * (epoch-10) / 10 if 10 < epoch < 20 else 0.5"`.
        """
        self.scaling_fn = defaultdict(lambda: lambda epoch: 1.0)
        self.scaling_fn.update({outp_id: eval(fn) for outp_id, fn in scaling_fn.items()})

    def get_loss_scaling_factor(self, outp_id, epoch):
        """Retrieve the loss scaling factor for the given output ID in the given epoch."""
        return self.scaling_fn[outp_id](epoch)
