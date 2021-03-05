#  Copyright (c) 2020 Robert Bosch GmbH
#  All rights reserved.
#
#  This source code is licensed under the AGPL v3 license found in the
#  LICENSE file in the root directory of this source tree.
#
#  Author: Stefan Grünewald

from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import LambdaLR

import math


class SqrtSchedule:
    """Wrapper for Noam LR schedule."""
    def __init__(self, warmup_steps):
        """
        Args:
            warmup_steps: Number of steps for linear warmup.
        """
        self.warmup_steps = warmup_steps
        self.sqrt_warmup_steps = warmup_steps**0.5
        self.inv_warmup_steps = warmup_steps**(-1.5)

    def __call__(self, step):
        if step == 0:
            return 0
        else:
            return self.sqrt_warmup_steps * min(step**(-0.5), step*self.inv_warmup_steps)


class WarmRestartSchedule:
    """Wrapper for cosine annealing with warmup and warm restarts."""
    def __init__(self, warmup_steps, T_0, T_mult=1, eta_min=0):
        """
        Args:
            warmup_steps: Number of linear warmup steps.
            T_0: Initial cycle length.
            T_mult: Cycle length growth factor.
            eta_min: Minimum learning rate scaling factor.
        """
        self.warmup_steps = warmup_steps
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min

    def __call__(self, step):
        if step <= self.warmup_steps:
            return step / self.warmup_steps
        else:
            step = step - self.warmup_steps

            # Determine current cycle
            if step >= self.T_0:
                if self.T_mult == 1:
                    T_cur = step % self.T_0
                    T_i = self.T_0
                else:
                    n = int(math.log((step / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    T_cur = step - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    T_i = self.T_0 * self.T_mult ** (n)
            else:
                T_i = self.T_0
                T_cur = step

        return self.eta_min + (1 - self.eta_min) * (1 + math.cos(math.pi * T_cur / T_i)) / 2


class CustomLRScheduler(_LRScheduler):
    """This LR scheduler implements the LR schedule as described by Kondratyuk & Straka (2019).
    It assumes that there are two parameter groups: One "default" group (containing parameters like the biaffine
    classifier weights) and one "special" group (containing e.g. BERT weights) which we would like to keep frozen
    for some period of time. The two different groups can also have different base learning rates.
    """
    def __init__(self, optimizer, frozen_steps, warmup_steps, factor):
        """
        Args:
            optimizer: The optimizer for which to schedule the learning rate.
            frozen_steps: Number of steps during which the parameters in the "special" group are frozen (lr=0) and
              the parameters in the "default" group are trained at base lr.
            warmup_steps: Number of steps during which the learning rate increases linearly.
            factor: Factor by which the base learning rate is multiplied.
        """
        self.optimizer = optimizer
        self.frozen_steps = frozen_steps
        self.warmup_steps = warmup_steps
        self.factor = factor

        # We assume there is one "default" param group (index 0) and one "special" param group (index 1)
        assert len(self.optimizer.param_groups) == 2
        self.base_lr_default = self.optimizer.param_groups[0]["lr"]
        self.base_lr_special = self.optimizer.param_groups[1]["lr"]

        super(CustomLRScheduler, self).__init__(optimizer)

    def get_lr(self):
        """Return the current learning rates for the default and special parameter groups."""
        # NOTE: Since the base class assumes that we are only updating LR after each epoch, the nomenclature is
        # confusing here; self.last_epoch actually represents the number of scheduler steps we have already done.
        if self.last_epoch < self.frozen_steps:
            lr_default = self.base_lr_default
            lr_special = 0.0
        else:
            steps = max(self.last_epoch - self.frozen_steps, 1)
            scale = self.factor * min(steps**(-0.5), steps * self.warmup_steps**(-1.5))
            lr_default = self.base_lr_default * scale
            lr_special = self.base_lr_special * scale

        if self.last_epoch % 50 == 0:
            print("Default LR is now {}".format(lr_default))
            print("Special LR is now {}".format(lr_special))

        return lr_default, lr_special
