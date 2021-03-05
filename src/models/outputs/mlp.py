#  Copyright (c) 2020 Robert Bosch GmbH
#  All rights reserved.
#
#  This source code is licensed under the AGPL v3 license found in the
#  LICENSE file in the root directory of this source tree.
#
#  Author: Stefan Grünewald

import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Simple multilayer perceptron."""
    def __init__(self, n_in, n_hidden, n_out, hidden_activation=F.relu, input_dropout=0.0, hidden_dropout=0.0):
        """
        Args:
            n_in: Dimensions of input.
            n_hidden: Dimensions of hidden layer.
            n_out: Dimensions of output.
            hidden_activation: Activation function of hidden layer. Default: ReLU.
            input_dropout: Dropout ratio to apply to input. Default: 0.0.
            hidden_dropout: Dropout ratio to apply to hidden layer. Default: 0.0.
        """
        super(MLP, self).__init__()

        if n_hidden is None or n_hidden <= 0:
            self.has_hidden_units = False
            self.in_to_out = nn.Linear(n_in, n_out)
            self.in_to_hidden = None
        else:
            self.has_hidden_units = True
            self.in_to_hidden = nn.Linear(n_in, n_hidden)
            self.hidden_to_out = nn.Linear(n_hidden, n_out)
            self.hidden_activation = hidden_activation

        self.input_dropout = nn.Dropout(p=input_dropout)
        self.hidden_dropout = nn.Dropout(p=hidden_dropout)

    def forward(self, x):
        """Apply the MLP to input x (expected shape: batch_size * max_seq_len * n_in)."""
        x = self.input_dropout(x)

        if self.has_hidden_units:
            x = self.in_to_hidden(x)
            x = self.hidden_activation(x)
            x = self.hidden_dropout(x)
            x = self.hidden_to_out(x)
        else:
            x = self.in_to_out(x)

        return x
