#  Copyright (c) 2020 Robert Bosch GmbH
#  All rights reserved.
#
#  This source code is licensed under the AGPL v3 license found in the
#  LICENSE file in the root directory of this source tree.
#
#  Author: Stefan Grünewald

from torch import nn

from torch.nn import LSTM
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTMProcessor(nn.Module):
    """Module for generating distributed token representations by using an LSTM on top of a transformer-based language
    model.
    """
    def __init__(self, embeddings_wrapper, output_ids, hidden_size, num_shared_layers, num_taskspecific_layers, dropout,
                 shared_embeddings=None):
        """
        Args:
            embeddings_wrapper: Wrapper instance to use for the transformer LM.
            output_ids: Output IDs to generate word vectors for.
            hidden_size: Size of hidden states of the LSTM layers.
            num_shared_layers: Number of LSTM layers shared for all output tasks.
            num_taskspecific_layers: Number of LSTM layers dedicated to each individual output task.
            dropout: Dropout probability to apply to the LSTM layers.
            shared_embeddings: If specified (as list of lists of output IDs), the specified groups of outputs will
              share the same task-specific LSTM layers (and thus embeddings). Default: None.
        """
        super(LSTMProcessor, self).__init__()
        self.embeddings_wrapper = embeddings_wrapper
        self.output_ids = output_ids

        self.shared_lstm, self.task_lstm = self._init_lstms(output_ids, hidden_size, num_shared_layers,
                                                            num_taskspecific_layers, dropout,
                                                            shared_embeddings=shared_embeddings)
        self.embedding_dim = 2*hidden_size

    def _init_lstms(self, output_ids, hidden_size, num_shared_layers, num_taskspecific_layers, dropout,
                    shared_embeddings=None):
        assert num_shared_layers + num_taskspecific_layers > 0, "There must be at least one LSTM layer"

        if num_shared_layers > 0:
            shared_lstm = LSTM(input_size=self.embeddings_wrapper.embedding_dim, hidden_size=hidden_size,
                               num_layers=num_shared_layers, batch_first=True, dropout=dropout, bidirectional=True)
        else:
            shared_lstm = None

        taskspecific_input_size = 2*hidden_size if num_shared_layers > 0 else self.embeddings_wrapper.embedding_dim

        if num_taskspecific_layers > 0:
            if shared_embeddings is None:
                task_lstm = nn.ModuleDict({outp_id: LSTM(input_size=taskspecific_input_size, hidden_size=hidden_size,
                                                         num_layers=num_taskspecific_layers, dropout=dropout,
                                                         batch_first=True, bidirectional=True) for outp_id in self.output_ids})
            else:
                task_lstm = nn.ModuleDict()
                for group in shared_embeddings:
                    curr_lstm = LSTM(input_size=taskspecific_input_size, hidden_size=hidden_size,
                                     num_layers=num_taskspecific_layers, dropout=dropout,
                                     batch_first=True, bidirectional=True)
                    for outp_id in group:
                        task_lstm[outp_id] = curr_lstm
            for outp_id in self.output_ids:
                if outp_id not in task_lstm:
                    # Add LSTMs for all outputs that don't have one yet
                    task_lstm[outp_id] = LSTM(input_size=taskspecific_input_size, hidden_size=hidden_size,
                                              num_layers=num_taskspecific_layers, dropout=dropout,
                                              batch_first=True, bidirectional=True)
        else:
            task_lstm = None

        return shared_lstm, task_lstm

    def forward(self, input_sentences):
        """Transform a bunch of input sentences (list of lists of tokens) into a batch (tensor) of LSTM-based word
        vectors.

        Args:
            input_sentences: The input sentences to transform into word vectors (list of lists of tokens).

        Returns: A tuple consisting of (a) a dictionary with the word vectors for each output/annotation ID
          (shape: batch_size * max_seq_len * embedding_dim); (b) a tensor containing the length (number of tokens)
          of each sentence (shape: batch_size).
        """
        embeddings, true_seq_lengths = self.embeddings_wrapper(input_sentences)
        embeddings = embeddings["lstm"]

        packed_embeddings = pack_padded_sequence(embeddings, true_seq_lengths, batch_first=True, enforce_sorted=False)

        if self.shared_lstm is not None:
            shared_lstm_output, _ = self.shared_lstm(packed_embeddings)
        else:
            shared_lstm_output = packed_embeddings

        if self.task_lstm is not None:
            taskspecific_lstm_outputs = dict()
            for outp_id in self.output_ids:
                curr_lstm_output, _ = self.task_lstm[outp_id](shared_lstm_output)
                taskspecific_lstm_outputs[outp_id] = pad_packed_sequence(curr_lstm_output, batch_first=True)[0]
        else:
            taskspecific_lstm_outputs = {outp_id: pad_packed_sequence(shared_lstm_output, batch_first=True)[0] for outp_id in self.output_ids}

        return taskspecific_lstm_outputs, true_seq_lengths

    def save_transformer_config(self, model_directory):
        """Save this module's transformer configuration to the specified directory."""
        self.embeddings_wrapper.save_transformer_config(model_directory)
