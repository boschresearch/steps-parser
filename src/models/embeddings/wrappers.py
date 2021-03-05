#  Copyright (c) 2020 Robert Bosch GmbH
#  All rights reserved.
#
#  This source code is licensed under the AGPL v3 license found in the
#  LICENSE file in the root directory of this source tree.
#
#  Author: Stefan Grünewald

import torch
import random

from torch import nn

from transformers.tokenization_bert import BertTokenizer
from transformers.modeling_bert import BertModel, BertConfig

from transformers.tokenization_roberta import RobertaTokenizer
from transformers.modeling_roberta import RobertaModel, RobertaConfig

from transformers.tokenization_xlm_roberta import XLMRobertaTokenizer
from transformers.modeling_xlm_roberta import XLMRobertaModel, XLMRobertaConfig

from torch.nn import Dropout

from models.embeddings.scalar_mix import ScalarMixWithDropout


class Wrapper(nn.Module):
    """Base class for turning batches of sentences into tensors of (BERT/RoBERTa/...) embeddings.

    An object of this class takes as input a bunches of sentences (represented as a lists of lists of tokens) and
    returns, for each specified output ID, tensors (shape: batch_size * max_sent_len * embedding_dim) of token
    embeddings. The embeddings for the different outputs are generated using the same underlying transformer model, but
    by default use different scalar mixtures of the internal transformer layers to generate final embeddings.
    """
    def __init__(self, model_class, tokenizer_class, config_class, model_path, output_ids, tokenizer_path=None,
                 config_only=False, fine_tune=True, shared_embeddings=None, hidden_dropout=0.2, attn_dropout=0.2,
                 output_dropout=0.5, scalar_mix_layer_dropout=0.1, token_mask_prob=0.2):
        """
        Args:
            model_class: Class of transformer model to use for token embeddings.
            tokenizer_class: Class of tokenizer to use for tokenization.
            config_class: Class of transformer config.
            model_path: Path to load transformer model from.
            output_ids: Output IDs to generate embeddings for.
            tokenizer_path: Path to load tokenizer from (default: None; specify when using config_only option).
            config_only: If True, only load model config, not weights (default: False).
            fine_tune: Whether to fine-tune the transformer language model. If False, weights of the transformer model
              will not be trained. Default: True.
            shared_embeddings: If specified (as list of lists of output IDs), the specified groups of outputs will
              share the same scalar mixture (and thus embeddings). Default: None.
            hidden_dropout: Dropout ratio for hidden layers of the transformer model.
            attn_dropout: Dropout ratio for the attention probabilities.
            output_dropout: Dropout ratio for embeddings output.
            scalar_mix_layer_dropout: Dropout ratio for transformer layers.
            token_mask_prob: Probability of replacing input tokens with mask token.
        """
        super(Wrapper, self).__init__()
        self.output_ids = output_ids

        self.model, self.tokenizer = self._init_model(model_class, tokenizer_class, config_class,
                                                      model_path, tokenizer_path, config_only=config_only,
                                                      hidden_dropout=hidden_dropout, attn_dropout=attn_dropout)

        self.token_mask_prob = token_mask_prob
        self.embedding_dim = self.model.config.hidden_size
        self.fine_tune = fine_tune

        self.scalar_mix = self._init_scalar_mix(shared_embeddings=shared_embeddings,
                                                layer_dropout=scalar_mix_layer_dropout)

        self.root_embedding = nn.Parameter(torch.randn(self.embedding_dim), requires_grad=True)

        if output_dropout > 0.0:
            self.output_dropout = Dropout(p=output_dropout)

    def _init_model(self, model_class, tokenizer_class, config_class, model_path, tokenizer_path, config_only=False,
                    hidden_dropout=0.2, attn_dropout=0.2):
        """Initilaize the transformer language model."""
        if config_only:
            model = model_class(config_class.from_json_file(str(model_path)))
            tokenizer = tokenizer_class.from_pretrained(str(tokenizer_path))
        else:
            model = model_class.from_pretrained(model_path,
                                                output_hidden_states=True,
                                                hidden_dropout_prob=hidden_dropout,
                                                attention_probs_dropout_prob=attn_dropout)
            tokenizer = tokenizer_class.from_pretrained(model_path)

        return model, tokenizer

    def _init_scalar_mix(self, shared_embeddings=None, layer_dropout=0.1):
        """Initialize the scalar mixture module."""
        num_layers = len(self.model.encoder.layer) + 1  # Add 1 because of input embeddings

        if shared_embeddings is None:
            scalar_mix = nn.ModuleDict(
                {output_id: ScalarMixWithDropout(mixture_size=num_layers, layer_dropout=layer_dropout) for
                 output_id in self.output_ids})
        else:
            scalar_mix = nn.ModuleDict()
            for group in shared_embeddings:
                curr_scalarmix = ScalarMixWithDropout(mixture_size=num_layers, layer_dropout=layer_dropout)
                for outp_id in group:
                    scalar_mix[outp_id] = curr_scalarmix
            for outp_id in self.output_ids:
                if outp_id not in scalar_mix:
                    # Add scalar mixes for all outputs that don't have one yet
                    scalar_mix[outp_id] = ScalarMixWithDropout(mixture_size=num_layers, layer_dropout=layer_dropout)

        return scalar_mix

    def forward(self, input_sentences):
        """Transform a bunch of input sentences (list of lists of tokens) into a batch (tensor) of
        BERT/RoBERTa/etc. embeddings.

        Args:
            input_sentences: The input sentences to transform into embeddings (list of lists of tokens).

        Returns: A tuple consisting of (a) a dictionary with the embeddings for each output/annotation ID
          (shape: batch_size * max_seq_len * embedding_dim); (b) a tensor containing the length (number of tokens)
          of each sentence (shape: batch_size).
        """
        # Retrieve inputs for BERT model
        model_inputs = self._get_model_inputs(input_sentences)

        # Get embeddings tensors (= a dict containing one tensor for each output)
        raw_embeddings = self._get_raw_embeddings(model_inputs)

        # For each output, extract only the relevant embeddings and add a learned [root] token at the beginning
        processed_embeddings = dict()
        for output_id in self.output_ids:
            processed_embeddings[output_id] = self._process_embeddings(raw_embeddings[output_id], model_inputs["original_token_mask"])

        # Compute true sequence lengths and put into tensor (new method)
        true_seq_lengths = self._compute_true_seq_lengths(input_sentences, device=model_inputs["device"])

        return processed_embeddings, true_seq_lengths

    def _get_model_inputs(self, input_sentences):
        """Take a list of sentences and return tensors for token IDs, attention mask, and original token mask"""
        mask_prob = self.token_mask_prob if self.training else 0.0
        input_sequences = [BertInputSequence(sent, self.tokenizer, token_mask_prob=mask_prob) for sent in
                           input_sentences]
        max_input_seq_len = max(len(input_sequence) for input_sequence in input_sequences)
        device = next(iter(self.model.parameters())).device  # Ugly :(
        for input_sequence in input_sequences:
            input_sequence.tensorize(device, padded_length=max_input_seq_len)

        # Batch components of input sequences
        input_ids = torch.stack([input_seq.token_ids for input_seq in input_sequences])
        attention_mask = torch.stack([input_seq.attention_mask for input_seq in input_sequences])
        original_token_mask = torch.stack([input_seq.original_token_mask for input_seq in input_sequences])

        assert input_ids.shape[0] == len(input_sentences)
        assert input_ids.shape[1] == max_input_seq_len or input_ids.shape[1] == 512

        return {"input_ids": input_ids, "attention_mask": attention_mask, "original_token_mask": original_token_mask,
                "device": device}

    def _get_raw_embeddings(self, model_inputs):
        """Take tensors for input tokens and run them through underlying BERT-based model, performing the learned scalar
         mixture for each output"""
        raw_embeddings = dict()

        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]

        with torch.set_grad_enabled(self.fine_tune):
            embedding_layers = torch.stack(self.model(input_ids, attention_mask=attention_mask)[2])
            for output_id in self.output_ids:
                if self.output_dropout:
                    embedding_layers = self.output_dropout(embedding_layers)

        for output_id in self.output_ids:
            curr_output = self.scalar_mix[output_id](embedding_layers)
            raw_embeddings[output_id] = curr_output

        return raw_embeddings

    def _process_embeddings(self, raw_embeddings, original_token_mask):
        """Extract only those embeddings which correspond to original tokens in the input sentence, i.e., discard
        all the embeddings which are marked 0 in the original token mask.

        Additionally, add a learned [root] embedding to the beginning of each embedding sequence."""
        batch_size = raw_embeddings.shape[0]
        max_input_seq_len = raw_embeddings.shape[1]

        # ---------------------------------------------------------------------------------------------------------
        # Step 1: Reorder model output so that all embeddings corresponding to original tokens are at the beginning
        # ---------------------------------------------------------------------------------------------------------
        inverted_token_mask = 1 - original_token_mask

        # (The following three steps are needed because torch.argsort is not stable, i.e. we have to explicitly encode
        # the original order)
        multiplied_mask = inverted_token_mask * max_input_seq_len
        token_pos = torch.arange(0, max_input_seq_len, device=multiplied_mask.device).unsqueeze(0).expand((batch_size, max_input_seq_len))
        overall_mask = multiplied_mask + token_pos

        permutation = torch.argsort(overall_mask)
        embeddings_reordered = torch.gather(raw_embeddings, 1, permutation.unsqueeze(-1).expand(raw_embeddings.shape))

        # ---------------------------------------------------------------------------------------------------------
        # Step 2: Cut off the excess embeddings so that the sequence length is reduced to the length of the longest
        # original (i.e. non-BERT) sentence
        # ---------------------------------------------------------------------------------------------------------
        max_orig_seq_len = torch.max(torch.sum(original_token_mask, dim=1))
        embeddings_stripped = embeddings_reordered[:, 0:max_orig_seq_len, :]

        # ---------------------------------------------------------------------------------------------------------
        # Step 3: Add learned [root] embedding to the beginning of each sentence
        # ---------------------------------------------------------------------------------------------------------
        root_embedding_expanded = self.root_embedding.unsqueeze(0).unsqueeze(0).expand((batch_size, 1, self.embedding_dim))
        embeddings_with_root = torch.cat((root_embedding_expanded, embeddings_stripped), dim=1)

        return embeddings_with_root

    def _compute_true_seq_lengths(self, sentences, device=None):
        seq_lengths = torch.tensor([len(sent) for sent in sentences], device=device)
        if self.root_embedding is not None:
            seq_lengths += 1

        return seq_lengths

    def save_transformer_config(self, model_directory):
        """Save this module's transformer configuration to the specified directory."""
        self.model.config.to_json_file(model_directory / "transformer.json")
        self.tokenizer.save_pretrained(model_directory / "tokenizer")

    def parallelize(self, device_ids):
        """Parallelize this module for multi-GPU setup-"""
        self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)


class BertWrapper(Wrapper):
    """Embeddings wrapper class for models based on BERT."""
    def __init__(self, *args, **kwargs):
        super(BertWrapper, self).__init__(BertModel, BertTokenizer, BertConfig, *args, **kwargs)


class RobertaWrapper(Wrapper):
    """Embeddings wrapper class for models based on RoBERTa."""
    def __init__(self, *args, **kwargs):
        super(RobertaWrapper, self).__init__(RobertaModel, RobertaTokenizer, RobertaConfig, *args, **kwargs)


class XLMRobertaWrapper(Wrapper):
    """Embeddings wrapper class for models based on XLM-R."""
    def __init__(self, *args, **kwargs):
        super(XLMRobertaWrapper, self).__init__(XLMRobertaModel, XLMRobertaTokenizer, XLMRobertaConfig, *args, **kwargs)


class BertInputSequence:
    """Class for representing the features of a single, dependency-annotated sentence in tensor
    form, for usage in models based on BERT.

    Example:
    ```
    Input sentence:                Beware      the     jabberwock                 ,    my   son    !
    BERT tokens:          [CLS]    be ##ware   the     ja ##bber   ##wo  ##ck     ,    my   son    ! [SEP]  ([PAD] [PAD] [PAD]  ...)
    BERT token IDs:         101  2022   8059  1996  14855  29325  12155  3600  1010  2026  2365  999   102  (    0     0     0  ...)
    BERT attention mask:      1     1      1     1      1      1      1     1     1     1     1    1     1  (    0     0     0  ...)
    Original token mask:      0     1      0     1      1      0      0     0     1     1     1    1     0  (    0     0     0  ...)
    ```
    """

    def __init__(self, orig_tokens, tokenizer, token_mask_prob=0.0):
        """
        Args:
            orig_tokens: Tokens to convert into a BertInputSequence.
            tokenizer: Tokenizer to use to split original tokens into word pieces.
            token_mask_prob: Probability of replacing an input token with a mask token. All word pieces of a given token
              will be replaced.
        """
        self.tokenizer = tokenizer

        self.tokens = list()
        self.attention_mask = list()
        self.original_token_mask = list()

        assert orig_tokens[0] != "[root]"  # The embedding model will add its own learned root token (see Wrapper class),
                                           # so make sure that it was not passed here

        self.append_special_token(self.tokenizer.cls_token)  # BOS marker

        for orig_token in orig_tokens:
            self.append_regular_token(orig_token, mask_prob=token_mask_prob)

        self.append_special_token(self.tokenizer.sep_token)  # EOS marker

        assert len(orig_tokens) <= len(self.tokens) == len(self.attention_mask) == len(self.original_token_mask)

        # Convert BERT tokens to IDs
        self.token_ids = self.tokenizer.convert_tokens_to_ids(self.tokens)

    def __len__(self):
        return len(self.tokens)

    def append_special_token(self, token):
        """Append a special token (e.g. BOS token, MASK token) to the sequence. The token will receive attention in the
        model, but will not be counted as an original token.
        """
        self.tokens.append(token)
        self.attention_mask.append(1)
        self.original_token_mask.append(0)

    def append_regular_token(self, token, mask_prob=0.0):
        """Append regular token (i.e., a word from the input sentence) to the sequence. The token will be split further
        into word pieces by the tokenizer. All word pieces will receive attention, but only the first word piece will
        be counted as an original token."""
        if isinstance(self.tokenizer, RobertaTokenizer):
            curr_bert_tokens = self.tokenizer.tokenize(token, add_prefix_space=True)
        else:
            curr_bert_tokens = self.tokenizer.tokenize(token)

        if len(curr_bert_tokens) == 0:
            print("WARNING: Replacing non-existent BERT token with UNK")
            curr_bert_tokens = [self.tokenizer.unk_token]

        if mask_prob > 0.0 and random.random() < mask_prob:
            curr_bert_tokens = [self.tokenizer.mask_token] * len(curr_bert_tokens)

        self.tokens += curr_bert_tokens
        self.attention_mask += [1] * len(curr_bert_tokens)
        self.original_token_mask += [1] + [0] * (len(curr_bert_tokens) - 1)

    def pad_to_length(self, padded_length):
        """Pad the sentence to the specified length. This will increase the length of all fields to padded_length by
        adding the padding label/index."""
        assert padded_length >= len(self.tokens)
        padding_length = padded_length - len(self.tokens)

        self.tokens += [self.tokenizer.pad_token] * padding_length
        self.token_ids += [self.tokenizer.pad_token_id] * padding_length
        self.attention_mask += [0] * padding_length
        self.original_token_mask += [0] * padding_length

        assert len(self.tokens) == len(self.token_ids) == len(self.attention_mask) == len(self.original_token_mask)

    def tensorize(self, device, padded_length=None):
        """Convert the numerical fields of this BERT sentence into PyTorch tensors. The sentence may be padded to a
        specified length beforehand."""
        if len(self.token_ids) > 512:
            self._cut_off()

        if padded_length is not None:
            padded_length = min(padded_length, 512)
            self.pad_to_length(padded_length)

        self.token_ids = torch.tensor(self.token_ids, device=device)
        self.attention_mask = torch.tensor(self.attention_mask, device=device)
        self.original_token_mask = torch.tensor(self.original_token_mask, device=device)

    def _cut_off(self):
        """Throw away non-original tokens in order to reduce sequence length."""
        print("WARNING: Cutting off sentence to stay below maximum sequence length")
        print("Number of word pieces:", len(self.tokens))
        assert len(self.tokens) == len(self.token_ids) > 512

        num_orig_tokens = sum(self.original_token_mask)
        print("Number of original tokens:", num_orig_tokens)
        assert num_orig_tokens <= 512

        self.tokens = [self.tokenizer.cls_token] + self.tokens[1:num_orig_tokens+1] + [self.tokenizer.sep_token]
        self.token_ids = [self.tokenizer.cls_token_id] + self.token_ids[1:num_orig_tokens+1] + [self.tokenizer.sep_token_id]
        self.attention_mask = [1] * (num_orig_tokens+2)
        self.original_token_mask = [0] + [1] * num_orig_tokens + [0]

        assert len(self.tokens) == len(self.token_ids) == len(self.attention_mask) == len(self.original_token_mask)
