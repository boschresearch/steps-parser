# This source code is from the PyTorch Template Project (w/ very heavy adaptations)
#   (https://github.com/victoresque/pytorch-template/blob/master/parse_config.py)
# Copyright (c) 2018 Victor Huang
# This source code is licensed under the MIT license found in the
# 3rd-party-licenses.txt file in the root directory of this source tree.

import transformers
import json

import data_handling.vocab as vocab_module
import data_handling.data_loaders as data_loaders_module
import torch.optim as optimizers_module
import trainer.losses as loss_module
import models.embeddings as embeddings_module
import models.outputs as outputs_module
import trainer.lr_scheduler as lr_scheduler_module
import models.post_processing as post_processing_module

from pathlib import Path
from functools import reduce
from operator import getitem
from datetime import datetime
from collections import OrderedDict
from torch.nn import ModuleDict

from logger import Logger
from trainer.trainer import Trainer
from models.multi_parser import MultiParser
from trainer.eval_criterion import EvaluationCriterion
from trainer.loss_scaler import LossScaler
from trainer.lr_scheduler import SqrtSchedule, WarmRestartSchedule


class ConfigParser:
    """This class parses the configuration json file and handles hyperparameters for training, checkpoint saving
    and logging module.

    Most importantly, however, it initializes the actual model itself, including all of its components (e.g.
    the embedding layer or the transformer encoder). This means that this class actually does a lot of heavy lifting
    that you might expect to find in the constructor of the model itself. However, I decided to put it here to keep
    that class readable.
    """
    def __init__(self, config, resume=None, modification=None, run_id=None):
        """
        Args:
            config: Dict containing configurations, hyperparameters for training. Contents of `config.json` file, for
              example.
            resume: String, path to a checkpoint to load. Default: None.
            modification: Dict keychain:value, specifying position values to be replaced in config dict.
            run_id: Unique identifier for training processes. Used to save checkpoints and training log. Timestamp
              is being used as default.
        """
        # load config file and apply modification
        self._config = _update_config(config, modification)
        self.resume = resume

        experiment = self.config.get('experiment', None)
        run_name = self.config['name']

        self.saving = self.config.get("saving", True)
        if self.saving:
            # set save_dir where trained model and log will be saved.
            save_dir = Path(self.config['trainer']['save_dir'])

            if run_id is None:  # use timestamp as default run-id
                run_id = datetime.now().strftime(r'%m%d_%H%M%S')

            if experiment is None:
                self._save_dir = save_dir / run_name / run_id
            else:
                self._save_dir = save_dir / experiment / run_name / run_id

            # make directory for saving checkpoints and log.
            exist_ok = run_id == ''
            self.save_dir.mkdir(parents=True, exist_ok=exist_ok)

            # save updated config file to the checkpoint dir
            write_json(self.config, self.save_dir / 'config.json')
        else:
            self._save_dir = None

        # Set up logging
        transformers.logging.set_verbosity_info()
        self.logger = Logger(self.save_dir, use_mlflow=True, experiment_id=experiment, run_name=run_name)
        if self.save_dir is not None:
            self.logger.log_config(config)
            self.logger.log_artifact(self.save_dir / 'config.json')

    @classmethod
    def from_args(cls, args, modification=None):
        """Initialize this class from CLI arguments. Used in training, parsing.

        Args:
            args: CLI arguments (as returned by argparse.ArgumentParser).
            modification: ict keychain:value, specifying position values to be replaced in config dict.
        """
        if hasattr(args, "model_dir") and args.model_dir is not None:
            assert (not hasattr(args, "resume") or args.resume is None) and \
                   (not hasattr(args, "config") or args.config is None)
            model_dir = Path(args.model_dir)
            assert model_dir.is_dir(), "Model directory must be an actual directory!"
            resume = model_dir / 'model_best.pth'
            cfg_fname = model_dir / 'config.json'
        else:
            if args.resume is not None:
                resume = Path(args.resume)
                cfg_fname = resume.parent / 'config.json'
            else:
                msg_no_cfg = "Configuration file need to be specified."
                assert args.config is not None, msg_no_cfg
                resume = None
                cfg_fname = Path(args.config)
        
        config = read_json(cfg_fname)
        if hasattr(args, "config") and resume:
            # update new config for fine-tuning
            config.update(read_json(args.config))

        return cls(config, resume, modification=modification)

    def init_model(self):
        """Initialize the model as specified in the configuration file."""
        params = self["model"]

        model_type = params["type"]
        assert model_type == "MultiParser"
        model_args = params["args"]

        # Initialise the components of the model
        embeddings_processor = self._init_embeddings(model_args["embeddings_processor"], set(model_args["outputs"].keys()))

        if self.saving:
            embeddings_processor.save_transformer_config(self.save_dir)

        embeddings_dim = embeddings_processor.embedding_dim

        model_outputs = self._init_outputs(model_args["outputs"], embeddings_dim)

        post_processors = self._init_post_processors(model_args["post_processors"], model_outputs)

        # Build and return the actual model
        return MultiParser(embeddings_processor, model_outputs, post_processors=post_processors)

    def _init_embeddings(self, params, output_ids):
        """Internal method to initialize the part of the model that provides the input embeddings for the main model."""
        embeddings_type = params["type"]
        embeddings_args = params["args"]

        embeddings_args["output_ids"] = output_ids
        if embeddings_type == "LSTMProcessor":
            embeddings_args["embeddings_wrapper"] = self._init_embeddings(embeddings_args["embeddings_wrapper"],
                                                                          output_ids={"lstm"})

        return getattr(embeddings_module, embeddings_type)(**embeddings_args)

    def _init_outputs(self, params, input_dim):
        """Internal method to initialize the output modules of the model."""
        outputs = ModuleDict({outp_id: self._init_output(params[outp_id], input_dim) for outp_id in params})

        # Save output_vocabs to model directory if model is being saved
        if self.saving:
            self._save_output_vocabs(outputs)

        return outputs

    def _init_output(self, params, input_dim):
        """Internal method to initialize a singular model output."""
        output_type = params["type"]
        output_args = params["args"]

        output_args["vocab"] = self._init_vocab(output_args["vocab"])
        output_args["input_dim"] = input_dim

        return getattr(outputs_module, output_type)(**output_args)

    def _init_post_processors(self, params, model_outputs):
        """Internal method to initialize post-processing units for the parser.

        Post-processors are assumed to be static classes. (This may change in the future.)
        """
        output_vocabs = {outp_id: model_outputs[outp_id].vocab for outp_id in model_outputs}

        post_processors = list()
        for curr_params in params:
            pp_type = curr_params["type"]
            pp_args = curr_params["args"]
            pp_args["vocabs"] = output_vocabs

            post_processors.append(getattr(post_processing_module, pp_type)(**pp_args))

        return post_processors

    def _init_vocab(self, params):
        """Internal method to initialize the output vocabulary"""
        vocab_type = params["type"]
        vocab_args = params["args"] if "args" in params else {}

        return getattr(vocab_module, vocab_type)(**vocab_args)

    def init_data_loaders(self, model):
        """Initialize the data loaders as specified in the configuration file, and in such a way that they provide
        valid input for the given model.
        """
        params = self["data_loaders"]
        data_loader_type = params["type"]
        data_loader_args = params["args"]

        data_loader_args["output_vocabs"] = {outp_id: model.outputs[outp_id].vocab for outp_id in model.outputs}

        data_loaders = dict()
        for p in params["paths"]:
            data_loader_args["corpus_path"] = params["paths"][p]
            data_loaders[p] = getattr(data_loaders_module, data_loader_type)(**data_loader_args)

        return data_loaders

    def init_trainer(self, model, train_data_loader, dev_data_loader):
        """Initialize the trainer for the given model. The model is trained on the specified train_data_loader and
        validated on the specified dev_data_loader.

        Args:
            model: Model to load data for.
            train_data_loader: Data loader for training data.
            dev_data_loader: Data loader for validation data.

        Returns:
            A trainer that trains the given model on data provided by the specified data loaders.
        """
        params = self["trainer"]

        if "param_groups" in params:
            param_groups = self._init_param_groups(model, params["param_groups"])
        else:
            param_groups = None

        optimizer = self._init_optimizer(model, params["optimizer"], param_groups=param_groups)

        losses = self._init_losses(params["loss"], model.outputs)

        eval_criterion = self._init_eval_criterion(params["validation_criterion"])

        if "loss_scaling" in params:
            loss_scaler = self._init_loss_scaler(params["loss_scaling"])
        else:
            loss_scaler = None

        if "lr_scheduler" in params:
            lr_scheduler = self._init_lr_scheduler(optimizer, params["lr_scheduler"])
        else:
            lr_scheduler = None

        trainer = Trainer(model, self, optimizer, losses, eval_criterion, train_data_loader, dev_data_loader,
                          loss_scaler=loss_scaler, lr_scheduler=lr_scheduler)

        return trainer

    def _init_optimizer(self, model, params, param_groups=None):
        optimizer_type = params["type"]
        optimizer_args = params["args"]

        if param_groups is not None:
            return getattr(optimizers_module, optimizer_type)(param_groups, **optimizer_args)
        else:
            return getattr(optimizers_module, optimizer_type)(model.parameters(), **optimizer_args)

    def _init_param_groups(self, model, params):
        """Divide the model's parameters into groups: One default group and one or more "special" groups. The latter
        are specified via strings that must be part of a parameter's name to become part of the respective special
        group.
        """
        special_param_groups = list()
        signifiers = list()

        for param_group_params in params:
            signifier = param_group_params["signifier"]
            lr = param_group_params["lr"]

            special_param_group = {"params": [p[1] for p in model.named_parameters() if signifier in p[0]],
                                   "lr": lr}
            signifiers.append(signifier)
            special_param_groups.append(special_param_group)

        default_param_group = {"params": [p[1] for p in model.named_parameters() if all(signifier not in p[0] for signifier in signifiers)]}

        return [default_param_group] + special_param_groups

    def _init_losses(self, params, model_outputs):
        losses = dict()

        if params.keys() == {"type", "args"}:
            # If only one loss is specified, use that for all the outputs
            loss = self._init_loss(params)
            for outp_id in model_outputs:
                losses[outp_id] = loss
        else:
            assert params.keys() == model_outputs.keys()
            for outp_id in model_outputs:
                loss = self._init_loss(params[outp_id])
                losses[outp_id] = loss

        return losses

    def _init_loss(self, params):
        loss_type = params["type"]
        loss_args = params["args"]

        return getattr(loss_module, loss_type)(**loss_args)

    def _init_eval_criterion(self, params):
        return EvaluationCriterion(params["metrics"], params["weighting"])

    def _init_loss_scaler(self, params):
        return LossScaler(params)

    def _init_lr_scheduler(self, optimizer, params):
        scheduler_type = params["type"]
        scheduler_args = params["args"]

        if "lr_lambda" in scheduler_args:
            if isinstance(scheduler_args["lr_lambda"], str):
                scheduler_args["lr_lambda"] = eval(scheduler_args["lr_lambda"])
            elif isinstance(scheduler_args["lr_lambda"], list):
                scheduler_args["lr_lambda"] = [eval(l) for l in scheduler_args["lr_lambda"]]
            else:
                raise Exception("lr_lambda must be a string or a list of strings")

        return getattr(lr_scheduler_module, scheduler_type)(optimizer, **scheduler_args)

    def _save_output_vocabs(self, outputs):
        for outp_id, output in outputs.items():
            if hasattr(output.vocab, "to_file"):
                output.vocab.to_file(self.save_dir / "{}.vocab".format(outp_id))

    def __getitem__(self, name):
        """Access items like ordinary dict."""
        return self.config[name]

    # setting read-only attributes
    @property
    def config(self):
        return self._config

    @property
    def save_dir(self):
        return self._save_dir

    @property
    def log_dir(self):
        return self._log_dir


# Helper functions to update config dict with custom cli options
def _update_config(config, modification):
    if modification is None:
        return config

    for k, v in modification.items():
        if v is not None:
            _set_by_path(config, k, v)
    return config


def _get_opt_name(flags):
    for flg in flags:
        if flg.startswith('--'):
            return flg.replace('--', '')
    return flags[0].replace('--', '')


def _set_by_path(tree, keys, value):
    """Set a value in a nested object in tree by sequence of keys."""
    keys = keys.split('.')
    _get_by_path(tree, keys[:-1])[keys[-1]] = value


def _get_by_path(tree, keys):
    """Access a nested object in tree by sequence of keys."""
    return reduce(getitem, keys, tree)


# Helper functions for reading/writing JSON
def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)
