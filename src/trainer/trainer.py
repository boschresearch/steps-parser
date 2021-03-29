# This source code is from the PyTorch Template Project (w/ very heavy adaptations)
#   (https://github.com/victoresque/pytorch-template/blob/master/trainer/trainer.py)
# Copyright (c) 2018 Victor Huang
# This source code is licensed under the MIT license found in the
# 3rd-party-licenses.txt file in the root directory of this source tree.

import time
import torch

from numpy import inf
from torch.nn.utils import clip_grad_norm_

from models.multi_parser import update_eval_counts


class Trainer:
    """An object of this class is responsible for executing the training logic on a given model: Loading the data,
    running training (backpropagation), validation, and evaluation.
    """
    def __init__(self, model, config, optimizer, loss, eval_criterion, train_data_loader, valid_data_loader,
                 loss_scaler=None, lr_scheduler=None, clip_grad_norm=None):
        """
        Args:
            model: The model to train.
            config: The experimental configuration.
            optimizer: The optimizer to use in training.
            loss: Dictionary mapping output IDs to the loss functions to use for them.
            eval_criterion: EvaluationCriterion to use for validation.
            train_data_loader: Data loader for training data.
            valid_data_loader: Data loader for validation data.
            loss_scaler: LossScaler object to use for scaling losses for output tasks (default: None).
            lr_scheduler: Learning rate scheduler to use (default: None).
            clip_grad_norm: Value to use for gradient clippint (default: None).
        """
        self.config = config
        #self.logger = config.get_logger('trainer', config['trainer']['verbosity'])
        self.logger = config.logger

        # Set up GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(config['n_gpu'])
        self.parser = model.to(self.device)
        if len(device_ids) > 1:
            self.parser.parallelize(device_ids)

        # Set up evaluation criterion, loss, optimizer, and LR scheduler (optional)
        self.loss = loss
        self.optimizer = optimizer
        self.loss_scaler = loss_scaler
        self.lr_scheduler = lr_scheduler
        self.eval_criterion = eval_criterion

        # Set up gradient norm clip
        self.clip_grad_norm = clip_grad_norm

        # Set up epochs and checkpoint frequency
        cfg_trainer = config['trainer']
        self.min_epochs = cfg_trainer['min_epochs']
        self.max_epochs = cfg_trainer['max_epochs']
        self.save_period = cfg_trainer['save_period']
        self.start_epoch = 1
        self.early_stop = cfg_trainer.get('early_stop', inf)

        # Set up data loaders for training/validation examples
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader

        # Set up checkpoint saving and loading
        self.checkpoint_dir = config.save_dir
        if config.resume is not None:
            self._resume_checkpoint(config.resume)

    def train(self):
        """Commence training on the model, using the parameters specified for the trainer."""
        training_starttime = time.time()
        not_improved_count = 0

        for epoch in range(self.start_epoch, self.max_epochs + 1):
            # Perform one training epoch and output training metrics
            epoch_starttime = time.time()
            training_metrics = self.run_epoch(epoch, self.train_data_loader, training=True)
            epoch_duration = (time.time() - epoch_starttime) / 60
            self.logger.info("Training epoch {} finished. (Took {:.1f} mins.)".format(epoch, epoch_duration))
            self.logger.log_epoch_metrics(training_metrics, step=epoch, suffix="_train")

            # Perform one validation epoch and output validation metrics
            epoch_starttime = time.time()
            validation_metrics = self.run_epoch(epoch, self.valid_data_loader, training=False)
            epoch_duration = (time.time() - epoch_starttime) / 60
            self.logger.info("Validation epoch {} finished. (Took {:.1f} mins.)".format(epoch, epoch_duration))
            self.logger.log_epoch_metrics(validation_metrics, step=epoch, suffix="_valid")

            # Check if model is new best according to validation F1 score
            improved = self.eval_criterion.last_update_improved_best()
            if improved:
                not_improved_count = 0
            else:
                not_improved_count += 1

            if improved or epoch % self.save_period == 0:
                self._save_checkpoint(epoch, is_best=improved)

            if not_improved_count > self.early_stop and epoch >= self.min_epochs:
                self.logger.info("Validation criterion didn\'t improve for {} epochs. "
                                 "Training stops.".format(self.early_stop))
                training_duration = (time.time() - training_starttime) / 60
                self.logger.info("Training took {:.1f} mins.".format(training_duration))
                return

        self.logger.info("Maximum epoch number reached. Training stops.")
        training_duration = (time.time() - training_starttime) / 60
        self.logger.info("Training took {:.1f} mins.".format(training_duration))

    def run_epoch(self, epoch, data_loader, training=False):
        """Run one epoch.

        Args:
            epoch: Current epoch number (integer).
            data_loader: Data loader to fetch training examples from.
            training: If true, model will be trained (i.e. backpropagation happens). Default: False.

        Returns:
            A dictionary that contains information about metrics (loss, precision, recall, f1).
        """

        if training:
            self.parser.train()
        else:
            self.parser.eval()

        epoch_loss = 0.0
        epoch_annotation_counts = dict()
        num_evaluated_batches = 0

        with torch.set_grad_enabled(training):
            for batch in data_loader:
                batch_loss, batch_annotation_counts = self.process_batch(batch, epoch, training=training)

                # Add batch counts to epoch counts
                update_eval_counts(epoch_annotation_counts, batch_annotation_counts)
                epoch_loss += batch_loss

                # Print progress
                num_evaluated_batches += 1
                self.logger.debug('{} Epoch: {} {} Loss: {:.6f}'.format(
                    "Training" if training else "Validation",
                    epoch,
                    self._progress(num_evaluated_batches, data_loader),
                    batch_loss))

        # Compute epoch metrics; log if validation epoch
        epoch_metrics = self.eval_criterion.compute_metrics_for_counts(epoch_annotation_counts)
        if not training:
            self.eval_criterion.log_metrics(epoch_metrics)  # Log validation metrics for early stopping
        elif training and not self.eval_criterion.weighting == "pareto":
            epoch_metrics["_AGGREGATE_"] = self.eval_criterion.compute_aggregate_metric(epoch_metrics)
        epoch_metrics["_loss"] = epoch_loss

        return epoch_metrics

    def process_batch(self, batch, epoch, training=False):
        """Run a single batch through the model during a training epoch.

        Args:
            batch: The batch to feed to the model.
            epoch: Current epoch number (integer).
            training: If true, model will be trained (i.e. backpropagation happens). Default: False.

        Returns:
            A tuple containing (a)
        """
        batch_loss = 0.0

        # Run model
        sentences, targets = batch
        targets = self._to_device(targets)

        outputs, batch_annotation_counts = self.parser.evaluate_batch(sentences)

        assert targets.keys() == outputs.keys()

        # Compute losses for all output tasks
        for i, outp_id in enumerate(outputs):
            output = self._unroll_sequence_batch(outputs[outp_id])
            target = self._unroll_sequence_batch(targets[outp_id])

            loss = self.loss[outp_id](output, target)

            # Scale loss (if specified)
            if self.loss_scaler is not None:
                loss *= self.loss_scaler.get_loss_scaling_factor(outp_id, epoch)

            # Add metrics to overall total
            batch_loss += loss.item()

            # Perform backpropagation (when training)
            if training:
                retain_graph = i != len(outputs) - 1  # Release graph after last output
                loss.backward(retain_graph=retain_graph)

        # Run optimizer on the accumulated gradients
        if training:
            if self.clip_grad_norm:
                clip_grad_norm_(self.parser.parameters(), self.clip_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()  # Zero gradients for next batch
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()  # Take an LR scheduler step after each batch
                self.logger.info("LRs are now: " + ", ".join("{:.2e}".format(lr) for lr in self.lr_scheduler.get_lr()))

        return batch_loss, batch_annotation_counts

    def _unroll_sequence_batch(self, batch):
        """Unroll a batch of sequences, i.e. flatten batch and sequence dimension. (Used for loss computation)"""
        shape = batch.shape
        if len(shape) == 3:  # Model output
            return batch.view(shape[0]*shape[1], shape[2])
        elif len(shape) == 2:  # Target labels
            return batch.view(shape[0]*shape[1])

    def _progress(self, num_completed_batches, data_loader):
        """Nicely formatted epoch progress"""
        return '[{}/{} ({:.0f}%)]'.format(num_completed_batches, len(data_loader),
                                          100.0 * num_completed_batches / len(data_loader))

    def _prepare_device(self, n_gpu_use):
        """
        setup GPU device if available, move model into configured device
        """
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning("Warning: There\'s no GPU available on this machine,"
                                "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning("Warning: The number of GPU\'s configured to use is {}, but only {} are available "
                                "on this machine.".format(n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _save_checkpoint(self, epoch, is_best=False):
        """Save a checkpoint.

        Args:
            epoch: current epoch number
            log: logging information of the epoch
            is_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.parser).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.parser.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.config
        }
        filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))

        if is_best:
            best_path = str(self.checkpoint_dir / 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")
        else:
            torch.save(state, filename)
            self.logger.info("Saving regular checkpoint: {} ...".format(filename))

    def _resume_checkpoint(self, resume_path):
        """Resume from saved checkpoint.

        Args:
            resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path, map_location="cpu")
        self.start_epoch = checkpoint['epoch'] + 1

        # load architecture params from checkpoint.
        if checkpoint['config']['model'] != self.config['model']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.parser.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['trainer']['optimizer']['type'] != self.config['trainer']['optimizer']['type']:
            self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info("Checkpoint loaded. Resume from epoch {}".format(self.start_epoch))

    def _to_device(self, data):
        if isinstance(data, torch.Tensor):
            assert data.device != self.device
            return data.to(self.device)
        elif isinstance(data, dict):
            assert all(isinstance(val, torch.Tensor) for val in data.values())
            assert all(val.device != self.device for val in data.values())
            data_on_device = dict()
            for key in data:
                data_on_device[key] = data[key].to(self.device)
            return data_on_device
        else:
            raise Exception("Cannot move this kind of data to a device!")
