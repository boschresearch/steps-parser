#  Copyright (c) 2020 Robert Bosch GmbH
#  All rights reserved.
#
#  This source code is licensed under the AGPL v3 license found in the
#  LICENSE file in the root directory of this source tree.
#
#  Author: Stefan Grünewald

import numpy as np


class EvaluationCriterion:
    """Class for encoding the evaluation/validation criterion for a given model.

    An instance of this class is responsible for calculating standard evaluation metrics (PRF, accuracy) from
    given observed counts, as well as keeping track of the metrics calculated so far (for bookkeeping and in order
    to check whether an improvement has happened).
    """

    def __init__(self, metrics, weighting):
        """
        Args:
            metrics: Dictonary mapping annotation IDs onto the metric to compute for them.
            weighting: How to weigh the different metrics when calculating the overall score (which is used to check
              whether performance has improved). Possible weighting schemes: "pareto", "multiplicative", or dictionary
              mapping output IDs to weighting factors.
        """
        self.metrics_map = metrics
        self.weighting = weighting

        self.logged_metrics = []
        self.best_time_step = 0

    def compute_metrics_for_counts(self, counts_dict):
        """For a given nested dictionary of annotation counts (outp_id->"predicted"/"gold"/"correct"->count), compute
        precision, recall, F-score.
        """
        metrics = dict()
        for outp_id in counts_dict:
            metrics[outp_id] = compute_prf(counts_dict[outp_id])

        return metrics

    def log_metrics(self, metrics):
        """Internally log a metrics dictionary (as returned by the compute_metrics_for_counts method) and determine whether
        the logged metrics are the new best.
        """
        if self.weighting != "pareto":
            metrics["_AGGREGATE_"] = self.compute_aggregate_metric(metrics)

        self.logged_metrics.append(metrics)

        if self.weighting == "pareto":
            # Note that the ">=" is assuming we always want to maximize!
            improved = all(
                self.logged_metrics[-1][outp_id][metric] >= self.logged_metrics[self.best_time_step][outp_id][metric]
                for outp_id, metric in self.metrics_map.items())
        else:
            # Note that the ">=" is assuming we always want to maximize!
            improved = self.logged_metrics[-1]["_AGGREGATE_"] >= self.logged_metrics[self.best_time_step]["_AGGREGATE_"]

        if improved:
            self.best_time_step = len(self.logged_metrics) - 1

    def last_update_improved_best(self):
        """Check whether the latest logged metrics constitute an improvement over the previously best logged metrics as
        determined by the weighting scheme.
        """
        if len(self.logged_metrics) == 0:
            raise Exception("Can only calculate improvement if something has been logged!")
        else:
            return self.best_time_step == len(self.logged_metrics) - 1

    def compute_aggregate_metric(self, metrics):
        """Compute aggregate of the metrics for the different output IDs, as specified by the weighting scheme."""
        if isinstance(self.weighting, dict):
            return self._compute_weighted_metrics_sum(metrics)
        elif self.weighting == "multiplicative":
            return self._compute_metrics_product(metrics)

    def _compute_metrics_product(self, metrics):
        return np.prod([metrics[outp_id][metric] for outp_id, metric in self.metrics_map.items()])

    def _compute_weighted_metrics_sum(self, metrics):
        assert isinstance(self.weighting, dict)

        aggregate_metric = 0
        for outp_id, weight in self.weighting.items():
            metric = self.metrics_map[outp_id]
            aggregate_metric += weight * metrics[outp_id][metric]

        return aggregate_metric


def compute_prf(counts_dict):
    """Compute precision, recall and F-score for a single class based on counts (gold, predicted, correct)."""
    precision = counts_dict["correct"] / counts_dict["predicted"] if counts_dict["predicted"] else 0.0
    recall = counts_dict["correct"] / counts_dict["gold"] if counts_dict["gold"] else 0.0
    fscore = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {"precision": precision, "recall": recall, "fscore": fscore}
