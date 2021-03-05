#  Copyright (c) 2020 Robert Bosch GmbH
#  All rights reserved.
#
#  This source code is licensed under the AGPL v3 license found in the
#  LICENSE file in the root directory of this source tree.
#
#  Author: Stefan Grünewald

from torch.utils.data.sampler import Sampler
from random import shuffle
from itertools import chain


class BucketBatchSampler(Sampler):
    """This class implements the following procedure of sampling batches of instances from a PyTorch dataset:

        1. Shuffle the data and split it up into "buckets" of a predefined size.
        2. Sort the instances within each bucket using the specified `size_fn`.
        3. Sample batches of instances from the buckets, maintaining the constraint that the cumulative size of
          each batch does not exceed `max_cumsize_per_batch`. If the addition of an instance would lead to a batch
          exceeding this size, it is moved to the next batch. The cumulative size of a batch is computed as
          `len(batch) * largest_instance_size` to account for padding.
    """

    def __init__(self, dataset, batch_size, bucket_size, size_fn=lambda x: len(x) ** 2, max_cumsize_per_batch=None):
        """
        Args:
            dataset: Dataset containing the instances.
            batch_size: Maximum number of instances per batch.
            bucket_size: Size of buckets that the data is split up into. The larger the bucket size, the more
              homogeneous the individual batches will be in terms of the size of instances
            size_fn: Function to compute the size of an instance with (default: `len(inst)**2`).
            max_cumsize_per_batch: The maximum cumulative size that a batch is allowed to have.
        """
        super(BucketBatchSampler, self).__init__(dataset)

        self.batch_size = batch_size
        self.bucket_size = bucket_size
        self.size_fn = size_fn
        self.max_cumsize_per_batch = max_cumsize_per_batch

        self.instance_sizes = [size_fn(instance) for instance in dataset]

        self.num_batches = None

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        buckets = self._rebucket()

        batches = []

        curr_batch = []
        largest_instance_in_batch = 0
        for instance_ix, instance_size in chain(*buckets):
            largest_instance_after_adding = max(largest_instance_in_batch, instance_size)
            batch_elements_after_adding = (len(curr_batch)+1)*largest_instance_after_adding

            if self.max_cumsize_per_batch is None or batch_elements_after_adding <= self.max_cumsize_per_batch:
                curr_batch.append(instance_ix)
                largest_instance_in_batch = largest_instance_after_adding

                if len(curr_batch) == self.batch_size:
                    batches.append(curr_batch)
                    curr_batch = []
                    largest_instance_in_batch = 0
            else:
                if len(curr_batch) > 0:
                    batches.append(curr_batch)
                curr_batch = [instance_ix]
                largest_instance_in_batch = instance_size

        if curr_batch:
            batches.append(curr_batch)

        self.num_batches = len(batches)

        shuffle(batches)

        # Move largest batch to front
        largest_batch_ix = 0
        largest_bach_size = 0
        for i, batch in enumerate(batches):
            batch_size = max(self.instance_sizes[inst_ix] for inst_ix in batch) * len(batch)
            if batch_size > largest_bach_size:
                largest_bach_size = batch_size
                largest_batch_ix = i

        batches[0], batches[largest_batch_ix] = batches[largest_batch_ix], batches[0]

        return iter(batches)

    def _rebucket(self):
        ixs_with_sort_keys = list(enumerate(self.instance_sizes))
        shuffle(ixs_with_sort_keys)
        unsorted_buckets = [ixs_with_sort_keys[i:i+self.bucket_size]
                            for i in range(0, len(ixs_with_sort_keys), self.bucket_size)]
        sorted_buckets = [sorted(bucket, key=lambda x: x[1]) for bucket in unsorted_buckets]

        return sorted_buckets
