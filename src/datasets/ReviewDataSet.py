import numpy as np
import torch
import random
import io
import boto3
import pickle
from typing import List
import os
import sys
import ipdb

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(f"{ROOT_DIR}/utils")

from utils import pull_from_s3


class ReviewDataSet:
    def __init__(
        self,
        target_fun: List,
        num_samples=None,
        bucket_name="review-shards",
        max_key=26_000,
        start_token=10,
        num_samples_per_shard=1_024,
        clip_input_size=2000,
    ):
        self.target_fun = target_fun
        self.bucket_name = bucket_name
        self.num_samples = num_samples
        self.max_key = max_key
        self.start_token = np.array([start_token])
        self.num_samples_per_shard = num_samples_per_shard
        self.num_samples = (
            max_key * num_samples_per_shard if num_samples is None else num_samples
        )
        self.clip_input_size = clip_input_size
        self.curr_shard = None
        self.curr_shard_ind = None
        self.reset_shard()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.curr_shard_ind >= self.num_samples_per_shard:
            self.reset_shard()
        curr_review = self.curr_shard[self.curr_shard_ind]["review"][
            : self.clip_input_size
        ]
        # ensure all input to model are of clip_input_size (largest size of input)
        input = np.concatenate(
            (
                curr_review,
                np.array(
                    [self.start_token.item()]
                    * (self.clip_input_size - len(curr_review))
                ),
            )
        )

        # generate target from target functions
        target = torch.tensor(
            [foo(self.curr_shard[self.curr_shard_ind]) for foo in self.target_fun],
            dtype=torch.float,
        )
        self.curr_shard_ind += 1
        return {
            "input": torch.tensor(input, dtype=torch.int),
            "target": torch.tensor(target, dtype=torch.float),
        }

    def get_shard(self):
        ind = random.randint(0, self.max_key)
        return pull_from_s3(str(ind), bucket_name=self.bucket_name)

    def reset_shard(self):
        del self.curr_shard
        self.curr_shard = self.get_shard()
        self.curr_shard_ind = 0
