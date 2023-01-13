import torch
from typing import List
import os
import sys
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(f"{ROOT_DIR}/utils")
import utils


class OneDimMultiDistributionSingleTarget:
    def __init__(
        self,
        input_fun: List,
        target_fun: List,
        dist_probs: List[float],
        num_samples: int,
        auto_regressive=False,
        base=-1,
        exp_min=-1,
        exp_max=-1,
    ):
        self.input_fun = input_fun
        self.target_fun = target_fun
        self.dist_probs = dist_probs
        self.num_samples = num_samples
        assert len(target_fun) == len(input_fun)
        assert len(target_fun) == len(dist_probs)
        assert all([p_i > 0 for p_i in dist_probs])
        assert all([p_i <= 1 for p_i in dist_probs])
        assert sum(dist_probs) > 0.99 and sum(dist_probs) < 1.01
        self.auto_regressive = auto_regressive
        self.base = base
        self.exp_min = exp_min
        self.exp_max = exp_max

    def __len__(self):
        return self.num_samples

    def __find_correct_distribution(self):
        return np.random.choice(len(self.target_fun), p=self.dist_probs)

    def __getitem__(self, idx: int):
        distribution_ind = self.__find_correct_distribution()
        input = [self.input_fun[distribution_ind]()]
        target = [self.target_fun[distribution_ind](input[0])]
        if not self.auto_regressive:
            return {
                "input": torch.tensor(input, dtype=torch.float),
                "target": torch.tensor(target, dtype=torch.float),
                "distribution_ind": distribution_ind,
                "orig_value": torch.tensor(input, dtype=torch.float),
            }
        else:
            target_output = []
            current_target_outputs = []
            exponent_notation = utils.float_to_exponent_notation(
                target[0],
                self.base,
                self.exp_min,
                self.exp_max,
            )
            for bin_ind in range(self.exp_max - self.exp_min + 1):
                one_hot_tensor = torch.zeros(self.base)
                one_hot_tensor[exponent_notation[bin_ind]] = 1
                current_target_outputs.append(one_hot_tensor)
            target_output.append(current_target_outputs)
            return {
                "input": torch.tensor(input, dtype=torch.float),
                "target": target_output,
                "distribution_ind": distribution_ind,
                "orig_value": torch.tensor(input, dtype=torch.float),
            }
