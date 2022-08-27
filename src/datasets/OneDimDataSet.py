import torch
from typing import List
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(f"{ROOT_DIR}/utils")
import utils


class OneDimDataSet:
    def __init__(
        self,
        input_fun: List,
        target_fun: List,
        num_samples: int,
        auto_regressive=False,
        bases=[],
        exp_min=[],
        exp_max=[],
    ):
        self.input_fun = input_fun
        self.target_fun = target_fun
        self.num_samples = num_samples
        assert len(target_fun) == len(input_fun)
        self.auto_regressive = auto_regressive
        self.bases = bases
        self.exp_min = exp_min
        self.exp_max = exp_max
        if auto_regressive:
            assert len(bases) == len(exp_min)
            assert len(bases) == len(exp_max)
            assert len(bases) == len(input_fun)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        inputs = [in_fun() for in_fun in self.input_fun]
        targets = [self.target_fun[i](inputs[i]) for i in range(len(self.target_fun))]
        if not self.auto_regressive:
            return {
                "input": torch.tensor(inputs, dtype=torch.float),
                "target": torch.tensor(targets, dtype=torch.float),
            }
        else:
            target_output = []
            for target_index in range(len(targets)):
                current_target_outputs = []
                exponent_notation = utils.float_to_exponent_notation(
                    targets[target_index],
                    self.bases[target_index],
                    self.exp_min[target_index],
                    self.exp_max[target_index],
                )
                for bin_ind in range(
                    self.exp_max[target_index] - self.exp_min[target_index] + 1
                ):
                    one_hot_tensor = torch.zeros(self.bases[target_index])
                    one_hot_tensor[exponent_notation[bin_ind]] = 1
                    current_target_outputs.append(one_hot_tensor)
                target_output.append(current_target_outputs)
            return {
                "input": torch.tensor(inputs, dtype=torch.float),
                "target": target_output,
            }
