import torch
from typing import List


class OneDimDataSet:
    def __init__(self, input_fun: List, target_fun: List, num_samples: int):
        self.input_fun = input_fun
        self.target_fun = target_fun
        self.num_samples = num_samples
        assert len(target_fun) == len(input_fun)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        inputs = [in_fun() for in_fun in self.input_fun]
        targets = [self.target_fun[i](inputs[i]) for i in range(len(self.target_fun))]
        return {
            "input": torch.tensor(inputs, dtype=torch.float),
            "target": torch.tensor(targets, dtype=torch.float),
        }
