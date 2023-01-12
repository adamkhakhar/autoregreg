import torchvision
import torch
import numpy as np
import os
import sys
from typing import List
import random

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(f"{ROOT_DIR}/utils")
import utils


class MNISTDataSet:
    def __init__(
        self,
        input_fun: List,
        target_fun: List,
        num_samples: int,
        dist_probs: List[float],
        data_dir=f"{ROOT_DIR}/mnist/files",
        flat_input=False,
        auto_regressive=False,
        base=-1,
        exp_min=-1,
        exp_max=-1,
    ):
        self.input_fun = input_fun
        self.target_fun = target_fun
        self.num_samples = num_samples
        self.flat_input = flat_input
        self.data_set = torchvision.datasets.MNIST(
            data_dir,
            train=True,
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        )
        self.auto_regressive = auto_regressive
        self.base = base
        self.exp_min = exp_min
        self.exp_max = exp_max
        assert all([p_i > 0 for p_i in dist_probs])
        assert all([p_i <= 1 for p_i in dist_probs])
        assert sum(dist_probs) > 0.99 and sum(dist_probs) < 1.01
        assert len(input_fun) == len(target_fun)
        assert len(target_fun) == len(dist_probs)

    def __len__(self):
        return self.num_samples

    def __find_correct_distribution(self):
        return np.random.choice(len(self.target_fun), p=self.dist_probs)

    def __getitem__(self, idx):
        # generate MNIST 4 image input
        samples = []
        x = 0
        for i in range(4):
            rand_idx = random.randint(0, self.data_set.__len__() - 1)
            curr_sample = self.data_set.__getitem__(rand_idx)
            samples.append(curr_sample[0])
            x += curr_sample[1] * (10 ** (3 - i))
        x = x / 10_000

        # reshape input
        list_of_tensors = []
        for i in range(samples[0].shape[2]):
            list_of_tensors.append(torch.cat((samples[0][0][i], samples[1][0][i])))
        for i in range(samples[0].shape[2]):
            list_of_tensors.append(torch.cat((samples[2][0][i], samples[3][0][i])))
        tensor_of_tensors = torch.vstack(tuple(list_of_tensors))
        tensor_of_tensor_of_tensors = tensor_of_tensors.reshape(
            1, samples[0].shape[2] * 2, samples[0].shape[2] * 2
        )
        input = tensor_of_tensor_of_tensors

        distribution_ind = self.__find_correct_distribution()
        # modify input according to input_fun
        input = self.input_fun[distribution_ind](input)

        # create targets from input
        target_lst = [self.target_fun[distribution_ind](x)]
        target = torch.tensor(
            target_lst,
            dtype=torch.float,
        )
        if self.flat_input:
            input = torch.Tensor(np.reshape(input.numpy(), (56 * 56)))
        if not self.auto_regressive:
            return {"input": input, "target": target}
        else:
            target_output = []
            for target_index in range(len(target)):
                current_target_outputs = []
                exponent_notation = utils.float_to_exponent_notation(
                    target[target_index],
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
                "input": input,
                "target": target_output,
                "distribution_ind": distribution_ind,
            }
