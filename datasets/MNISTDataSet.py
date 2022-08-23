import torchvision
import torch
import numpy as np
import os
from typing import List
import random

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


class MNISTDataSet:
    def __init__(
        self,
        target_fun: List,
        num_samples: int,
        data_dir=f"{ROOT_DIR}/mnist/files",
        flat_input=False,
        start_token=None,
    ):
        self.target_fun = target_fun
        self.num_samples = num_samples
        self.flat_input = flat_input
        self.start_token = start_token
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

    def __len__(self):
        return self.num_samples

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

        # create targets from input
        target_lst = [foo(x) for foo in self.target_fun]
        target = torch.tensor(
            target_lst,
            dtype=torch.float,
        )
        if self.flat_input:
            input = np.insert(
                np.reshape(input.numpy(), (56 * 56)), 0, self.start_token
            )
        return {"input": input, "target": target}
