import torch
import torch.nn as nn
from typing import List


class AutoRegressiveHead(nn.Module):
    def __init__(self, input_size: int, bin_size: List[int], number_steps: List[int]):
        super().__init__()
        self.input_size = input_size
        self.bin_size = bin_size
        self.number_steps = number_steps
        assert len(bin_size) == len(number_steps)

        self.num_targets = len(self.bin_size)
        self.layers = []
        for target_index in range(self.num_targets):
            curr_target_layers = []
            for step_index in range(self.number_steps[target_index]):
                curr_target_layers.append(nn.ReLU())
                curr_target_layers.append(
                    nn.Linear(
                        input_size + step_index * self.bin_size[target_index],
                        self.bin_size[target_index],
                    )
                )
            self.layers.append(curr_target_layers)

    def forward(self, x):
        """
        Returns
        -------
        list
            list of outputs for each target, where each output is a list of
            tensors corresponding to the logits of each bin. # outer lists = #
            targets, # inner lists = # autoregressive steps
        """
        outputs = []
        for target_index in range(self.num_targets):
            curr_target_outputs = []
            for step_index in range(self.number_steps[target_index]):
                input_and_prev_steps = torch.concat([x] + curr_target_outputs, axis=1)
                post_relu = self.layers[target_index][step_index * 2](
                    input_and_prev_steps
                )
                output = self.layers[target_index][step_index * 2 + 1](post_relu)
                curr_target_outputs.append(output)
            outputs.append(curr_target_outputs)
        return outputs

    def move_all_layers_to_device(self, device):
        for target_index in range(len(self.layers)):
            for layer_index in range(len(self.layers[target_index])):
                self.layers[target_index][layer_index] = self.layers[target_index][
                    layer_index
                ].to(device)
