import torch
import torch.nn as nn
import os
import sys
from typing import List
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(f"{ROOT_DIR}/utils")
import utils

sys.path.append(f"{ROOT_DIR}/src/training")
from Train import Train

import ipdb


class ARRTrain(Train):
    def __init__(
        self,
        experiment_name: str,
        number_targets: int,
        model,
        data_loader,
        optimizer,
        gpu_ind: int,
        log_every: int,
        num_grad_steps: int,
        bases: List[int],
        exp_min: List[int],
        exp_max: List[int],
        num_samples_error_track=100,
        save_local=False,
        upload_to_s3=True,
        bucket_name="arr-saved-experiment-data",
        print_every=False,
        use_wandb=False,
    ):
        # variables from parameters
        self.device = (
            f"cuda:{gpu_ind}" if torch.cuda.is_available() and gpu_ind >= 0 else "cpu"
        )
        self.experiment_name = experiment_name
        self.number_targets = number_targets
        self.model = model.to(self.device)
        self.data_loader = data_loader
        self.optimizer = optimizer
        self.log_every = log_every
        self.num_grad_steps = num_grad_steps
        self.bases = bases
        self.exp_min = exp_min
        self.exp_max = exp_max
        self.num_samples_error_track = num_samples_error_track
        self.print_every = print_every
        self.save_local = save_local
        self.upload_to_s3 = upload_to_s3
        self.bucket_name = bucket_name

        super().__init__(
            self.experiment_name,
            self.model,
            self.data_loader,
            self.optimizer,
            self.device,
            self.log_every,
            self.num_grad_steps,
            save_local=save_local,
            upload_to_s3=self.upload_to_s3,
            bucket_name=self.bucket_name,
            print_every=self.print_every,
            use_wandb=use_wandb,
        )

        # variables used during execution
        self.in_sample_soft_mean_squared_error = {}
        self.out_of_sample_soft_mean_squared_error = {}
        self.in_sample_hard_mean_squared_error = {}
        self.out_of_sample_hard_mean_squared_error = {}
        for i in range(self.number_targets):
            self.in_sample_soft_mean_squared_error[i] = []
            self.out_of_sample_soft_mean_squared_error[i] = []
            self.in_sample_hard_mean_squared_error[i] = []
            self.out_of_sample_hard_mean_squared_error[i] = []
        self.num_distributions = None

    def calculate_loss(self, outputs, targets):
        """Calculates loss of a minibatch

        Parameters
        ----------
        outputs : tensor
            prediction of model

        targets : tensor
            correct answer for model's predictions

        Returns
        -------
        tensor
            scalar loss
        """
        loss = 0
        for target_index in range(len(outputs)):
            for bin_index in range(len(outputs[target_index])):
                loss += nn.CrossEntropyLoss()(
                    outputs[target_index][bin_index],
                    targets[target_index][bin_index].to(self.device),
                )
        if not torch.isfinite(loss):
            print("LOSS IS NAN", flush=True)
            raise Exception("LOSS IS NAN")
        return loss

    def _l_arg_max_to_mse(
        self, l_output_arg_max: List, l_target_arg_max: List, target_index
    ):
        tensor_output_arg_max = torch.stack(l_output_arg_max).T
        tensor_target_arg_max = torch.stack(l_target_arg_max).T
        assert tensor_output_arg_max.shape == tensor_target_arg_max.shape
        batch_size = tensor_output_arg_max.shape[0]
        output_value = utils.matrix_exponent_notation_to_float(
            tensor_output_arg_max,
            self.bases[target_index],
            self.exp_min[target_index],
            self.exp_max[target_index],
        )
        target_value = utils.matrix_exponent_notation_to_float(
            tensor_target_arg_max,
            self.bases[target_index],
            self.exp_min[target_index],
            self.exp_max[target_index],
        )
        mse = torch.sum((output_value - target_value) ** 2).item() / batch_size
        return mse, output_value, target_value

    def input_and_target_to_device(self, input, target):
        """Moves input and target to device

        Parameters
        ----------
        input : input from data loader

        target : target from data loader

        Returns
        -------
        tuple (input, target)
            input and target on device
        """
        return (
            input.to(self.device),
            [[bin.to(self.device) for bin in t] for t in target],
        )

    def compute_mini_batch_metrics(
        self,
        inputs,
        outputs,
        targets,
        in_sample_distribution_ind,
        in_sample_orig_value,
        out_of_sample_input,
        out_of_sample_target,
        out_of_sample_distribution_ind,
        out_of_sample_orig_value,
    ):
        """Calculates loss of a minibatch

        Parameters
        ----------
        inputs : tensor
            inputs of last mini batch

        outputs : tensor
            outputs of last mini batch

        targets : tensor
            features of last mini batch

        in_sample_distribution_ind : List[int]
            index of target distribution sampled from

        in_sample_orig_value : tensor
            original value of target from data

        out_of_sample_input : tensor
            inputs of out of sample mini batch

        out_of_sample_target : tensor
            targets of out of sample mini batch

        out_of_sample_distribution_ind: List[int]
            index of target distribution sampled from

        out_of_sample_orig_value : tensor
            original value of target from data

        Returns
        -------
        dict
            scalar metrics
        """
        with torch.no_grad():
            out_of_sample_output = self.model(out_of_sample_input)
            if in_sample_distribution_ind is None:
                for target_index in range(self.number_targets):
                    for (
                        c_output,
                        c_target,
                        c_hard_dict_to_update,
                        c_soft_dict_to_update,
                    ) in [
                        (
                            outputs,
                            targets,
                            self.in_sample_hard_mean_squared_error,
                            self.in_sample_soft_mean_squared_error,
                        ),
                        (
                            out_of_sample_output,
                            out_of_sample_target,
                            self.out_of_sample_hard_mean_squared_error,
                            self.out_of_sample_soft_mean_squared_error,
                        ),
                    ]:

                        # restrict output and targets to be only
                        for bin_index in range(len(c_output[target_index])):
                            c_output[target_index][bin_index] = c_output[target_index][
                                bin_index
                            ][: self.num_samples_error_track]
                            c_target[target_index][bin_index] = c_target[target_index][
                                bin_index
                            ][: self.num_samples_error_track]

                        # hard error
                        output_logit_arg_max = []
                        target_logit_arg_max = []
                        for bin_index in range(len(c_output[target_index])):
                            output_logit_arg_max.append(
                                torch.argmax(
                                    c_output[target_index][bin_index], dim=1
                                ).cpu()
                            )
                            target_logit_arg_max.append(
                                torch.argmax(
                                    c_target[target_index][bin_index], dim=1
                                ).cpu()
                            )
                        hard_mse, _, target_value = self._l_arg_max_to_mse(
                            output_logit_arg_max, target_logit_arg_max, target_index
                        )
                        c_hard_dict_to_update[target_index].append(hard_mse)

                        # soft error
                        output_soft_max = []
                        for sample_index in range(len(c_output[0][0])):
                            curr_samples = []
                            for bin_index in range(len(c_output[target_index])):
                                soft_max = nn.Softmax(dim=1)(
                                    c_output[target_index][bin_index]
                                ).tolist()[0]
                                soft_max_sum = sum(soft_max)
                                soft_max = [x / soft_max_sum for x in soft_max]
                                curr_samples.append(
                                    np.random.choice(
                                        self.bases[target_index],
                                        self.num_samples_error_track,
                                        p=soft_max,
                                    )
                                )
                            curr_sample_tensor = torch.Tensor(
                                np.stack(curr_samples, axis=1)
                            )
                            float_values = utils.matrix_exponent_notation_to_float(
                                curr_sample_tensor,
                                self.bases[target_index],
                                self.exp_min[target_index],
                                self.exp_max[target_index],
                            )
                            avg_value = torch.mean(float_values)
                            output_soft_max.append(avg_value.item())
                        soft_mse = torch.sum(
                            (torch.Tensor(output_soft_max) - target_value) ** 2
                        ).item() / len(output_soft_max)
                        c_soft_dict_to_update[target_index].append(soft_mse)

                return {
                    "in_sample_hard_mean_squared_error": self.in_sample_hard_mean_squared_error,
                    "out_of_sample_hard_mean_squared_error": self.out_of_sample_hard_mean_squared_error,
                    "in_sample_soft_mean_squared_error": self.in_sample_soft_mean_squared_error,
                    "out_of_sample_soft_mean_squared_error": self.out_of_sample_soft_mean_squared_error,
                }
            else:
                if self.num_distributions is None:
                    self.num_distributions = max(in_sample_distribution_ind) + 1
                    for i in range(self.num_distributions):
                        self.in_sample_soft_mean_squared_error[i] = []
                        self.out_of_sample_soft_mean_squared_error[i] = []
                        self.in_sample_hard_mean_squared_error[i] = []
                        self.out_of_sample_hard_mean_squared_error[i] = []
                for distribution_index in range(self.num_distributions):
                    for (
                        c_output,
                        c_target,
                        distribution_ind,
                        c_hard_dict_to_update,
                        c_soft_dict_to_update,
                    ) in [
                        (
                            outputs,
                            targets,
                            in_sample_distribution_ind,
                            self.in_sample_hard_mean_squared_error,
                            self.in_sample_soft_mean_squared_error,
                        ),
                        (
                            out_of_sample_output,
                            out_of_sample_target,
                            out_of_sample_distribution_ind,
                            self.out_of_sample_hard_mean_squared_error,
                            self.out_of_sample_soft_mean_squared_error,
                        ),
                    ]:
                        mask = distribution_ind == distribution_index
                        indices = torch.squeeze(torch.nonzero(mask), dim=(0))
                        filtered_c_output = []
                        filtered_c_target = []
                        for exp_index in range(len(c_output[0])):
                            if len(indices) > 1:
                                filtered_c_output.append(
                                    torch.squeeze(c_output[0][exp_index][indices])
                                )
                                filtered_c_target.append(
                                    torch.squeeze(c_target[0][exp_index][indices])
                                )
                            else:
                                filtered_c_output.append(
                                    torch.squeeze(c_output[0][exp_index][indices]),
                                    dim=(1, self.exp_max - self.exp_min + 1),
                                )
                                filtered_c_target.append(
                                    torch.squeeze(c_target[0][exp_index][indices]),
                                    dim=(1, self.exp_max - self.exp_min + 1),
                                )
                        c_output = [filtered_c_output]
                        c_target = [filtered_c_target]
                        # restrict output and targets to be only
                        for bin_index in range(len(c_output[0])):
                            c_output[0][bin_index] = c_output[0][bin_index][
                                : self.num_samples_error_track
                            ]
                            c_target[0][bin_index] = c_target[0][bin_index][
                                : self.num_samples_error_track
                            ]

                        # hard error
                        output_logit_arg_max = []
                        target_logit_arg_max = []
                        for bin_index in range(len(c_output[0])):
                            output_logit_arg_max.append(
                                torch.argmax(c_output[0][bin_index], dim=1).cpu()
                            )
                            target_logit_arg_max.append(
                                torch.argmax(c_target[0][bin_index], dim=1).cpu()
                            )
                        hard_mse, _, target_value = self._l_arg_max_to_mse(
                            output_logit_arg_max, target_logit_arg_max, 0
                        )
                        c_hard_dict_to_update[distribution_index].append(hard_mse)

                        # soft error
                        output_soft_max = []
                        for sample_index in range(len(c_output[0][0])):
                            curr_samples = []
                            for bin_index in range(len(c_output[0])):
                                soft_max = nn.Softmax(dim=1)(
                                    c_output[0][bin_index]
                                ).tolist()[0]
                                soft_max_sum = sum(soft_max)
                                soft_max = [x / soft_max_sum for x in soft_max]
                                curr_samples.append(
                                    np.random.choice(
                                        self.bases[0],
                                        self.num_samples_error_track,
                                        p=soft_max,
                                    )
                                )
                            curr_sample_tensor = torch.Tensor(
                                np.stack(curr_samples, axis=1)
                            )
                            float_values = utils.matrix_exponent_notation_to_float(
                                curr_sample_tensor,
                                self.bases[0],
                                self.exp_min[0],
                                self.exp_max[0],
                            )
                            avg_value = torch.mean(float_values)
                            output_soft_max.append(avg_value.item())
                        soft_mse = torch.sum(
                            (torch.Tensor(output_soft_max) - target_value) ** 2
                        ).item() / len(output_soft_max)
                        c_soft_dict_to_update[distribution_index].append(soft_mse)

                return {
                    "in_sample_hard_mean_squared_error": self.in_sample_hard_mean_squared_error,
                    "out_of_sample_hard_mean_squared_error": self.out_of_sample_hard_mean_squared_error,
                    "in_sample_soft_mean_squared_error": self.in_sample_soft_mean_squared_error,
                    "out_of_sample_soft_mean_squared_error": self.out_of_sample_soft_mean_squared_error,
                }
