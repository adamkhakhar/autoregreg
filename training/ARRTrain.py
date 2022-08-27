import torch
import torch.nn as nn
import os
import sys
from typing import List
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(f"{ROOT_DIR}/utils")
import utils
from training.Train import Train


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
        num_samples_soft_error=50,
        save_local=False,
        upload_to_s3=True,
        bucket_name="arr-saved-experiment-data",
        print_every=False,
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
        self.num_samples_soft_error = num_samples_soft_error
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
        self, inputs, outputs, targets, out_of_sample_input, out_of_sample_target
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

        out_of_sample_input : tensor
            inputs of out of sample mini batch

        out_of_sample_target : tensor
            targets of out of sample mini batch

        Returns
        -------
        dict
            scalar metrics
        """
        with torch.no_grad():
            out_of_sample_output = self.model(out_of_sample_input)
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

                    # hard error
                    output_logit_arg_max = []
                    target_logit_arg_max = []
                    for bin_index in range(len(c_output[target_index])):
                        output_logit_arg_max.append(
                            torch.argmax(c_output[target_index][bin_index], dim=1).cpu()
                        )
                        target_logit_arg_max.append(
                            torch.argmax(c_target[target_index][bin_index], dim=1).cpu()
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
                                    self.num_samples_soft_error,
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