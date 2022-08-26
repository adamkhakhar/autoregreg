import torch
import torch.nn as nn
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(f"{ROOT_DIR}/utils")
from training.Train import Train


class MSETrain(Train):
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
        self.in_sample_mean_squared_error = {}
        self.out_of_sample_mean_squared_error = {}
        for i in range(self.number_targets):
            self.in_sample_mean_squared_error[i] = []
            self.out_of_sample_mean_squared_error[i] = []

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
        for i in range(self.number_targets):
            curr_output = outputs[:, i]
            curr_target = targets[:, i]
            loss += nn.MSELoss()(
                curr_output.reshape(curr_output.shape[0], 1),
                curr_target.reshape(curr_target.shape[0], 1),
            )
        return loss

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
            for i in range(self.number_targets):
                for (c_output, c_target, c_dict_to_update) in [
                    (outputs, targets, self.in_sample_mean_squared_error),
                    (
                        out_of_sample_output,
                        out_of_sample_target,
                        self.out_of_sample_mean_squared_error,
                    ),
                ]:
                    curr_output = c_output[:, i]
                    curr_target = c_target[:, i]
                    mse = nn.MSELoss()(
                        curr_output.reshape(curr_output.shape[0], 1),
                        curr_target.reshape(curr_target.shape[0], 1),
                    ).item()
                    c_dict_to_update[i].append(mse)

            return {
                "in_sample_mean_squared_error": self.in_sample_mean_squared_error,
                "out_of_sample_mean_squared_error": self.out_of_sample_mean_squared_error,
            }
