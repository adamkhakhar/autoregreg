import torch
import torch.nn as nn
import os
import sys
import code

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(f"{ROOT_DIR}/src/training")
from Train import Train


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
        use_wandb=False,
        output_transform=None,
        target_fun=None,
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
        self.output_transform = output_transform
        self.target_fun = target_fun

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
        self.in_sample_mean_squared_error = {}
        self.out_of_sample_mean_squared_error = {}
        for i in range(self.number_targets):
            self.in_sample_mean_squared_error[i] = []
            self.out_of_sample_mean_squared_error[i] = []
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
        for i in range(self.number_targets):
            curr_output = outputs[:, i]
            curr_target = targets[:, i]
            loss += nn.MSELoss()(
                curr_output.reshape(curr_output.shape[0], 1),
                curr_target.reshape(curr_target.shape[0], 1),
            )
        if not torch.isfinite(loss):
            print("LOSS IS NAN", flush=True)
            raise Exception("LOSS IS NAN")
        return loss

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
                    "in_sample_error": self.in_sample_mean_squared_error,
                    "out_of_sample_error": self.out_of_sample_mean_squared_error,
                }
            else:
                if self.num_distributions is None:
                    self.num_distributions = max(in_sample_distribution_ind) + 1
                    for i in range(self.num_distributions):
                        self.in_sample_mean_squared_error[i] = []
                        self.out_of_sample_mean_squared_error[i] = []
                for distribution_index in range(self.num_distributions):
                    for (
                        c_orig,
                        c_output,
                        c_target,
                        distribution_ind,
                        c_dict_to_update,
                    ) in [
                        (
                            in_sample_orig_value,
                            outputs,
                            targets,
                            in_sample_distribution_ind,
                            self.in_sample_mean_squared_error,
                        ),
                        (
                            out_of_sample_orig_value,
                            out_of_sample_output,
                            out_of_sample_target,
                            out_of_sample_distribution_ind,
                            self.out_of_sample_mean_squared_error,
                        ),
                    ]:
                        mask = distribution_ind == distribution_index
                        indices = torch.nonzero(mask)
                        curr_input = c_orig[indices, 0]
                        curr_output = c_output[indices, 0]
                        curr_target = c_target[indices, 0]
                        if self.output_transform is not None:
                            curr_output = self.output_transform(curr_output.cpu())
                            curr_target = self.target_fun[distribution_index](
                                curr_input.cpu()
                            )
                        mse = nn.MSELoss()(
                            curr_output.reshape(curr_output.shape[0], 1),
                            curr_target.reshape(curr_target.shape[0], 1),
                        ).item()
                        c_dict_to_update[distribution_index].append(mse)

                return {
                    "in_sample_error": self.in_sample_mean_squared_error,
                    "out_of_sample_error": self.out_of_sample_mean_squared_error,
                }
