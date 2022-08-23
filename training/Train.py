import time
import torch
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(f"{ROOT_DIR}/utils")
import utils.utils as utils


class Train:
    """
    Class used to implement the training of a neural network.

    ...
    Methods
    -------
    calculate_loss(outputs, targets):
        Calculates loss of a minibatch
    compute_mini_batch_metrics(
        inputs, outputs, targets, out_of_sample_input, out_of_sample_target
    ):
        Compute metrics every minibatch used in iteration_update.
    iteration_update(i, features, outputs, targets, loss):
        Is called every minibatch. Used for logging.
    save_state(loss):
        Saves state of model
    train():
        Trains model using base_model_runner train method
    """

    def __init__(
        self,
        experiment_name,
        model,
        data_loader,
        optimizer,
        device,
        log_every,
        num_grad_steps,
        upload_to_s3=True,
        bucket_name="arr-saved-experiment-data",
        print_every=False,
    ):
        # variables from parameters
        self.experiment_name = experiment_name
        self.model = model
        self.data_loader = (data_loader,)
        self.optimizer = optimizer
        self.device = device
        self.log_every = log_every
        self.num_grad_steps = num_grad_steps
        self.print_every = print_every
        self.upload_to_s3 = upload_to_s3
        self.bucket_name = bucket_name

        # variables used during execution
        self.curr_loss = 0
        self.train_loss_lst = []
        self.start_time = None
        self.train_time_lst = []

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
        raise NotImplementedError

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
        raise NotImplementedError

    def save_state(self, loss, model, optimizer):
        """Saves state of model to s3

        Parameters
        ----------
        loss : scalar tensor
            current loss of model

        model : nn.Module
            PyTorch model

        optimizer : torch.optim
            model optimizer
        """
        experiment_data = {
            "loss": loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        if self.upload_to_s3:
            utils.upload_to_s3(
                self.bucket_name,
                f"{self.experiment_name}_experiment_data.bin",
                experiment_data,
            )

    def iteration_update(self, i, inputs, outputs, targets, loss):
        """Is called every minibatch. Used for logging.

        Parameters
        ----------
        i : int
            minibatch iteration

        inputs : tensor
            tensor of inputs

        outputs : tensor
            prediction of model

        targets : tensor
            correct answer for model's predictions

        loss : tensor
            current loss of model
        """
        with torch.no_grad():
            self.curr_loss += loss.item()

            # save current data state and compute metrics
            if i % self.log_every == 0 and i != 0:
                scaled_curr_loss = self.curr_loss / self.log_every
                self.train_loss_lst.append(scaled_curr_loss)
                self.train_time_lst.append(time.time() - self.start_time)
                out_of_sample_data = next(iter(self.data_loader))
                out_of_sample_input = out_of_sample_data["input"].to(self.device)
                out_of_sample_target = out_of_sample_data["target"].to(self.device)

                # compute metrics
                mini_batch_metrics = self.compute_mini_batch_metrics(
                    inputs,
                    outputs,
                    targets,
                    out_of_sample_input,
                    out_of_sample_target,
                )

                # save and print metrics
                if self.upload_to_s3:
                    utils.upload_to_s3(
                        self.bucket_name,
                        f"{self.experiment_name}_mini_batch_metrics.bin",
                        mini_batch_metrics,
                    )
                print(
                    f"[{i} / {self.num_grad_steps}] Train Loss: {scaled_curr_loss} | Time {int(time.time() - self.start_time)}"
                )
                self.curr_loss = 0

    def train(self):
        """
        Trains model
        """
        self.start_time = time.time()
        for i, data in enumerate(self.data_loader, 0):
            if self.print_every:
                print(i, int(time.time() - self.start_time), flush=True)
            inputs = data["input"].to(self.device)
            targets = data["target"].to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.calculate_loss(outputs, targets)
            loss.backward()
            self.optimizer.step()
            with torch.no_grad():
                self.iteration_update(
                    i, inputs, outputs, targets, loss, self.data_loader, self.model
                )
        if self.upload_to_s3:
            self.save_state(loss)
