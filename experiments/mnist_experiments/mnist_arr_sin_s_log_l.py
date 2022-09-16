import os
import sys
import numpy as np
import torch
import argparse

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(ROOT_DIR)
sys.path.append(ROOT_DIR + "/utils")

from src.datasets.MNISTDataSet import MNISTDataSet
from src.models.CNN import CNN
from src.models.AutoRegressiveHead import AutoRegressiveHead
from src.models.EncoderDecoder import EncoderDecoder
from src.training.ARRTrain import ARRTrain
from target_functions import sin_small, log_large


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MNIST Variable Target auto regressive Feed Forward Model"
    )
    parser.add_argument("experiment_name", type=str)
    parser.add_argument("--base", dest="base", type=int, default=100)
    parser.add_argument("--exp_min", dest="exp_min", type=int, default=-2)
    parser.add_argument("--exp_max", dest="exp_max", type=int, default=2)
    parser.add_argument(
        "--num_samples", dest="num_samples", type=int, default=1_000_000
    )
    parser.add_argument("--batch_size", dest="batch_size", type=int, default=100)
    parser.add_argument("--num_workers", dest="num_workers", type=int, default=1)
    parser.add_argument(
        "--num_samples_soft_error", dest="num_samples_soft_error", type=int, default=50
    )
    parser.add_argument("--gpu_ind", dest="gpu_ind", type=int, default=-1)
    parser.add_argument("--log_every", dest="log_every", type=int, default=10_000)
    parser.add_argument("--layer_dim", dest="layer_dim", type=int, default=1024)
    parser.add_argument(
        "--seed", dest="seed", type=int, default=np.random.randint(low=1, high=1943)
    )
    parser.add_argument(
        "--learning_rate", dest="learning_rate", type=float, default=1e-3
    )
    parser.add_argument("--save_local", dest="save_local", type=str, default="F")
    parser.add_argument("--upload_to_s3", dest="upload_to_s3", type=str, default="T")
    parser.add_argument("--print_every", dest="print_every", type=str, default="F")
    args = parser.parse_args()

    # Convert boolean vars in to boolean
    assert args.save_local in ["T", "F"]
    assert args.upload_to_s3 in ["T", "F"]
    assert args.print_every in ["T", "F"]
    args.save_local = args.save_local == "T"
    args.upload_to_s3 = args.upload_to_s3 == "T"
    args.print_every = args.print_every == "T"
    print(args, flush=True)

    # set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # construct dataloader
    ### MODIFY DATA FUNCTIONS HERE ###
    target_fun = [sin_small, log_large]
    bases = [args.base] * len(target_fun)
    exp_min = [args.exp_min] * len(target_fun)
    exp_max = [args.exp_max] * len(target_fun)
    ##################################
    assert len(bases) == len(exp_min)
    assert len(bases) == len(exp_min)
    assert len(bases) == len(exp_max)

    data_loader = torch.utils.data.DataLoader(
        MNISTDataSet(
            target_fun,
            args.num_samples,
            auto_regressive=True,
            bases=bases,
            exp_min=exp_min,
            exp_max=exp_max,
        ),
        batch_size=args.batch_size,
        pin_memory=torch.cuda.is_available(),
        num_workers=args.num_workers,
    )

    # construct model
    cnn = CNN(args.layer_dim, (28 * 2) ** 2, args.layer_dim)
    number_steps = [
        exp_max[i] - exp_min[i] + 1 for i in range(len(exp_min))
    ]  # number of auto regressive steps for each target
    auto_regressive_head = AutoRegressiveHead(args.layer_dim, bases, number_steps)
    model = EncoderDecoder(cnn, auto_regressive_head)

    # construct optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # construct training
    training = ARRTrain(
        args.experiment_name,
        len(target_fun),  # number targets
        model,
        data_loader,
        optimizer,
        args.gpu_ind,
        args.log_every,
        args.num_samples // args.batch_size,  # number of gradient steps
        bases,
        exp_min,
        exp_max,
        num_samples_error_track=args.num_samples_soft_error,
        save_local=args.save_local,
        upload_to_s3=args.upload_to_s3,
        bucket_name="arr-saved-experiment-data",
        print_every=args.print_every,
    )

    # run training
    training.train()
