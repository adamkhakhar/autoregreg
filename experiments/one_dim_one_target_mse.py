import os
import sys
import numpy as np
import torch
import argparse

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(ROOT_DIR)

from datasets.OneDimDataSet import OneDimDataSet
from models.FeedForward import FeedForward
from training.MSETrain import MSETrain


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Variable Target MSE Feed Forward Model"
    )
    parser.add_argument("experiment_name", type=str)
    parser.add_argument(
        "--num_samples", dest="num_samples", type=int, default=1_000_000
    )
    parser.add_argument("--batch_size", dest="batch_size", type=int, default=1_000)
    parser.add_argument("--gpu_ind", dest="gpu_ind", type=int, default=-1)
    parser.add_argument("--log_every", dest="log_every", type=int, default=1000)
    parser.add_argument("--num_layers", dest="num_layers", type=int, default=3)
    parser.add_argument("--layer_dim", dest="layer_dim", type=int, default=1024)
    parser.add_argument(
        "--learning_rate", dest="learning_rate", type=float, default=1e-3
    )
    parser.add_argument("--upload_to_s3", dest="upload_to_s3", type=str, default="T")
    parser.add_argument("--print_every", dest="print_every", type=str, default="F")
    args = parser.parse_args()

    # Convert boolean vars in to boolean
    assert args.upload_to_s3 in ["T", "F"]
    assert args.print_every in ["T", "F"]
    args.upload_to_s3 = args.upload_to_s3 == "T"
    args.print_every = args.print_every == "T"
    print(args, flush=True)

    # construct dataloader
    input_fun = [lambda: np.random.uniform(low=0, high=1)]
    target_fun = [lambda x: x]
    data_loader = torch.utils.data.DataLoader(
        OneDimDataSet(input_fun, target_fun, args.num_samples),
        batch_size=args.batch_size,
        pin_memory=torch.cuda.is_available(),
    )

    # construct model
    model = FeedForward(1, len(input_fun), args.num_layers, args.layer_dim)

    # construct optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # construct training
    training = MSETrain(
        args.experiment_name,
        len(input_fun),
        model,
        data_loader,
        optimizer,
        args.gpu_ind,
        args.log_every,
        args.num_samples // args.batch_size,
        upload_to_s3=args.upload_to_s3,
        bucket_name="arr-saved-experiment-data",
        print_every=args.print_every,
    )

    # run training
    training.train()
