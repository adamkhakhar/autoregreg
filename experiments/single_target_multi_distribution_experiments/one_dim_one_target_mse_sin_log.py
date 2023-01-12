import os
import sys
import numpy as np
import torch
import argparse
import wandb
from pprint import pprint

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(ROOT_DIR)
sys.path.append(ROOT_DIR + "/utils")

from src.datasets.OneDimMultiDistributionSingleTarget import (
    OneDimMultiDistributionSingleTarget,
)
from src.models.FeedForward import FeedForward
from src.training.MSETrain import MSETrain
from target_functions import sin_small, sin_large, log_small, log_large


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Variable Target MSE Feed Forward Model"
    )
    parser.add_argument("experiment_name", type=str)
    parser.add_argument(
        "--num_samples", dest="num_samples", type=int, default=1_000_000
    )
    parser.add_argument("--sin", dest="sin", type=str, default="s")
    parser.add_argument("--log", dest="log", type=str, default="s")
    parser.add_argument("--batch_size", dest="batch_size", type=int, default=1_000)
    parser.add_argument("--num_workers", dest="num_workers", type=int, default=1)
    parser.add_argument("--gpu_ind", dest="gpu_ind", type=int, default=-1)
    parser.add_argument("--log_every", dest="log_every", type=int, default=1000)
    parser.add_argument("--num_layers", dest="num_layers", type=int, default=3)
    parser.add_argument("--layer_dim", dest="layer_dim", type=int, default=1024)
    parser.add_argument(
        "--seed", dest="seed", type=int, default=np.random.randint(low=1, high=1943)
    )
    parser.add_argument(
        "--learning_rate", dest="learning_rate", type=float, default=1e-3
    )
    parser.add_argument("--save_local", action="store_true")
    parser.add_argument("--upload_to_s3", action="store_true")
    parser.add_argument("--print_every", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    # Convert boolean vars in to boolean
    assert args.sin in ["s", "l"]
    assert args.log in ["s", "l"]
    args.sin_small = args.sin == "s"
    args.log_small = args.log == "s"
    pprint(vars(args))

    # set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # construct dataloader
    input_fun = [
        lambda: np.random.uniform(low=0, high=1) * -1,
        lambda: np.random.uniform(low=0, high=1),
    ]
    target_fun = [
        sin_small if args.sin_small else sin_large,
        log_small if args.log_small else log_large,
    ]
    dist_probs = [0.5, 0.5]
    data_loader = torch.utils.data.DataLoader(
        OneDimMultiDistributionSingleTarget(
            input_fun, target_fun, dist_probs, args.num_samples
        ),
        batch_size=args.batch_size,
        pin_memory=torch.cuda.is_available(),
        num_workers=args.num_workers,
    )

    # wandb setup
    if args.wandb:
        wandb.init(
            config=vars(args),
            name=args.experiment_name,
            project=os.path.basename(__file__),
        )

    # construct model
    model = FeedForward(1, 1, args.num_layers, args.layer_dim)

    # construct optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # construct training
    training = MSETrain(
        args.experiment_name,
        1,
        model,
        data_loader,
        optimizer,
        args.gpu_ind,
        args.log_every,
        args.num_samples // args.batch_size,
        save_local=args.save_local,
        upload_to_s3=args.upload_to_s3,
        bucket_name="arr-saved-experiment-data",
        print_every=args.print_every,
        use_wandb=args.wandb,
    )

    # run training
    training.train()
