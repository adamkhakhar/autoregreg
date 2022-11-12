import os
import sys
import numpy as np
import torch
import argparse

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(f"{ROOT_DIR}/src")
sys.path.append(f"{ROOT_DIR}/src/datasets")

from ReviewDataSet import ReviewDataSet
from models.Transformer import Transformer
from models.AutoRegressiveHead import AutoRegressiveHead
from models.EncoderDecoder import EncoderDecoder
from training.ARRTrain import ARRTrain

import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Amazon Review ARR Transformer Model")
    parser.add_argument("experiment_name", type=str)
    parser.add_argument("--base", dest="base", type=int, default=100)
    parser.add_argument("--exp_min", dest="exp_min", type=int, default=-2)
    parser.add_argument("--exp_max", dest="exp_max", type=int, default=3)
    parser.add_argument("--om", dest="om", type=float, default=1)
    parser.add_argument(
        "--num_samples", dest="num_samples", type=int, default=1_000_000
    )
    parser.add_argument("--batch_size", dest="batch_size", type=int, default=512)
    parser.add_argument("--num_workers", dest="num_workers", type=int, default=0)
    parser.add_argument(
        "--num_samples_soft_error", dest="num_samples_soft_error", type=int, default=50
    )
    parser.add_argument("--gpu_ind", dest="gpu_ind", type=int, default=-1)
    parser.add_argument("--layer_dim", dest="layer_dim", type=int, default=512)
    parser.add_argument("--ntoken", dest="ntoken", type=int, default=10_002)
    parser.add_argument("--input_size", dest="input_size", type=int, default=500)
    parser.add_argument("--d_model", dest="d_model", type=int, default=128)
    parser.add_argument("--nhead", dest="nhead", type=int, default=4)
    parser.add_argument(
        "--nlayers_encoder", dest="nlayers_encoder", type=int, default=4
    )
    parser.add_argument(
        "--nlayers_decoder", dest="nlayers_decoder", type=int, default=3
    )
    parser.add_argument(
        "--seed", dest="seed", type=int, default=np.random.randint(low=1, high=1943)
    )
    parser.add_argument(
        "--learning_rate", dest="learning_rate", type=float, default=1e-3
    )
    parser.add_argument("--log_every", dest="log_every", type=int, default=100)
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
    args.experiment_name += (
        f"_arr_om_{args.om}_lr_{args.learning_rate}_seed_{args.seed}"
    )
    print(args, flush=True)

    # set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # construct dataloader
    target_fun = [lambda x: x["rating"] / 5, lambda x: x["num_char"] * args.om]

    bases = [args.base] * len(target_fun)
    exp_min = [args.exp_min] * len(target_fun)
    exp_max = [args.exp_max] * len(target_fun)
    ##################################
    assert len(bases) == len(exp_min)
    assert len(bases) == len(exp_min)
    assert len(bases) == len(exp_max)

    data_loader = torch.utils.data.DataLoader(
        ReviewDataSet(
            target_fun,
            num_samples=args.num_samples,
            bucket_name="review-shards",
            max_key=26_000,
            start_token=args.ntoken - 1,
            num_samples_per_shard=1_024,
            clip_input_size=args.input_size,
            arr={"base": bases, "exp_min": exp_min, "exp_max": exp_max},
        ),
        batch_size=args.batch_size,
        pin_memory=torch.cuda.is_available(),
        num_workers=args.num_workers,
    )

    # construct model
    number_steps = [
        exp_max[i] - exp_min[i] + 1 for i in range(len(exp_min))
    ]  # number of auto regressive steps for each target
    transformer = Transformer(
        args.ntoken,
        args.input_size,
        args.d_model,
        args.nhead,
        args.layer_dim,
        args.nlayers_encoder,
        args.nlayers_decoder,
        args.layer_dim,
    )
    auto_regressive_head = AutoRegressiveHead(args.layer_dim, bases, number_steps)
    model = EncoderDecoder(transformer, auto_regressive_head)

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
