import os
import sys
import numpy as np
import torch
import argparse

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(f"{ROOT_DIR}/src")
sys.path.append(f"{ROOT_DIR}/src/datasets")

from TestTransformerDataSet import TestTransformerDataSet
from models.Transformer import Transformer
from training.MSETrain import MSETrain

import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Toy Data MSE Transformer Model")
    parser.add_argument("experiment_name", type=str)
    parser.add_argument(
        "--num_samples", dest="num_samples", type=int, default=1_000_000
    )
    parser.add_argument("--batch_size", dest="batch_size", type=int, default=1_000)
    parser.add_argument("--num_workers", dest="num_workers", type=int, default=1)
    parser.add_argument("--gpu_ind", dest="gpu_ind", type=int, default=-1)
    parser.add_argument("--layer_dim", dest="layer_dim", type=int, default=1024)
    parser.add_argument("--ntoken", dest="ntoken", type=int, default=100)
    parser.add_argument("--input_size", dest="input_size", type=int, default=10)
    parser.add_argument("--d_model", dest="d_model", type=int, default=128)
    parser.add_argument("--nhead", dest="nhead", type=int, default=4)
    parser.add_argument(
        "--nlayers_encoder", dest="nlayers_encoder", type=int, default=4
    )
    parser.add_argument(
        "--nlayers_decoder", dest="nlayers_decoder", type=int, default=4
    )
    parser.add_argument(
        "--seed", dest="seed", type=int, default=np.random.randint(low=1, high=1943)
    )
    parser.add_argument(
        "--learning_rate", dest="learning_rate", type=float, default=1e-3
    )
    parser.add_argument("--log_every", dest="log_every", type=int, default=10_000)
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

    input_fun = lambda x: np.random.randint(
        low=0, high=args.ntoken - 1, size=args.input_size
    )
    target_fun = [lambda x: np.sum(x), lambda x: len(x)]
    data_loader = torch.utils.data.DataLoader(
        TestTransformerDataSet(input_fun, target_fun, args.num_samples),
        batch_size=args.batch_size,
        pin_memory=torch.cuda.is_available(),
        num_workers=args.num_workers,
    )

    # construct model
    model = Transformer(
        args.ntoken,
        args.input_size,
        args.d_model,
        args.nhead,
        args.layer_dim,
        args.nlayers_encoder,
        args.nlayers_decoder,
        len(target_fun),
    )

    # construct optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # construct training
    training = MSETrain(
        args.experiment_name,
        len(target_fun),
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
    )

    # run training
    training.train()
