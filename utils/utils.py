import pickle
import boto3
import io
import torch


def float_to_exponent_notation(x, base, exp_min, exp_max):
    x_scaled = int(x * base ** (-1 * exp_min))
    notation = []
    for _ in range(exp_max - exp_min + 1):
        notation.append(x_scaled % base)
        x_scaled = int(x_scaled / base)
    notation.reverse()
    return notation


def exponent_notation_to_float(l: torch.Tensor, base: int, exp_min: int, exp_max: int):
    assert len(l) == exp_max - exp_min + 1
    x = 0
    curr_exp = exp_max
    for i in range(len(l)):
        x += l[i].item() * base ** (curr_exp)
        curr_exp -= 1
    return x


def matrix_exponent_notation_to_float(
    m: torch.Tensor, base: int, exp_min: int, exp_max: int
):
    exponent_values = torch.stack(
        [torch.Tensor([base**e for e in range(exp_max, exp_min - 1, -1)])]
        * m.shape[0]
    )
    pre_sum_matrix = m * exponent_values
    values = torch.sum(pre_sum_matrix, dim=1)
    return values


def store_data(fname, data):
    with open(fname, "wb") as f:
        pickle.dump(data, f)


def load_data(fname):
    with open(fname, "rb") as f:
        data = pickle.load(f)
    return data


def upload_to_s3(bucket_name, key, data):
    bytes = io.BytesIO()
    pickle.dump(data, bytes)
    bytes.seek(0)
    boto3.client("s3").upload_fileobj(bytes, bucket_name, key)


def pull_from_s3(key, bucket_name="arr-saved-experiment-data"):
    b = io.BytesIO()
    boto3.client("s3").download_fileobj(bucket_name, key, b)
    b.seek(0)
    data = pickle.load(b)
    return data
