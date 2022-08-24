import pickle
import boto3
import io


def float_to_exponent_notation(self, x, base, exp_min, exp_max):
    x_scaled = int(x * base ** (-1 * exp_min))
    notation = []
    for _ in range(exp_max - exp_min + 1):
        notation.append(x_scaled % base)
        x_scaled = int(x_scaled / base)
    notation.reverse()
    return notation


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
