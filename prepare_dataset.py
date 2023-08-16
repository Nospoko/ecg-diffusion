import os
import math
from glob import glob

import wfdb
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.ndimage import binary_dilation
from datasets import Value, Array2D, Dataset, Features, DatasetDict


def create_save_path(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def ltafdb_paths(folder: str) -> list[str]:
    # Get all the signal headers
    query = os.path.join(folder, "*.hea")
    paths = glob(query)

    # Get rid of the extension
    paths = [path[:-4] for path in paths]

    return paths


def load_ltafdb_record(record_path: str):
    ann = wfdb.rdann(record_path, "atr")
    # Convert to a convenient dataframe
    adf = pd.DataFrame({"symbol": ann.symbol, "aux": ann.aux_note, "position": ann.sample})

    signals, fields = wfdb.rdsamp(record_path)

    return signals, adf, fields


def create_mask(data: pd.DataFrame, dilation: int):
    condition = data["symbol"].isin(["N", "V", "A"])

    mask = np.where(binary_dilation(condition, iterations=dilation), 1, 0)

    return mask


def process_file(
    path: str, sequence_window: int = 1000, area_around_beat_ms: float = 100
) -> list[dict[str, np.ndarray, np.ndarray]]:
    # list that will contain of subsets of size sequence_window
    records = []

    signals, adf, fields = load_ltafdb_record(path)
    # get file name
    filename = path.split("/")[-1]

    # extract sampling rate
    fs = fields["fs"]
    # calculating dilation to get fields that are part of the beat
    # area_around_beat_ms is value in miliseconds
    dilation = int((area_around_beat_ms * fs / 1000) // 2)

    # create dataframe from signals
    signals_df = pd.DataFrame(signals, columns=["channel_1", "channel_2"])
    # set beat position as index, it will be used to merge signals and adf
    adf.set_index("position", inplace=True)

    data = pd.merge(signals_df, adf, how="left", right_index=True, left_index=True)

    # add mask column
    data["mask"] = create_mask(data, dilation=dilation)
    data = data[["channel_1", "channel_2", "mask"]]

    for subset in tqdm(data.rolling(window=sequence_window, step=sequence_window), desc=f"Processing {filename} file"):
        # rolling sometimes creates subsets with shorter sequence length, they are filtered here
        if len(subset) != sequence_window:
            continue

        # record_id = f"file-{filename}-indices-{subset.index[0]}-{subset.index[-1]}.csv"

        record = {
            "record_id": filename,
            # shape: [2, sequence_window]
            "signal": subset[["channel_1", "channel_2"]].astype("float32").values.T,
            # shape: [1, sequence_window]
            "mask": subset[["mask"]].astype("int8").values.T,
        }

        records.append(record)

    return records


def split_in_two(filenames: list[str], split_ratio: float = 0.8) -> tuple[list[str], list[str]]:
    ds_length = len(filenames)

    split = math.ceil(split_ratio * ds_length)

    return filenames[:split], filenames[split:]


if __name__ == "__main__":
    # get huggingface token from environment variables
    token = os.environ["HUGGINGFACE_TOKEN"]

    # hyperparameters
    sequence_window = 1000
    area_around_beat_ms = 100

    # That's where I'm downloading the LTAFDB data
    folder = "physionet.org/files/ltafdb/1.0.0/"
    paths = ltafdb_paths(folder)[0:1]

    # get name of file, this will be used for split
    filenames = [path.split("/")[-1] for path in paths]

    # creating split for train, val, test
    train_filenames, val_test_filenames = split_in_two(filenames, split_ratio=0.8)
    val_filenames, test_filenames = split_in_two(val_test_filenames, split_ratio=0.5)

    train_records = []
    val_records = []
    test_records = []

    for record_path in paths:
        filename = record_path.split("/")[-1]

        records = process_file(path=record_path, sequence_window=sequence_window, area_around_beat_ms=area_around_beat_ms)

        if filename in train_filenames:
            train_records += records
        elif filename in val_filenames:
            val_records += records
        elif filename in test_filenames:
            test_records += records

    # building huggingface dataset
    features = Features(
        {
            "record_id": Value(dtype="string"),
            "signal": Array2D(dtype="float32", shape=(2, sequence_window)),
            "mask": Array2D(dtype="int8", shape=(1, sequence_window)),
        }
    )

    # dataset = Dataset.from_list(records, features=features)
    dataset = DatasetDict(
        {
            "train": Dataset.from_list(train_records, features=features),
            "validation": Dataset.from_list(val_records, features=features),
            "test": Dataset.from_list(test_records, features=features),
        }
    )

    # print(dataset["train"])
    dataset.push_to_hub("JasiekKaczmarczyk/physionet-preprocessed", token=token)
