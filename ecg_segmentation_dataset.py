import torch
from torch.utils.data import Dataset
from datasets import Value, Array2D, Features, load_dataset


class ECGDataset(Dataset):
    def __init__(self, huggingface_path: str, split: str = "train"):
        super().__init__()

        features = Features(
            {
                "record_id": Value(dtype="string"),
                "signal": Array2D(dtype="float32", shape=(2, 1000)),
                "mask": Array2D(dtype="int8", shape=(1, 1000)),
            }
        )
        self.data = load_dataset(huggingface_path, features=features, split=split)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.data[index]

        # wrap signal and mask to torch.Tensor
        signal = torch.tensor(record["signal"], dtype=torch.float32)
        mask = torch.tensor(record["mask"], dtype=torch.float32)

        item = {
            "record_id": record["record_id"],
            "signal": signal,
            "mask": mask,
        }

        return item
