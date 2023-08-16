import torch
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from ecg_segmentation_dataset import ECGDataset
from models.forward_diffusion import ForwardDiffusion


def preprocess_dataset(dataset_name: str, batch_size: int, num_workers: int):
    train_ds = ECGDataset(dataset_name, split="train")
    val_ds = ECGDataset(dataset_name, split="validation")
    test_ds = ECGDataset(dataset_name, split="test")

    # dataloaders
    train_dataloader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers)
    val_dataloader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers)
    test_dataloader = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers)

    return train_dataloader, val_dataloader, test_dataloader


if __name__ == "__main__":
    # load data
    _, _, test_dataloader = preprocess_dataset("roszcz/ecg-segmentation-ltafdb", 4, 1)

    # initialing random input
    records = next(iter(test_dataloader))
    signal = records["signal"]

    schedule_type = "sigmoid"

    # initialize forward diffusion
    fdiff = ForwardDiffusion(beta_start=0.0001, beta_end=0.02, timesteps=256, schedule_type=schedule_type)

    # timesteps for visualization
    t = torch.tensor([0, 63, 127, 191, 255], dtype=torch.long)

    signals_list = []

    for s in signal:
        # duplicate each signal t times
        x = torch.cat([s.unsqueeze(0) for _ in range(len(t))], dim=0)

        # diffusion
        diffused_x, _ = fdiff(x, t)

        diffused_list = [diffused_x_t.squeeze(0) for diffused_x_t in diffused_x]

        # signals list: [[signal_1_t1, signal_1_t2, ...], [signal_2_t1, signal_2_t2, ...], ...]
        signals_list.append(diffused_list)

    # list of images to plot [original_img, transformed_img, diffused_imgs_0, ...]
    plot_titles = [f"diffusion step: {t[i]}" for i in range(len(t))]

    fig, axes = plt.subplots(len(signals_list), len(t), figsize=(20, 10))
    fig.suptitle(f"Forward diffusion with {schedule_type} schedule", fontsize=16)

    # iterate over signals
    for i, ax_rows in enumerate(axes):
        # iterate over timesteps
        for j, ax in enumerate(ax_rows):
            # shape: [2, 1000]
            s = signals_list[i][j]

            # plot channels
            sns.lineplot(s[0], ax=ax)
            sns.lineplot(s[1], ax=ax)
            ax.set_title(plot_titles[j])

    plt.tight_layout()
    plt.show()
