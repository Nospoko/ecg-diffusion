import torch
import seaborn as sns
import matplotlib.pyplot as plt

from train import preprocess_dataset
from models.reverse_diffusion import Unet
from models.forward_diffusion import ForwardDiffusion

if __name__ == "__main__":
    # initializing model
    checkpoint = torch.load("checkpoints/overfit-single-batch-2023-08-17-14-59.ckpt")

    # cfg = checkpoint["config"]
    schedule_type = "cosine"

    # initialize forward diffusion
    fdiff = ForwardDiffusion(beta_start=0.0001, beta_end=0.02, timesteps=256, schedule_type=schedule_type)

    model = Unet(in_channels=2, out_channels=2, dim=32, dim_mults=(1, 2, 4), kernel_size=7, resnet_block_groups=4)

    fdiff.load_state_dict(checkpoint["forward_diffusion"])
    model.load_state_dict(checkpoint["model"])

    _, _, test_dataloader = preprocess_dataset("roszcz/ecg-segmentation-ltafdb", 4, 1)

    # initialing random input
    record = next(iter(test_dataloader))
    signal = record["signal"]
    # sample random timestep
    t = torch.randint(0, 255, size=(len(signal),))
    # mask = record["mask"]

    # diffusion
    diffused_signal, _ = fdiff(signal, t)

    # predicting noise at timestep t
    pred_noise = model(diffused_signal, t)

    print(f"Shape of signal: {signal.shape}")
    print(f"Shape of pred_noise: {pred_noise.shape}")

    # assert that shapes are the same
    assert signal.shape == pred_noise.shape, "Something is wrong with shapes"

    pred_noise = pred_noise.detach()

    # plot
    fig, axes = plt.subplots(len(signal), 3, figsize=(15, 7))
    fig.suptitle("Reverse diffusion (model not trained)", fontsize=16)

    for i, ax in enumerate(axes):
        sns.lineplot(signal[i, 0], ax=axes[i, 0])
        sns.lineplot(signal[i, 1], alpha=0.6, ax=axes[i, 0])
        axes[i, 0].set_title("Signal")

        sns.lineplot(diffused_signal[i, 0], ax=axes[i, 1])
        sns.lineplot(diffused_signal[i, 1], alpha=0.6, ax=axes[i, 1])
        axes[i, 1].set_title(f"Diffused signal at timestep {t[i]}")

        sns.lineplot(pred_noise[i, 0], ax=axes[i, 2])
        sns.lineplot(pred_noise[i, 1], alpha=0.6, ax=axes[i, 2])
        axes[i, 2].set_title(f"Predicted noise at timestep {t[i]}")

    plt.tight_layout()
    plt.show()
