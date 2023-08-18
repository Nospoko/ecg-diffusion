import torch
import seaborn as sns
import matplotlib.pyplot as plt

from sample import Generator
from train import preprocess_dataset
from models.reverse_diffusion import Unet
from models.forward_diffusion import ForwardDiffusion

if __name__ == "__main__":
    checkpoint = torch.load("checkpoints/overfit-single-batch-2023-08-17-20-25.ckpt")

    cfg = checkpoint["config"]

    # model
    model = Unet(
        in_channels=cfg.models.unet.in_out_channels,
        out_channels=cfg.models.unet.in_out_channels,
        dim=cfg.models.unet.dim,
        dim_mults=cfg.models.unet.dim_mults,
        kernel_size=cfg.models.unet.kernel_size,
        resnet_block_groups=cfg.models.unet.num_resnet_groups,
    ).to(cfg.train.device)

    # forward diffusion
    forward_diffusion = ForwardDiffusion(
        beta_start=cfg.models.forward_diffusion.beta_start,
        beta_end=cfg.models.forward_diffusion.beta_end,
        timesteps=cfg.models.forward_diffusion.timesteps,
        schedule_type=cfg.models.forward_diffusion.schedule_type,
    ).to(cfg.train.device)

    model.load_state_dict(checkpoint["model"])
    forward_diffusion.load_state_dict(checkpoint["forward_diffusion"])

    gen = Generator(model, forward_diffusion)

    batch_size = 4

    noise = torch.randn((batch_size, 2, 1000)).to(cfg.train.device)

    # if intermediate_outputs=True returns dict of intermediate signals else only denoised signal
    fake_signals = gen.sample(noise, intermediate_outputs=True)

    # idx for plotted intermediate image
    idx_to_plot = [255, 127, 63, 30, 10, 0]
    plotted_signals = [fake_signals[i].cpu() for i in idx_to_plot]

    # true training signals
    train_dataloader, _, _ = preprocess_dataset("roszcz/ecg-segmentation-ltafdb", 16, 1, overfit_single_batch=True)
    records = next(iter(train_dataloader))
    signals = records["signal"]

    # plot fake
    fig, axes = plt.subplots(batch_size, len(plotted_signals), figsize=(20, 10))

    # iterate over signals
    for i, ax_rows in enumerate(axes):
        # iterate over timesteps
        for j, ax in enumerate(ax_rows):
            s = plotted_signals[j][i]

            sns.lineplot(s[0], ax=ax)
            sns.lineplot(s[1], alpha=0.6, ax=ax)

            ax.set_title(f"Signal {i} at timestep {idx_to_plot[j]}")

    plt.tight_layout()
    plt.show()

    # plot true
    fig, axes = plt.subplots(4, 4, figsize=(20, 10))

    for i, ax in enumerate(axes.flatten()):
        sns.lineplot(signals[i, 0], ax=ax)
        sns.lineplot(signals[i, 1], alpha=0.6, ax=ax)

    plt.tight_layout()
    plt.show()
