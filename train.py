import os

import hydra
import torch
import wandb
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
from omegaconf import OmegaConf
from huggingface_hub import upload_file
from torch.utils.data import Subset, DataLoader

from models.reverse_diffusion import Unet
from ecg_segmentation_dataset import ECGDataset
from models.forward_diffusion import ForwardDiffusion


def makedir_if_not_exists(dir: str):
    if not os.path.exists(dir):
        os.makedirs(dir)


def preprocess_dataset(dataset_name: str, batch_size: int, num_workers: int, *, overfit_single_batch: bool = False):
    train_ds = ECGDataset(dataset_name, split="train")
    val_ds = ECGDataset(dataset_name, split="validation")
    test_ds = ECGDataset(dataset_name, split="test")

    if overfit_single_batch:
        train_ds = Subset(train_ds, indices=range(batch_size))
        val_ds = Subset(val_ds, indices=range(batch_size))
        test_ds = Subset(test_ds, indices=range(batch_size))

    # dataloaders
    train_dataloader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers)
    val_dataloader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers)
    test_dataloader = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers)

    return train_dataloader, val_dataloader, test_dataloader


def forward_step(
    model: Unet,
    forward_diffusion: ForwardDiffusion,
    batch: dict[str, torch.Tensor, torch.Tensor],
    device: torch.device,
) -> float:
    x = batch["signal"].to(device)

    batch_size = x.shape[0]

    # sample t
    t = torch.randint(0, forward_diffusion.timesteps, size=(batch_size,), dtype=torch.long, device=device)

    # noise batch
    x_noisy, added_noise = forward_diffusion(x, t)

    # get predicted noise
    predicted_noise = model(x_noisy, t)

    # get loss value for batch
    loss = F.mse_loss(predicted_noise, added_noise)

    return loss


def save_checkpoint(model: Unet, forward_diffusion: ForwardDiffusion, optimizer: optim.Optimizer, cfg: OmegaConf, save_path: str):
    # saving models
    torch.save(
        {
            "model": model.state_dict(),
            "forward_diffusion": forward_diffusion.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": cfg,
        },
        f=save_path,
    )


def upload_to_huggingface(ckpt_save_path: str, cfg: OmegaConf):
    # get huggingface token from environment variables
    token = os.environ["HUGGINGFACE_TOKEN"]

    # upload model to hugging face
    upload_file(ckpt_save_path, path_in_repo=f"{cfg.logger.run_name}.ckpt", repo_id=cfg.paths.hf_repo_id, token=token)


@hydra.main(config_path="configs", config_name="config-default", version_base="1.3.2")
def train(cfg: OmegaConf):
    # create dir if they don't exist
    makedir_if_not_exists(cfg.paths.log_dir)
    makedir_if_not_exists(cfg.paths.save_ckpt_dir)

    # dataset
    train_dataloader, val_dataloader, _ = preprocess_dataset(
        dataset_name=cfg.train.dataset_name,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        overfit_single_batch=cfg.train.overfit_single_batch,
    )

    # logger
    wandb.init(project="ecg-diffusion", name=cfg.logger.run_name, dir=cfg.paths.log_dir)

    device = torch.device(cfg.train.device)

    # model
    unet = Unet(
        in_channels=cfg.models.unet.in_out_channels,
        out_channels=cfg.models.unet.in_out_channels,
        dim=cfg.models.unet.dim,
        dim_mults=cfg.models.unet.dim_mults,
        kernel_size=cfg.models.unet.kernel_size,
        resnet_block_groups=cfg.models.unet.num_resnet_groups,
    ).to(device)

    # forward diffusion
    forward_diffusion = ForwardDiffusion(
        beta_start=cfg.models.forward_diffusion.beta_start,
        beta_end=cfg.models.forward_diffusion.beta_end,
        timesteps=cfg.models.forward_diffusion.timesteps,
        schedule_type=cfg.models.forward_diffusion.schedule_type,
    ).to(device)

    # setting up optimizer
    optimizer = optim.AdamW(unet.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    # load checkpoint if specified in cfg
    if cfg.paths.load_ckpt_path is not None:
        checkpoint = torch.load(cfg.paths.load_ckpt_path)

        unet.load_state_dict(checkpoint["model"])
        forward_diffusion.load_state_dict(checkpoint["forward_diffusion"])
        optimizer.load_state_dict(checkpoint["optimizer"])

    # ckpt specifies directory and name of the file is name of the experiment in wandb
    save_path = f"{cfg.paths.save_ckpt_dir}/{cfg.logger.run_name}.ckpt"

    # step counts for logging to wandb
    step_count = 0

    for epoch in range(cfg.train.num_epochs):
        # train epoch
        unet.train()
        train_loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader))

        for batch_idx, batch in train_loop:
            # metrics returns loss and additional metrics if specified in step function
            loss = forward_step(unet, forward_diffusion, batch, device)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loop.set_postfix(loss=loss.item())

            step_count += 1

            if (batch_idx + 1) % cfg.logger.log_every_n_steps == 0:
                # log metrics
                wandb.log({"train/loss": loss.item()}, step=step_count)

                # save model and optimizer states
                save_checkpoint(unet, forward_diffusion, optimizer, cfg, save_path=save_path)

        # val epoch
        unet.eval()
        val_loop = tqdm(enumerate(val_dataloader), total=len(val_dataloader))
        loss_epoch = 0.0
        
        with torch.no_grad():
            for batch_idx, batch in val_loop:
                # metrics returns loss and additional metrics if specified in step function
                loss = forward_step(unet, forward_diffusion, batch, device)

                val_loop.set_postfix(loss=loss.item())

                loss_epoch += loss.item()

        wandb.log({"val/loss_epoch": loss_epoch / len(val_dataloader)}, step=epoch)

    # save model at the end of training
    save_checkpoint(unet, forward_diffusion, optimizer, cfg, save_path=save_path)

    wandb.finish()

    # upload model to huggingface if specified in cfg
    if cfg.paths.hf_repo_id is not None:
        upload_to_huggingface(save_path, cfg)


if __name__ == "__main__":
    wandb.login()

    train()
