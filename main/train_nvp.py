import copy
import logging
import os

import torch
import hydra
import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.seed import seed_everything
from torch.utils.data import DataLoader

from models.callbacks import EMAWeightUpdate
from models.diffusion import DDPM, DDPMv2, DDPMWrapper, SuperResModel, UNetModel
from models.vae import VAE
from realnvp import RealNVP  # Import RealNVP for flow-based prior
from util import configure_device, get_dataset

logger = logging.getLogger(__name__)

def __parse_str(s):
    split = s.split(",")
    return [int(s) for s in split if s != "" and s is not None]

class Hyperparameters:
    def __init__(
        self,
        base_dim,
        res_blocks,
        bottleneck,
        skip,
        weight_norm,
        coupling_bn,
        affine,
    ):
        self.base_dim = base_dim
        self.res_blocks = res_blocks
        self.bottleneck = bottleneck
        self.skip = skip
        self.weight_norm = weight_norm
        self.coupling_bn = coupling_bn
        self.affine = affine
        
class DataInfo:
    def __init__(self, channel, size, name):
        self.channel = channel
        self.size = size
        self.name = name  # name 속성 추가


@hydra.main(config_path="configs", version_base=None)
def train(config):
    # Get config and setup
    config = config.dataset.ddpm
    logger.info(OmegaConf.to_yaml(config))

    # Set seed
    seed_everything(config.training.seed, workers=True)

    # Dataset
    root = config.data.root
    d_type = config.data.name
    image_size = config.data.image_size
    dataset = get_dataset(
        d_type, root, image_size, norm=config.data.norm, flip=config.data.hflip
    )
    
    N = len(dataset)
    batch_size = config.training.batch_size
    batch_size = min(N, batch_size)

    # Model
    lr = config.training.lr
    attn_resolutions = __parse_str(config.model.attn_resolutions)
    dim_mults = __parse_str(config.model.dim_mults)
    ddpm_type = config.training.type

    # Use the superres model for conditional training
    decoder_cls = UNetModel if ddpm_type == "uncond" else SuperResModel
    decoder = decoder_cls(
        in_channels=config.data.n_channels,
        model_channels=config.model.dim,
        out_channels=3,
        num_res_blocks=config.model.n_residual,
        attention_resolutions=attn_resolutions,
        channel_mult=dim_mults,
        use_checkpoint=False,
        dropout=config.model.dropout,
        num_heads=config.model.n_heads,
        z_dim=config.training.z_dim,
        use_scale_shift_norm=config.training.z_cond,
        use_z=config.training.z_cond,
    )

    # EMA parameters are non-trainable
    ema_decoder = copy.deepcopy(decoder)
    for p in ema_decoder.parameters():
        p.requires_grad = False

    ddpm_cls = DDPMv2 if ddpm_type == "form2" else DDPM
    online_ddpm = ddpm_cls(
        decoder,
        beta_1=config.model.beta1,
        beta_2=config.model.beta2,
        T=config.model.n_timesteps,
    )
    target_ddpm = ddpm_cls(
        ema_decoder,
        beta_1=config.model.beta1,
        beta_2=config.model.beta2,
        T=config.model.n_timesteps,
    )

    # VAE setup with RealNVP for flow-based prior
    vae = VAE.load_from_checkpoint(
        config.training.vae_chkpt_path,
        input_res=image_size,
    )
    vae.train()

    # Create Hyperparameters object
    hps = Hyperparameters(
        base_dim=config.training.flow_base_dim,
        res_blocks=config.training.flow_res_blocks,
        bottleneck=config.training.flow_bottleneck,
        skip=config.training.flow_skip,
        weight_norm=config.training.flow_weight_norm,
        coupling_bn=config.training.flow_coupling_bn,
        affine=config.training.flow_affine,
    )
    
    # Create DataInfo object
    datainfo = DataInfo(
        channel=512,
        size=image_size,
        name=config.data.name
    )

    flow_prior = RealNVP(  # Configure RealNVP as prior
        datainfo=datainfo,
        prior=torch.distributions.Normal(torch.tensor(0.0), torch.tensor(1.0)),
        hps=hps,
    )

    vae.set_prior(flow_prior)  # Attach flow-based prior to VAE

    for p in vae.parameters():
        p.requires_grad = True

    assert isinstance(online_ddpm, ddpm_cls)
    assert isinstance(target_ddpm, ddpm_cls)
    logger.info(f"Using DDPM with type: {ddpm_cls} and data norm: {config.data.norm}")

    ddpm_wrapper = DDPMWrapper(
        online_ddpm,
        target_ddpm,
        vae,
        lr=lr,
        cfd_rate=config.training.cfd_rate,
        n_anneal_steps=config.training.n_anneal_steps,
        loss=config.training.loss,
        conditional=False if ddpm_type == "uncond" else True,
        grad_clip_val=config.training.grad_clip,
        z_cond=config.training.z_cond,
        vae_weight=config.training.vae_weight,
        ddpm_weight=config.training.ddpm_weight,
        anneal_start=config.training.anneal_start,
        anneal_end=config.training.anneal_end,
        adaptive_conditioning=True,  # Enable adaptive conditioning
    )

    # Trainer
    train_kwargs = {}
    restore_path = config.training.restore_path
    if restore_path != "":
        train_kwargs["ckpt_path"] = restore_path

    # Setup callbacks
    results_dir = config.training.results_dir
    chkpt_callback = ModelCheckpoint(
        dirpath=os.path.join(results_dir, "checkpoints"),
        filename=f"ddpmv2-{config.training.chkpt_prefix}" + "-{epoch:02d}-{loss:.4f}",
        every_n_epochs=config.training.chkpt_interval,
        save_on_train_epoch_end=True,
    )

    ema_chkpt_callback = ModelCheckpoint(
        dirpath=os.path.join(results_dir, "ema_checkpoints"),
        filename="ema-{epoch:02d}-{loss:.4f}",
        every_n_epochs=10,
        save_top_k=-1,
    )

    train_kwargs["default_root_dir"] = results_dir
    train_kwargs["max_epochs"] = config.training.epochs
    train_kwargs["log_every_n_steps"] = config.training.log_step
    train_kwargs["callbacks"] = [chkpt_callback, ema_chkpt_callback]

    if config.training.use_ema:
        ema_callback = EMAWeightUpdate(tau=config.training.ema_decay)
        train_kwargs["callbacks"].append(ema_callback)

    device = config.training.device
    loader_kws = {}
    if device.startswith("gpu"):
        _, devs = configure_device(device)
        train_kwargs["gpus"] = devs

        from pytorch_lightning.plugins import DDPPlugin
        train_kwargs["plugins"] = DDPPlugin(find_unused_parameters=False)
        loader_kws["persistent_workers"] = True
    elif device == "tpu":
        train_kwargs["tpu_cores"] = 8

    if config.training.fp16:
        train_kwargs["precision"] = 16

    loader = DataLoader(
        dataset,
        batch_size,
        num_workers=config.training.workers,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
        **loader_kws,
    )

    logger.info(f"Running Trainer with kwargs: {train_kwargs}")
    trainer = pl.Trainer(**train_kwargs)
    trainer.fit(ddpm_wrapper, train_dataloader=loader)

if __name__ == "__main__":
    train()
