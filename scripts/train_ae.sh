# # CelebAMaskHQ training
# python main/train_ae.py +dataset=celebamaskhq128/train \
#                      dataset.vae.data.root='/data1/kushagrap20/datasets/CelebAMask-HQ/' \
#                      dataset.vae.data.name='celebamaskhq' \
#                      dataset.vae.data.hflip=True \
#                      dataset.vae.training.batch_size=42 \
#                      dataset.vae.training.log_step=50 \
#                      dataset.vae.training.epochs=500 \
#                      dataset.vae.training.device=\'gpu:0,1,3\' \
#                      dataset.vae.training.results_dir=\'/data1/kushagrap20/vae_cmhq128_alpha=1.0/\' \
#                      dataset.vae.training.workers=2 \
#                      dataset.vae.training.chkpt_prefix=\'cmhq128_alpha=1.0\' \
#                      dataset.vae.training.alpha=1.0

# # FFHQ 128 training
# python main/train_ae.py +dataset=ffhq/train \
#                      dataset.vae.data.root='/data1/kushagrap20/datasets/ffhq/' \
#                      dataset.vae.data.name='ffhq' \
#                      dataset.vae.data.hflip=True \
#                      dataset.vae.training.batch_size=32 \
#                      dataset.vae.training.log_step=50 \
#                      dataset.vae.training.epochs=1500 \
#                      dataset.vae.training.device=\'gpu:0,1,2,3\' \
#                      dataset.vae.training.results_dir=\'/data1/kushagrap20/vae_ffhq128_11thJune_alpha=1.0/\' \
#                      dataset.vae.training.workers=2 \
#                      dataset.vae.training.chkpt_prefix=\'ffhq128_11thJune_alpha=1.0\' \
#                      dataset.vae.training.alpha=1.0

# # AFHQv2 training
# python main/train_ae.py +dataset=afhq256/train \
#                      dataset.vae.data.root='/data1/kushagrap20/datasets/afhq_v2/' \
#                      dataset.vae.data.name='afhq' \
#                      dataset.vae.training.batch_size=8 \
#                      dataset.vae.training.epochs=500 \
#                      dataset.vae.training.device=\'gpu:0,1,2,3\' \
#                      dataset.vae.training.results_dir=\'/data1/kushagrap20/vae_afhq256_10thJuly_alpha=1.0/\' \
#                      dataset.vae.training.workers=2 \
#                      dataset.vae.training.chkpt_prefix=\'afhq256_10thJuly_alpha=1.0\' \
#                      dataset.vae.training.alpha=1.0


# # CelebA training
# python main/train_ae.py +dataset=celeba64/train \
#                      dataset.vae.data.root='/data1/kushagrap20/datasets/img_align_celeba/' \
#                      dataset.vae.data.name='celeba' \
#                      dataset.vae.training.batch_size=32 \
#                      dataset.vae.training.epochs=1500 \
#                      dataset.vae.training.device=\'gpu:0,1,2,3\' \
#                      dataset.vae.training.results_dir=\'/data1/kushagrap20/vae_celeba64_alpha=1.0/\' \
#                      dataset.vae.training.workers=4 \
#                      dataset.vae.training.chkpt_prefix=\'celeba64_alpha=1.0\' \
#                      dataset.vae.training.alpha=1.0

# # CelebA training
# python main/train_ae_nvp.py +dataset=cifar10/train \
#                      dataset.vae.data.root='/data/jupyter/yeon/DiffuseVAE/main/datasets/' \
#                      dataset.vae.data.name='cifar10' \
#                      dataset.vae.training.batch_size=32 \
#                      dataset.vae.training.epochs=1500 \
#                      dataset.vae.training.device=\'gpu:0\' \
#                      dataset.vae.training.results_dir=\'/data/jupyter/yeon/DiffuseVAE/vae_nvp/\' \
#                      dataset.vae.training.workers=4 \
#                      dataset.vae.training.chkpt_prefix=\'cifar10_alpha=1.0\' \
#                      dataset.vae.training.alpha=1.0

# CIFAR-10 (Form-1)
python main/train_nvp.py +dataset=cifar10/train \
                     dataset.ddpm.data.root=\'/data/jupyter/yeon/DiffuseVAE/main/datasets/\' \
                     dataset.ddpm.data.name='cifar10' \
                     dataset.ddpm.data.norm=True \
                     dataset.ddpm.data.hflip=True \
                     dataset.ddpm.model.dim=128 \
                     dataset.ddpm.model.dropout=0.3 \
                     dataset.ddpm.model.attn_resolutions=\'16,\' \
                     dataset.ddpm.model.n_residual=2 \
                     dataset.ddpm.model.dim_mults=\'1,2,2,2\' \
                     dataset.ddpm.model.n_heads=8 \
                     dataset.ddpm.training.type='form1' \
                     dataset.ddpm.training.cfd_rate=0.0 \
                     dataset.ddpm.training.epochs=500 \
                     dataset.ddpm.training.z_cond=False \
                     dataset.ddpm.training.batch_size=32 \
                     dataset.ddpm.training.vae_chkpt_path=\'/data/jupyter/yeon/DiffuseVAE/vae_cifar10_alpha=1.0/checkpoints/vae-cifar10_alpha=1.0-epoch=1499-train_loss=0.0000.ckpt\' \
                     dataset.ddpm.training.device=\'gpu:0\' \
                     dataset.ddpm.training.results_dir=\'/data/jupyter/yeon/DiffuseVAE/vae_nvp_ddpm/\' \
                     dataset.ddpm.training.workers=4 \
                     dataset.ddpm.training.chkpt_prefix=\'cifar10_rework_form1_28thJuly_sota_nheads=8_dropout=0.3\' \
                     dataset.ddpm.training.vae_weight=1.0 \
                     dataset.ddpm.training.ddpm_weight=1.0 \
                     dataset.ddpm.training.anneal_start=0 \
                     dataset.ddpm.training.anneal_end=100 \
                     dataset.ddpm.training.flow_base_dim=64 \
                     dataset.ddpm.training.flow_res_blocks=8 \
                     dataset.ddpm.training.flow_bottleneck=True \
                     dataset.ddpm.training.flow_skip=True \
                     dataset.ddpm.training.flow_weight_norm=True \
                     dataset.ddpm.training.flow_coupling_bn=True \
                     dataset.ddpm.training.flow_affine=True