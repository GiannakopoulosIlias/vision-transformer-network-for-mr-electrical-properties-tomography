import torch
from models_module import NetworkModule
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
from pathlib import Path
import os
import math
import glob

# General Inputs
root_dir        = 'your/dir'
mode            = 'train'          # Switch to: train, test, test_OOD, fine_tune_train, fine_tune_test, fine_tune_test_OOD
                                   # The "fine_tune_train" option fine-tunes a model that was originally trained using "train", 
                                   # When fine-tuning an additional cascade is included in the original architecture.
                                   # The OOD options are used to test the network for a different test dataset, outside-of-distribution of your standard test dataset. 
# Architecture Inputs
architecture    = 'TransUNet'  # UNet, TransUNet
normalization   = 'FiLM'       # IN, FiLM
cascades        = 3            # Any number you want

# Unet Inputs
in_chans        = 3           # Three inputs for the B1+, Transceive Phase, and Edge mask.
out_chans       = 2           # Two outputs for relative permittivity and conductivity.
chans           = 64          # Channels of the UNets
num_pool_layers = 2           # Number of pooling layers per UNet
drop_prob       = 0.1         # Dropout probability

# Vision transformer Inputs, Control the size of the transformer and should be adapted depending on the size of your network and the size of your data.
num_heads       = 16 
num_layers      = 6
num_patches     = 1089

# Training Inputs
epochs             = 100   # Number of Training epochs
epochs_fine_tune   = 100   # Number of epochs for fine-tuning
div_factor         = 100   # Controls the learning rate, according to the scheduler
final_div_factor   = 1000  # Controls the learning rate, according to the scheduler
lr                 = 0.01  # Initial Learning Rate
lr_fine_tune       = 0.003 # Initial Learning Rate during fine-tuning
weight_decay       = 0     # Weight decay
gradient_clip_val  = 1.0   # Gradient Clip

# Normalization of the outputs. These are used so the network is trained with values between 0 and 1 and can be adjusted if needed. 
# Remember to multiply the outputs with these values to get the true electrical properties
norm_er = 135 # For relative permittivity
norm_se = 2.8 # For conductivity

# If fine tuning
fine_tune_str  = ''
if mode == 'fine_tune_train' or mode == 'fine_tune_test' or mode == 'fine_tune_test_OOD':
    fine_tune_str = '_fine_tune'

# If testing OOD
OOD_str = ''
if mode == 'test_OOD' or mode == 'fine_tune_test_OOD':
    OOD_str = '_OOD'

# Progress bar
progress_bar = RichProgressBar(
    theme=RichProgressBarTheme(
        description="green_yellow",
        progress_bar="green1",
        progress_bar_finished="green1",
        progress_bar_pulse="#6206E0",
        batch_progress="green_yellow",
        time="grey82",
        processing_speed="grey82",
        metrics="grey82"
    )
)

# Paths
logging_dir = Path(root_dir + "/logs/" + architecture + '_' + normalization + '_' + str(cascades) + "/")
mREDm_dir                   = logging_dir / 'mREDm'
mREDm_fine_tune_dir         = logging_dir / 'mREDm_fine_tune'
checkpoint_dir              = logging_dir / 'checkpoints'
checkpoints_fine_tune_dir   = logging_dir / 'checkpoints_fine_tune'
if not os.path.exists(logging_dir):
    os.makedirs(logging_dir)
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
if not os.path.exists(checkpoints_fine_tune_dir):
    os.makedirs(checkpoints_fine_tune_dir)
if not os.path.exists(mREDm_dir):
    os.makedirs(mREDm_dir)
if not os.path.exists(mREDm_fine_tune_dir):
    os.makedirs(mREDm_fine_tune_dir)
tb_logger           = TensorBoardLogger(logging_dir, name='mREDm')
tb_fine_tune_logger = TensorBoardLogger(logging_dir, name='mREDm_fine_tune')
ckpt_list           = sorted(checkpoint_dir.glob("*.ckpt"), key=os.path.getmtime)
ckpt_fine_tune_list = sorted(checkpoints_fine_tune_dir.glob("*.ckpt"), key=os.path.getmtime)

# Model
if mode == 'train':
    checkpoint_callback = ModelCheckpoint(
        dirpath = root_dir+'/logs/'+architecture+'_'+normalization+'_'+str(cascades)+'/checkpoints',
        save_top_k=True, verbose=True,
        monitor="val/MSE_loss",
        mode="min"
    )
    model = NetworkModule(
        in_chans=in_chans,
        out_chans=out_chans,
        chans=chans,
        num_pool_layers=num_pool_layers,
        drop_prob=drop_prob,
        lr=lr,
        epochs = epochs,
        div_factor = div_factor,
        final_div_factor = final_div_factor,
        weight_decay=weight_decay,
        data_path = root_dir,
        OOD_str=OOD_str,
        fine_tune_str = fine_tune_str,
        num_heads=num_heads,
        num_layers=num_layers,
        num_patches=num_patches,
        norm_er = norm_er,
        norm_se = norm_se,
        architecture = architecture,
        normalization = normalization,
        cascades = cascades
    )
    if ckpt_list:
        trainer = pl.Trainer(
            accelerator = 'gpu',
            devices = 1,
            max_epochs=epochs,
            default_root_dir=root_dir,
            resume_from_checkpoint=str(ckpt_list[-1]),
            callbacks=[checkpoint_callback, progress_bar],
            logger=tb_logger,
            log_every_n_steps=10,
            gradient_clip_val=gradient_clip_val
        )
    else:
        trainer = pl.Trainer(
            accelerator = 'gpu',
            devices = 1,
            max_epochs=epochs,
            default_root_dir=root_dir,
            callbacks=[checkpoint_callback, progress_bar],
            logger=tb_logger,
            log_every_n_steps=10,
            gradient_clip_val=gradient_clip_val
        )
    trainer.fit(model)

elif mode == 'fine_tune_train':
    checkpoint_callback = ModelCheckpoint(
        dirpath = root_dir+'/logs/'+architecture+'_'+normalization+'_'+str(cascades)+'/checkpoints_fine_tune',
        save_top_k=True,
        verbose=True,
        monitor="val/MSE_loss",
        mode="min"
    )
    model = NetworkModule(
        in_chans=in_chans,
        out_chans=out_chans,
        chans=chans,
        num_pool_layers=num_pool_layers,
        drop_prob=drop_prob,
        lr=lr_fine_tune,
        epochs=epochs_fine_tune,
        div_factor=div_factor,
        final_div_factor=final_div_factor,
        weight_decay=weight_decay,
        data_path=root_dir,
        OOD_str=OOD_str,
        fine_tune_str=fine_tune_str,
        num_heads=num_heads,
        num_layers=num_layers,
        num_patches=num_patches,
        norm_er = norm_er,
        norm_se = norm_se,
        architecture = architecture,
        normalization = normalization,
        cascades = cascades
    )
    ckpt_path = str(ckpt_list[-1])
    checkpoint = torch.load(ckpt_path, map_location="cuda:0")
    state_dict = checkpoint['state_dict']
    model_dict = model.state_dict()
    checkpoint = {k: v for k, v in state_dict.items() if k in model_dict}
    model_dict.update(checkpoint)
    model.load_state_dict(model_dict)
    model.requires_grad_(True)
    fine_tune_checkpoints = sorted(glob.glob(root_dir + '/logs/'+architecture+'_'+normalization+'_'+str(cascades)+'/checkpoints_fine_tune/*.ckpt'))
    latest_fine_tune_checkpoint = fine_tune_checkpoints[-1] if fine_tune_checkpoints else None
    trainer = pl.Trainer(
        accelerator = 'gpu',
        devices = 1,
        max_epochs=epochs_fine_tune,
        default_root_dir=root_dir,
        callbacks=[checkpoint_callback, progress_bar],
        logger=tb_fine_tune_logger,
        log_every_n_steps=10,
        gradient_clip_val=gradient_clip_val,
        resume_from_checkpoint=latest_fine_tune_checkpoint
    )
    trainer.fit(model)
elif mode == 'test' or mode == 'test_OOD':
    ckpt_path = str(ckpt_list[-1])
    model = NetworkModule.load_from_checkpoint(
        ckpt_path,
        in_chans=in_chans,
        out_chans=out_chans,
        chans=chans,
        num_pool_layers=num_pool_layers,
        drop_prob=drop_prob,
        lr=lr,
        epochs = epochs,
        div_factor = div_factor,
        final_div_factor = final_div_factor,
        weight_decay=weight_decay,
        data_path = root_dir,
        OOD_str=OOD_str,
        fine_tune_str=fine_tune_str,
        num_heads=num_heads,
        num_layers=num_layers,
        num_patches=num_patches,
        norm_er = norm_er,
        norm_se = norm_se,
        architecture = architecture,
        normalization = normalization,
        cascades = cascades,
    )
    trainer = pl.Trainer(accelerator='gpu', devices=1, callbacks=progress_bar)
    trainer.test(model)
elif mode == 'fine_tune_test' or mode == 'fine_tune_test_OOD':
    ckpt_path = str(ckpt_fine_tune_list[-1])
    model = NetworkModule.load_from_checkpoint(
        ckpt_path,
        in_chans=in_chans,
        out_chans=out_chans,
        chans=chans,
        num_pool_layers=num_pool_layers,
        drop_prob=drop_prob,
        lr=lr_fine_tune,
        epochs = epochs_fine_tune,
        div_factor = div_factor,
        final_div_factor = final_div_factor,
        weight_decay=weight_decay,
        data_path = root_dir,
        OOD_str=OOD_str,
        fine_tune_str=fine_tune_str,
        num_heads=num_heads,
        num_layers=num_layers,
        num_patches=num_patches,
        norm_er = norm_er,
        norm_se = norm_se,
        architecture = architecture,
        normalization = normalization,
        cascades = cascades
    )
    trainer = pl.Trainer(accelerator='gpu', devices=1, callbacks=progress_bar)
    trainer.test(model)
