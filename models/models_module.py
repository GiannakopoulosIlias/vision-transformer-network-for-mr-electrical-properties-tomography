import os
import torch
from torch import nn
import numpy as np
from lion_pytorch import Lion
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch import Tensor
from argparse import ArgumentParser
import pytorch_lightning as pl
import h5py
import math
from torchvision.utils import make_grid
import torchvision.utils as vutils
from tqdm import tqdm as _tqdm
import matplotlib.pyplot as plt

from datasets.dataset_handler import CustomDataset, CustomDataset_Test
from .models import Unet3D, Unet3D_ViT, Unet3D_FiLM, Unet3D_ViT, Unet3D_ViT_FiLM, FiLMGeneratorModel
from utils.losses import MSE_loss, L1_loss, SSIM_loss

class NetworkModule(pl.LightningModule):
    def __init__(self, in_chans=2, out_chans=2, chans=32, num_pool_layers=4, drop_prob=0.2, lr=0.001, epochs = 100, div_factor = 100, final_div_factor = 1000, weight_decay=0.0, data_path='', OOD_str=None, fine_tune_str=None, num_heads=None, num_layers=None, num_patches=None, norm_er = 120, norm_se = 2.5, architecture = 'TransUNet', normalization = 'FiLM', cascades = 3):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob
        self.lr = lr
        self.epochs = epochs
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        self.weight_decay = weight_decay
        self.data_path = data_path
        self.OOD_str = OOD_str
        self.fine_tune_str = fine_tune_str
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_patches = num_patches
        self.norm_er = norm_er
        self.norm_se = norm_se
        self.architecture = architecture
        self.normalization = normalization
        self.cascades = cascades
        
        self.unets = nn.ModuleList()

        for i in range(self.cascades):
            in_chans = self.in_chans if i == 0 else self.chans
            out_chans = self.out_chans if i == self.cascades - 1 else self.chans
            if self.architecture == 'UNet':
                if self.normalization == 'FiLM':
                    unet = Unet3D_FiLM(in_chans=in_chans,out_chans=out_chans,chans=self.chans,num_pool_layers=self.num_pool_layers,drop_prob=self.drop_prob)
                elif self.normalization == 'IN':
                    unet = Unet3D(in_chans=in_chans,out_chans=out_chans,chans=self.chans,num_pool_layers=self.num_pool_layers,drop_prob=self.drop_prob)
            elif self.architecture == 'TransUNet':
                if self.normalization == 'FiLM':
                    unet = Unet3D_ViT_FiLM(in_chans=in_chans,out_chans=out_chans,chans=self.chans,num_pool_layers=self.num_pool_layers,num_heads=self.num_heads,num_layers=self.num_layers,num_patches=self.num_patches,drop_prob=self.drop_prob)
                elif self.normalization == 'IN':
                    unet = Unet3D_ViT(in_chans=in_chans,out_chans=out_chans,chans=self.chans,num_pool_layers=self.num_pool_layers,num_heads=self.num_heads,num_layers=self.num_layers,num_patches=self.num_patches,drop_prob=self.drop_prob)
            self.unets.append(unet)

        
        if self.fine_tune_str == '_fine_tune':
            if self.architecture == 'UNet':
                unet_fine_tune = Unet3D(in_chans=self.out_chans,out_chans=self.out_chans,chans=self.chans,num_pool_layers=self.num_pool_layers,drop_prob=self.drop_prob)
            elif self.architecture == 'TransUNet':
                unet_fine_tune = Unet3D_ViT(in_chans=self.out_chans,out_chans=self.out_chans,chans=self.chans,num_pool_layers=self.num_pool_layers,num_heads=self.num_heads,num_layers=self.num_layers,num_patches=self.num_patches,drop_prob=self.drop_prob)
            self.unets.append(unet_fine_tune)

        if self.normalization == 'FiLM':
            self.film_generator = FiLMGeneratorModel(out_chans=self.chans, cascades_num=self.cascades, drop_prob=self.drop_prob)

        data_train_path = self.data_path + 'data/train'+self.fine_tune_str+'/'
        data_val_path = self.data_path + 'data/val'+self.fine_tune_str+'/'
        data_test_path = self.data_path + 'data/test'+self.fine_tune_str+self.OOD_str+'/'
        canny_data_train_path = self.data_path + 'data/canny_train'+self.fine_tune_str+'/'
        canny_data_val_path = self.data_path + 'data/canny_val'+self.fine_tune_str+'/'
        canny_data_test_path = self.data_path + 'data/canny_test'+self.fine_tune_str+self.OOD_str+'/'
        cache_train_path = self.data_path+'cache/'+'train'+self.fine_tune_str+'/dataset_cache.pkl'
        cache_val_path = self.data_path+'cache/'+'val'+self.fine_tune_str+'/dataset_cache.pkl'
        self.train_dataset = CustomDataset(data_path=data_train_path,canny_data_path=canny_data_train_path,cache_path=cache_train_path, norm_er=self.norm_er, norm_se=self.norm_se)
        self.val_dataset = CustomDataset(data_path=data_val_path,canny_data_path=canny_data_val_path,cache_path=cache_val_path, norm_er=self.norm_er, norm_se=self.norm_se)
        self.test_dataset = CustomDataset_Test(data_path=data_test_path,canny_data_path=canny_data_test_path, norm_er=self.norm_er, norm_se=self.norm_se)
        self.loss_mse = MSE_loss()
        self.loss_l1 = L1_loss()
        self.loss_ssim = SSIM_loss()

    def forward(self, x):
        
        u = x.to(self.device)
        if self.normalization == 'FiLM':
            [beta, gamma] = self.film_generator(x[:, 2, :, :, :].unsqueeze(1))
            for i, unet in enumerate(self.unets):
                if i < len(self.unets) - 1 or self.fine_tune_str != '_fine_tune':
                    b = torch.squeeze(beta[i,:])
                    g = torch.squeeze(gamma[i,:])
                    u = unet(u, b, g)
                else:
                    u = unet(u)
        elif self.normalization == 'IN':
            for i, unet in enumerate(self.unets):
                u = unet(u)
        return u
        
    def training_step(self, batch, batch_idx):
        x, y, filenames = batch
        y_hat = self(x)
        mse_loss_er  = self.loss_mse(y_hat[:,0,:,:,:], y[:,0,:,:,:])
        mse_loss_se  = self.loss_mse(y_hat[:,1,:,:,:], y[:,1,:,:,:])
        mse_loss     = (mse_loss_er + mse_loss_se)/2
        self.log('train/MSE_loss', mse_loss, batch_size=4)
        self.log('train/MSE_loss_er', mse_loss_er, batch_size=4)
        self.log('train/MSE_loss_se', mse_loss_se, batch_size=4)
        train_loss = mse_loss
        return {'loss': train_loss}

    def validation_step(self, batch, batch_idx):
        x, y, filenames = batch
        y_hat = self(x)
        mse_loss_er  = self.loss_mse(y_hat[:,0,:,:,:], y[:,0,:,:,:])
        mse_loss_se  = self.loss_mse(y_hat[:,1,:,:,:], y[:,1,:,:,:])
        mse_loss     = (mse_loss_er + mse_loss_se)/2
        ssim_loss_er = self.loss_ssim(y_hat[:,0,:,:,:], y[:,0,:,:,:])
        ssim_loss_se = self.loss_ssim(y_hat[:,1,:,:,:], y[:,1,:,:,:])
        ssim_loss    = (ssim_loss_er + ssim_loss_se)/2
        l1_loss_er   = self.loss_l1(y_hat[:,0,:,:,:], y[:,0,:,:,:])
        l1_loss_se   = self.loss_l1(y_hat[:,1,:,:,:], y[:,1,:,:,:])
        l1_loss      = (l1_loss_er + l1_loss_se)/2
        self.log('val/MSE_loss', mse_loss, batch_size=1)
        self.log('val/MSE_loss_er', mse_loss_er, batch_size=1)
        self.log('val/MSE_loss_se', mse_loss_se, batch_size=1)
        self.log('val/L1_loss', l1_loss, batch_size=1)
        self.log('val/L1_loss_er', l1_loss_er, batch_size=1)
        self.log('val/L1_loss_se', l1_loss_se, batch_size=1)
        self.log('val/SSIM_loss', ssim_loss, batch_size=1)
        self.log('val/SSIM_loss_er', ssim_loss_er, batch_size=1)
        self.log('val/SSIM_loss_se', ssim_loss_se, batch_size=1)
        val_loss = mse_loss

        if batch_idx < 8:
            middle_slice_idx = y.shape[2] // 2
            y_middle_slice = y[0, :, middle_slice_idx, :, :]
            y_hat_middle_slice = y_hat[0, :, middle_slice_idx, :, :]
            y_middle_slice = vutils.make_grid(y_middle_slice.to('cpu'), normalize=False, scale_each=False, nrow=1, padding=8, pad_value=1)
            y_hat_middle_slice = vutils.make_grid(y_hat_middle_slice.to('cpu'), normalize=False, scale_each=False, nrow=1, padding=8, pad_value=1)
            abs_diff = vutils.make_grid(torch.abs(y_hat_middle_slice - y_middle_slice), normalize=False, scale_each=False, nrow=1, padding=8, pad_value=1)
            image_row = torch.cat((y_middle_slice.squeeze(), y_hat_middle_slice.squeeze(), abs_diff.squeeze()), dim=1)
            self.logger.experiment.add_image('val/pred_vs_gt_example_' + str(batch_idx), image_row, self.current_epoch)

        return {'loss': val_loss}

    def test_step(self, batch, batch_idx):
        x, y, filenames = batch
        y_hat = self(x)
        mse_loss_er  = self.loss_mse(y_hat[:,0,:,:,:], y[:,0,:,:,:])
        mse_loss_se  = self.loss_mse(y_hat[:,1,:,:,:], y[:,1,:,:,:])
        mse_loss     = (mse_loss_er + mse_loss_se)/2
        l1_loss_er   = self.loss_l1(y_hat[:,0,:,:,:], y[:,0,:,:,:])
        l1_loss_se   = self.loss_l1(y_hat[:,1,:,:,:], y[:,1,:,:,:])
        l1_loss      = (l1_loss_er + l1_loss_se)/2
        ssim_loss_er = self.loss_ssim(y_hat[:,0,:,:,:], y[:,0,:,:,:])
        ssim_loss_se = self.loss_ssim(y_hat[:,1,:,:,:], y[:,1,:,:,:])
        ssim_loss    = (ssim_loss_er + ssim_loss_se)/2
        output_directory = self.data_path + "data/predictions" + self.fine_tune_str + self.OOD_str + "_" + self.architecture + '_' + self.normalization + '_' + str(self.cascades) + "/"
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        for i, filename in enumerate(filenames):
            output_filename = output_directory + "prediction_" + filename
            with h5py.File(output_filename, "w") as f:
                f.create_dataset("er", data=np.squeeze(y_hat[i,0,:,:,:].detach().cpu().numpy()))
                f.create_dataset("se", data=np.squeeze(y_hat[i,1,:,:,:].detach().cpu().numpy()))
        self.log('test/MSE_loss', mse_loss, prog_bar=True, batch_size=1)
        self.log('test/MSE_loss_er', mse_loss_er, batch_size=1)
        self.log('test/MSE_loss_se', mse_loss_se, batch_size=1)
        self.log('test/L1_loss', l1_loss, batch_size=1)
        self.log('test/L1_loss_er', l1_loss_er, batch_size=1)
        self.log('test/L1_loss_se', l1_loss_se, batch_size=1)
        self.log('test/SSIM_loss', ssim_loss, batch_size=1)
        self.log('test/SSIM_loss_er', ssim_loss_er, batch_size=1)
        self.log('test/SSIM_loss_se', ssim_loss_se, batch_size=1)
        test_loss = mse_loss
        return {'loss': test_loss}

    def train_dataloader(self):
        train_loader = DataLoader(self.train_dataset, batch_size=4, shuffle=True, num_workers=10)
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(self.val_dataset, batch_size=1, shuffle=False, num_workers=10)
        return val_loader

    def test_dataloader(self):
        test_loader = DataLoader(self.test_dataset, batch_size=1, shuffle=False, num_workers=10)
        return test_loader

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=self.lr,epochs=self.epochs,steps_per_epoch=len(self.train_dataset),div_factor=self.div_factor,final_div_factor=self.final_div_factor)
        return [optimizer], [scheduler]
