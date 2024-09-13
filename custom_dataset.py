import os
import torch
from torch import nn
import numpy as np
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch import Tensor
from argparse import ArgumentParser
import pytorch_lightning as pl
import h5py
import math
import pickle
import random
from pathlib import Path

class CustomDataset(Dataset):
    def __init__(self, data_path, canny_data_path, cache_path, norm_er, norm_se):
        self.data_path = data_path
        self.canny_data_path = canny_data_path
        self.cache_path = Path(cache_path)
        self.file_list = os.listdir(self.data_path)
        self.canny_file_list = os.listdir(self.canny_data_path)

        if os.path.exists(self.cache_path):
            with open(self.cache_path, 'rb') as fc:
                cached_data = pickle.load(fc)
                self.inputs, self.outputs, self.filenames = cached_data
        else:
            self.inputs = []
            self.outputs = []
            self.filenames = []
            for [data_file, canny_data_file] in zip(self.file_list,self.canny_file_list):
                with h5py.File(os.path.join(self.data_path, data_file), 'r') as f:
                    mag_b1p       = f['mag_b1p'][:]
                    tpa_b1p       = f['tpa_b1p'][:]
                    er            = f['er'][:]
                    se            = f['se'][:]

                    with h5py.File(os.path.join(self.canny_data_path, canny_data_file), 'r') as cf:
                        edges = cf['edges'][:]

                    conc_inp      = torch.stack([torch.tensor(mag_b1p)/mag_b1p.max(), torch.tensor(tpa_b1p)/tpa_b1p.max(), torch.tensor(edges)], dim=0)
                    conc_out      = torch.stack([torch.tensor(er)/norm_er, torch.tensor(se)/norm_se], dim=0)
                    self.inputs.append(conc_inp)
                    self.outputs.append(conc_out)
                    self.filenames.append(data_file)

            self.inputs = torch.stack(self.inputs, dim=-1)
            self.outputs = torch.stack(self.outputs, dim=-1)

            cached_data = (self.inputs, self.outputs, self.filenames)
            with open(self.cache_path, 'wb') as f:
                pickle.dump(cached_data, f)

    def __len__(self):
        return self.inputs.shape[-1]

    def __getitem__(self, idx):
        x = self.inputs[..., idx]
        y = self.outputs[..., idx]
        filename = self.filenames[idx]
        return x, y, filename


class CustomDataset_Test(Dataset):
    def __init__(self, data_path, canny_data_path, norm_er, norm_se):
        self.data_path = data_path
        self.canny_data_path = canny_data_path
        self.file_list = os.listdir(self.data_path)
        self.canny_file_list = os.listdir(self.canny_data_path)

        self.inputs = []
        self.outputs = []
        self.filenames = []
        for [data_file, canny_data_file] in zip(self.file_list,self.canny_file_list):
            with h5py.File(os.path.join(self.data_path, data_file), 'r') as f:
                mag_b1p       = f['mag_b1p'][:]
                tpa_b1p       = f['tpa_b1p'][:]
                er            = f['er'][:]
                se            = f['se'][:]

                with h5py.File(os.path.join(self.canny_data_path, canny_data_file), 'r') as cf:
                    edges = cf['edges'][:]

                conc_inp      = torch.stack([torch.tensor(mag_b1p)/mag_b1p.max(), torch.tensor(tpa_b1p)/tpa_b1p.max(), torch.tensor(edges)], dim=0)
                conc_out      = torch.stack([torch.tensor(er)/norm_er, torch.tensor(se)/norm_se], dim=0)
                self.inputs.append(conc_inp)
                self.outputs.append(conc_out)
                self.filenames.append(data_file)

        self.inputs = torch.stack(self.inputs, dim=-1)
        self.outputs = torch.stack(self.outputs, dim=-1)

    def __len__(self):
        return self.inputs.shape[-1]

    def __getitem__(self, idx):
        x = self.inputs[..., idx]
        y = self.outputs[..., idx]
        filename = self.filenames[idx]
        return x, y, filename
