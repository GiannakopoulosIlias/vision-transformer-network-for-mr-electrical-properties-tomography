import os
import torch
import numpy as np
import h5py
from canny_class import CannyFilter3D

# Define the base paths and folder names
base_data_path = '../dev/mylabspace/Ilias/GMT_data/data/Fast_EPR_Data/Simulated_Dataset/Birdcage_Coil_3T_3D/data/'
base_canny_path = '../dev/mylabspace/Ilias/GMT_data/data/Fast_EPR_Data/Simulated_Dataset/Birdcage_Coil_3T_3D/data/'
#folders = ['train', 'val', 'test', 'test_OOD', 'train_fine_tune', 'val_fine_tune', 'test_fine_tune', 'test_fine_tune_OOD']
folders = ['test_OOD']

# Canny filter initialization
canny = CannyFilter3D()

# Iterate over each folder
for folder in folders:
    data_path = os.path.join(base_data_path, folder)
    canny_path = os.path.join(base_canny_path, 'canny_' + folder)
    file_list = os.listdir(data_path)
    h5_files = [file for file in file_list if file.endswith('.h5')]

    c = 0
    for data_file in h5_files:
        print(c)
        c += 1
        with h5py.File(os.path.join(data_path, data_file), 'r') as f:
            se = f['se'][:]
            edges = canny(torch.tensor(se)/2.5)
        storage_path = os.path.join(canny_path, data_file)
        with h5py.File(storage_path, 'w') as hf:
            hf.create_dataset("edges", data=np.array(edges))
