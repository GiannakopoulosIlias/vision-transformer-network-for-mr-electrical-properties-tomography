# 3D Vision Transformer for MR Electrical Properties Tomography
This project provides a 3D vision transformer-based neural network originally designed to reconstruct the electrical properties of tissues and materials from magnetic resonance (MR) measurements.

## Installation
First, download the code and create a `conda` environment. The code runs with the following dependencies:

conda create -n DLEPT python=3.8  
conda activate DLEPT  
conda install pytorch==1.12.1 torchaudio==0.12.1 torchmetrics==0.10.0 torchvision==0.13.1 pytorch-lightning==1.7.7  

The code should also be compatible with later versions of these packages.

## Usage
- To train, fine-tune, or test the neural network, run the script `runner_DLEPT.py`.  
  python runner_DLEPT.py  

- The input parameters for `runner_DLEPT.py` are hard-coded in the script. You can customize them by editing lines 14-48 to suit your needs.

## Information on Each File
- `runner_DLEPT.py`: Controls network operations (training, fine-tuning, testing, etc.).
- `models_module.py`: Implements a PyTorch Lightning module to build the model specified in `runner_DLEPT.py`. It includes functionality for generating predictions and saving results.
- `models.py`: Contains all the neural network models (UNet, TransUNet, FiLM generator, etc.).
- `losses.py`: Defines various loss functions used during training.
- `edge_detector.py`: A custom Canny filter implementation for generating 3D edge masks.
- `store_canny.py`: Calls `canny_class.py` to generate edge masks for a specified directory.
- `dataset_handler.py`: Creates training, validation, and test datasets from two `.h5` files. It expects the following 3D tensors from the first file: `'mag_b1p'`, `'tpa_b1p'`, `'er'`, `'se'`, and `'edges'` from the second. All tensors must have the same dimensions and be real.
  - `'mag_b1p'`: B1+ magnitude
  - `'tpa_b1p'`: Transceive phase
  - `'er'`: Relative permittivity
  - `'se'`: Conductivity
  - `'edges'`: Canny edge masks

## Root Directory
Your root directory should be organized as follows:
- Root_Dir
  - data
    - train
      - afile_1.h5
      - ...
    - val
      - bfile_1.h5
      - ...
    - test
      - cfile_1.h5
      - ...
    - test_OOD
      - dfile_1.h5
      - ...
    - train_fine_tune
      - efile_1.h5
      - ...
    - val_fine_tune
      - ffile_1.h5
      - ...
    - test_fine_tune
      - gfile_1.h5
      - ...
    - test_fine_tune_OOD
      - hfile_1.h5
      - ...
  - logs   

## Modular Code
The repo includes modular code for various architectures, such as 3D UNet, 3D TransUNet, FiLM layers etc. You can use them in your project.

## Reference
If you use this work, please cite it as follows:

```bibtex
@article{giannakopoulos2024mr,
  title       = {MR electrical properties mapping using vision transformers and canny edge detectors},
  author      = {Giannakopoulos, Ilias I and Carluccio, Giuseppe and Keerthivasan, Mahesh B and 
                 Koerzdoerfer, Gregor and Lakshmanan, Karthik and De Moura, Hector L and 
                 Cruz Serrall{\'e}s, Jos{\'e} E and Lattanzi, Riccardo},
  journal     = {Magnetic Resonance in Medicine},
  year        = {2024},
  publisher   = {Wiley Online Library}
}
```

## License
This project is licensed under the [MIT License](https://choosealicense.com/licenses/mit/) - see the [LICENSE](LICENSE) file for details.

## Graphical Abstract

![Graphical Representation of the Training and Testing](https://github.com/GiannakopoulosIlias/vision-transformer-network-for-mr-electrical-properties-tomography/blob/main/figures/graphical_abstract.png)
