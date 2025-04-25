# 🧠 3D Vision Transformer for MR Electrical Properties Tomography

This project provides a 3D vision transformer-based neural network originally designed to reconstruct the electrical properties of tissues and materials from magnetic resonance (MR) measurements.

👉 [Jump to Graphical Abstract](#graphical-abstract)

---

## 🛠️ Installation

First, download the code and create a conda environment. The code runs with the following dependencies:

```bash
conda create -n DLEPT python=3.8  
conda activate DLEPT  
conda install pytorch==1.12.1 torchaudio==0.12.1 torchmetrics==0.10.0 torchvision==0.13.1 pytorch-lightning==1.7.7  
```

The code should also be compatible with later versions of these packages.

---

## 📂 Dataset and Pretrained Weights

The dataset used for training and evaluation in this project is **fully simulated** and publicly available on **Zenodo**:

- **Download the dataset**: [Download from Zenodo](https://zenodo.org/records/15258256)  
- **Dataset details**: [Check the MRM Paper](https://onlinelibrary.wiley.com/doi/full/10.1002/mrm.30338)

Each `.h5` file in the `train`, `val`, and `test` folders contains the following 3D tensors:
- `er`: Relative Permittivity Distribution  
- `se`: Electric Conductivity Distribution  
- `mag_b1p`: Magnitude of the Transmit Field  
- `tpa_b1p`: Transceive Phase  

Each `.h5` file in the `canny_train`, `canny_val`, and `canny_test` folders contains:
- `edges`: 3D Binary Edge Mask generated from Canny edge detection on the conductivity maps

The dataset is splitted to 
- 8065 (train) 3D volumes
- 1463 (val) 3D volumes
- 632 (test) 3D volumes 

**Pretrained model weights** are available here: [Download from Zenodo](https://zenodo.org/records/15258256)

---

## ⚠️ Disclaimer

This dataset is **entirely simulated** and intended for research purposes only. While every effort has been made to ensure its quality, it may contain artifacts or bugs. The authors assume no responsibility for any errors or issues resulting from its use. Users are encouraged to validate results independently.

---

## 🚀 Usage

To train, fine-tune, or test the neural network, run the script `runner_DLEPT.py`.

```bash
python runner_DLEPT.py
```

The input parameters for `runner_DLEPT.py` are hard-coded in the script. You can customize them by editing lines 14–48 to suit your needs.

---

## 📄 Information on Each File

- `runner_DLEPT.py`: Controls network operations (training, fine-tuning, testing, etc.).
- `models_module.py`: Implements a PyTorch Lightning module to build the model specified in `runner_DLEPT.py`. It includes functionality for generating predictions and saving results.
- `models.py`: Contains all the neural network models (UNet, TransUNet, FiLM generator, etc.).
- `losses.py`: Defines various loss functions used during training.
- `edge_detector.py`: A custom Canny filter implementation for generating 3D edge masks.
- `store_canny.py`: Calls `canny_class.py` to generate edge masks for a specified directory.
- `dataset_handler.py`: Creates training, validation, and test datasets from two `.h5` files. It expects the following 3D tensors from the first file: `mag_b1p`, `tpa_b1p`, `er`, `se`, and `edges` from the second. All tensors must have the same dimensions and be real.

---

## 🗂️ Root Directory Structure

Your root directory should be organized as follows:

```
Root_Dir
├── data
│   ├── train
│   ├── val
│   ├── test
│   ├── test_OOD
│   ├── train_fine_tune
│   ├── val_fine_tune
│   ├── test_fine_tune
│   └── test_fine_tune_OOD
└── logs
```

---

## 🧩 Modular Code

The repo includes modular code for various architectures, such as 3D UNet, 3D TransUNet, FiLM layers, etc. You can use them in your own project.

---

## 📚 Reference

If you use this work, please cite it as follows:

```bibtex
@article{giannakopoulos2025mr,
  title={MR electrical properties mapping using vision transformers and canny edge detectors},
  author={Giannakopoulos, Ilias I and Carluccio, Giuseppe and Keerthivasan, Mahesh B and Koerzdoerfer, Gregor and Lakshmanan, Karthik and De Moura, Hector L and Cruz Serrall{\'e}s, Jos{\'e} E and Lattanzi, Riccardo},
  journal={Magnetic Resonance in Medicine},
  volume={93},
  number={3},
  pages={1117--1131},
  year={2025},
  publisher={Wiley Online Library}
}
```

---

## 📝 License

This project is licensed under the [MIT License](https://choosealicense.com/licenses/mit/) – see the [LICENSE](LICENSE) file for details.

---

<h2 id="graphical-abstract">🖼️ Graphical Abstract</h2>

![Graphical Representation of the Training and Testing](https://github.com/GiannakopoulosIlias/vision-transformer-network-for-mr-electrical-properties-tomography/blob/main/figures/graphical_abstract.png)
