import torch
import torch.nn.functional as F
from torch import Tensor
import pytorch_lightning as pl
import math
import numpy as np

class CannyFilter3D(torch.nn.Module):
    def __init__(self, low_threshold=0.01, high_threshold=0.05, kernel_size=1, sigma=1.5):
        super(CannyFilter3D, self).__init__()
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.gaussian_kernel = self.create_gaussian_kernel(kernel_size, sigma).to('cpu')
        self.sobel_x = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=torch.float32).to('cpu')
        self.sobel_y = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], dtype=torch.float32).to('cpu')

    def forward(self, x):
        edges = []
        # Apply edge detection slice-by-slice
        for i in range(x.shape[0]):
            slice_i = x[i].unsqueeze(0).unsqueeze(0)
            edge_slice = self.detect_edges(slice_i)
            edges.append(edge_slice.squeeze().unsqueeze(0))

        edges_3D = torch.stack(edges, dim=0).squeeze(1)

        if edges_3D.shape[1] != x.shape[1]:
            diff = edges_3D.shape[1] - x.shape[1]
            left_trim = diff // 2
            right_trim = diff - left_trim
            edges_3D = edges_3D[:, left_trim:-right_trim, :]

        if edges_3D.shape[2] != x.shape[2]:
            diff = edges_3D.shape[2] - x.shape[2]
            left_trim = diff // 2
            right_trim = diff - left_trim
            edges_3D = edges_3D[:, :, left_trim:-right_trim]

        return edges_3D

    def detect_edges(self, x):
        x = F.conv2d(x, self.gaussian_kernel, padding=1)
        grad_x = F.conv2d(x, self.sobel_x, padding=1)
        grad_y = F.conv2d(x, self.sobel_y, padding=1)
        magnitude = torch.sqrt(grad_x**2 + grad_y**2)
        angle = torch.atan2(grad_y, grad_x)

        # Non-maximum suppression
        magnitude = self.non_max_suppression(magnitude, angle)
        strong, weak = self.double_threshold(magnitude)
        edges = self.hysteresis(strong, weak)

        return edges

    def gaussian_kernel2D(self, kernel_size, sigma):
        kernel = torch.zeros((kernel_size, kernel_size), dtype=torch.float32)
        k = kernel_size // 2

        for i in range(kernel_size):
            for j in range(kernel_size):
                val = -((i - k) ** 2 + (j - k) ** 2) / (2 * sigma ** 2)
                kernel[i, j] = torch.exp(torch.tensor(val, dtype=torch.float32))

        kernel_sum = torch.sum(kernel)
        if kernel_sum != 0:
            kernel = kernel / kernel_sum

        return kernel

    def create_gaussian_kernel(self, kernel_size, sigma):
        kernel = self.gaussian_kernel2D(kernel_size, sigma).unsqueeze(0).unsqueeze(0)
        return kernel

    def non_max_suppression(self, mag, angle):
        # Quantize the angles
        quantized_angle = (torch.round(angle / (np.pi / 4)) * 45) % 180
        output = torch.zeros_like(mag)

        # Define masks for different edge directions
        horizontal_mask = ((quantized_angle == 0) | (quantized_angle == 180))
        vertical_mask = (quantized_angle == 90)
        diag_45_mask = (quantized_angle == 45)
        diag_135_mask = (quantized_angle == 135)

        # Compare with neighbors in the specified direction
        # Horizontal edge
        output[horizontal_mask] = torch.where(
            (mag[horizontal_mask] >= torch.roll(mag, shifts=1, dims=3)[horizontal_mask]) &
            (mag[horizontal_mask] >= torch.roll(mag, shifts=-1, dims=3)[horizontal_mask]),
            mag[horizontal_mask],
            torch.tensor([0.0], device=mag.device)
        )

        # Vertical edge
        output[vertical_mask] = torch.where(
            (mag[vertical_mask] >= torch.roll(mag, shifts=1, dims=2)[vertical_mask]) &
            (mag[vertical_mask] >= torch.roll(mag, shifts=-1, dims=2)[vertical_mask]),
            mag[vertical_mask],
            torch.tensor([0.0], device=mag.device)
        )

        # 45-degree diagonal edge
        output[diag_45_mask] = torch.where(
            (mag[diag_45_mask] >= torch.roll(mag, shifts=(1, 1), dims=(2, 3))[diag_45_mask]) &
            (mag[diag_45_mask] >= torch.roll(mag, shifts=(-1, -1), dims=(2, 3))[diag_45_mask]),
            mag[diag_45_mask],
            torch.tensor([0.0], device=mag.device)
        )

        # 135-degree diagonal edge
        output[diag_135_mask] = torch.where(
            (mag[diag_135_mask] >= torch.roll(mag, shifts=(1, -1), dims=(2, 3))[diag_135_mask]) &
            (mag[diag_135_mask] >= torch.roll(mag, shifts=(-1, 1), dims=(2, 3))[diag_135_mask]),
            mag[diag_135_mask],
            torch.tensor([0.0], device=mag.device)
        )

        return output

    def double_threshold(self, mag):
        high = torch.max(mag) * self.high_threshold
        low = high * self.low_threshold

        strong = torch.where(mag >= high, torch.tensor([1.0], device=mag.device), torch.tensor([0.0], device=mag.device))
        weak = torch.where((mag <= high) & (mag >= low), torch.tensor([0.5], device=mag.device), torch.tensor([0.0], device=mag.device))

        return strong, weak

    def hysteresis(self, strong, weak):
        edges = F.max_pool2d(strong, kernel_size=3, stride=1, padding=1)
        edges = torch.where((edges == 1) & (weak == 0.5), torch.tensor([1.0], device=strong.device), strong)
        return edges
