import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class MSE_loss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_hat, y):
        return F.mse_loss(y_hat, y)

class L1_loss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_hat, y):
        return F.l1_loss(y_hat, y)

class CANNY_loss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y, edge_mask, canny):
        edge_mask_y = torch.unsqueeze(canny(torch.squeeze(y)),0)
        edge_loss = F.mse_loss(edge_mask_y, edge_mask)
        return edge_loss

class EDGE_loss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_hat, edge_mask):
        edge_diff = (y_hat - y) * edge_mask
        edge_loss = edge_diff.abs().sum() / edge_mask.sum()
        return edge_loss

class Gradient_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_hat, y):
        dy_hat_dx = y_hat[:, 1:, :, :] - y_hat[:, :-1, :, :]
        dy_hat_dy = y_hat[:, :, 1:, :] - y_hat[:, :, :-1, :]
        dy_hat_dz = y_hat[:, :, :, 1:] - y_hat[:, :, :, :-1]
        dy_dx = y[:, 1:, :, :] - y[:, :-1, :, :]
        dy_dy = y[:, :, 1:, :] - y[:, :, :-1, :]
        dy_dz = y[:, :, :, 1:] - y[:, :, :, :-1]
        loss_dx = F.mse_loss(dy_hat_dx, dy_dx, reduction='mean')
        loss_dy = F.mse_loss(dy_hat_dy, dy_dy, reduction='mean')
        loss_dz = F.mse_loss(dy_hat_dz, dy_dz, reduction='mean')
        return loss_dx + loss_dy + loss_dz

class SSIM_loss(nn.Module):

    def __init__(self, win_size: int = 7, k1: float = 0.01, k2: float = 0.03):
        super().__init__()
        self.win_size = win_size
        self.k1, self.k2 = k1, k2
        self.register_buffer("w", torch.ones(1, 1, win_size, win_size) / win_size ** 2)
        NP = win_size ** 2
        self.cov_norm = NP / (NP - 1)

    def forward(self, X: torch.Tensor, Y: torch.Tensor):
        assert isinstance(self.w, torch.Tensor)

        batch_size, depth, _, _ = X.shape
        ssim_per_slice = torch.zeros((batch_size, depth)).to(X.device)
        data_range = 1

        for i in range(depth):
            slice_X = X[:, i, :, :]
            slice_Y = Y[:, i, :, :]
            C1 = (self.k1 * data_range) ** 2
            C2 = (self.k2 * data_range) ** 2
            ux = F.conv2d(slice_X.unsqueeze(1), self.w)  # typing: ignore
            uy = F.conv2d(slice_Y.unsqueeze(1), self.w)  #
            uxx = F.conv2d(slice_X.unsqueeze(1) * slice_X.unsqueeze(1), self.w)
            uyy = F.conv2d(slice_Y.unsqueeze(1) * slice_Y.unsqueeze(1), self.w)
            uxy = F.conv2d(slice_X.unsqueeze(1) * slice_Y.unsqueeze(1), self.w)
            vx = self.cov_norm * (uxx - ux * ux)
            vy = self.cov_norm * (uyy - uy * uy)
            vxy = self.cov_norm * (uxy - ux * uy)
            A1, A2, B1, B2 = (
                2 * ux * uy + C1,
                2 * vxy + C2,
                ux ** 2 + uy ** 2 + C1,
                vx + vy + C2,
            )
            D = B1 * B2
            S = (A1 * A2) / D
            ssim_per_slice[:, i] = S.squeeze().mean()

        return 1 - ssim_per_slice.mean(dim=1)

class DICE_loss(nn.Module):
    def __init__(self, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon
    def forward(self, y_hat, y):
        y = torch.sigmoid(y)
        y = y.view(-1)
        y_hat = y_hat.view(-1)
        intersection = (y * y_hat).sum()
        dice_score = (2. * intersection + self.epsilon) / (y.sum() + y_hat.sum() + self.epsilon)
        return 1. - dice_score

class Boundary_loss(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def mask_to_boundary(self, mask):
        kernel = torch.ones((1, 1, 3, 3, 3), device=mask.device, dtype=mask.dtype)
        dilation = F.conv3d(mask, kernel, padding=1) - mask
        boundary = torch.clamp(dilation, 0, 1)
        return boundary

    def forward(self, y_hat, y):
        y_boundary = self.mask_to_boundary(y)
        y_hat_boundary = self.mask_to_boundary(y_hat)
        intersection = torch.sum(y_boundary * y_hat_boundary)
        union = torch.sum(y_boundary) + torch.sum(y_hat_boundary) + self.eps
        loss = 1 - (2 * intersection / union)
        return loss

class Hybrid_loss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse_loss = nn.MSELoss()
        self.boundary_loss = DICE_loss()

    def forward(self, y_hat, y):
        mse = self.mse_loss(y_hat, y)
        dice = self.boundary_loss(y_hat, y)
        return self.alpha * mse + self.beta * dice
