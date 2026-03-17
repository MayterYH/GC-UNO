"""
This module defines the core neural network architectures and physics-informed 
loss calculation methods for the GC-UNO (Grid-Constrained U-shaped Neural Operator) model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.cuda.amp import autocast, GradScaler
from torch.utils.checkpoint import checkpoint

from src.utilities3 import LpLoss
from src.config import par

# Global loss function instance
myloss = LpLoss(size_average=False)

# ==============================================================================
# Spectral and Gradient Utilities
# ==============================================================================

class FFTGradient3d(nn.Module):
    """
    Computes 3D gradients using Fast Fourier Transform (FFT) with optional padding.
    """
    def __init__(self, pad_x=0, pad_y=0, pad_z=0, dx=1.0):
        super().__init__()
        self.pad_x = pad_x
        self.pad_y = pad_y
        self.pad_z = pad_z
        self.dx = dx

    def forward(self, field):
        # field shape: (nx, ny, nz)
        nx, ny, nz = field.shape
        
        # Symmetric padding: (z_pad, y_pad, x_pad)
        padded = F.pad(field,
                       (self.pad_z // 2, self.pad_z - self.pad_z // 2,
                        self.pad_y // 2, self.pad_y - self.pad_y // 2,
                        self.pad_x // 2, self.pad_x - self.pad_x // 2))
                        
        # Compute frequencies
        kx = torch.fft.fftfreq(padded.size(0), d=self.dx, device=field.device) * 2 * torch.pi
        ky = torch.fft.fftfreq(padded.size(1), d=self.dx, device=field.device) * 2 * torch.pi
        kz = torch.fft.fftfreq(padded.size(2), d=self.dx, device=field.device) * 2 * torch.pi
        
        # Forward FFTn
        fft_p = torch.fft.fftn(padded, dim=(0, 1, 2), norm='forward')
        
        # Gradients in frequency domain
        grad_x_p = torch.fft.ifftn(1j * kx.view(-1, 1, 1) * fft_p, dim=(0, 1, 2), norm='forward').real
        grad_y_p = torch.fft.ifftn(1j * ky.view(1, -1, 1) * fft_p, dim=(0, 1, 2), norm='forward').real
        grad_z_p = torch.fft.ifftn(1j * kz.view(1, 1, -1) * fft_p, dim=(0, 1, 2), norm='forward').real
        
        # Crop back to original size
        xs, ys, zs = self.pad_x // 2, self.pad_y // 2, self.pad_z // 2
        grad_x = grad_x_p[xs:xs + nx, ys:ys + ny, zs:zs + nz]
        grad_y = grad_y_p[xs:xs + nx, ys:ys + ny, zs:zs + nz]
        grad_z = grad_z_p[xs:xs + nx, ys:ys + ny, zs:zs + nz]
        
        return grad_x, grad_y, grad_z


def spectral_divergence(Y):
    """
    Compute the divergence of a 3D vector field Y in spectral space.
    Assumes periodic boundary conditions and unit grid spacing.
    
    Args:
        Y: Vector field tensor of shape (Nz, Nx, Ny, 3)
    Returns:
        div: Divergence tensor of shape (Nz, Nx, Ny)
    """
    Yc = Y.permute(3, 0, 1, 2)
    Y_hat = torch.fft.rfftn(Yc, dim=(1, 2, 3))  # (3, Nz, Nx, Ny//2+1)
    Nz, Nx, Ny = Y.size(0), Y.size(1), Y.size(2)
    
    # Wave numbers
    kz = torch.fft.fftfreq(Nz, d=1e-5) * 2 * math.pi
    kx = torch.fft.fftfreq(Nx, d=1e-5) * 2 * math.pi
    ky = torch.fft.rfftfreq(Ny, d=1e-5) * 2 * math.pi
    
    KZ = kz[None, :, None, None].to(Y.device)
    KX = kx[None, None, :, None].to(Y.device)
    KY = ky[None, None, None, :].to(Y.device)
    
    # Divergence in frequency domain: i*(kx*Bx + ky*By + kz*Bz)
    div_hat = (1j * KX * Y_hat[0] + 
               1j * KY * Y_hat[1] + 
               1j * KZ * Y_hat[2])
               
    div = torch.fft.irfftn(div_hat, s=(Nz, Nx, Ny), dim=(0, 1, 2))
    return div


def spectral_jacobian(Y, dx=1.0, dy=1.0, dz=1.0):
    """
    Compute the Jacobian matrix of a 3D magnetic field B using spectral methods.
    
    Args:
        Y: Magnetic field components (Nz, Nx, Ny, 3). Order: (B_z, B_x, B_y)
        dx, dy, dz: Physical grid spacing
    Returns:
        jac: Jacobian matrix of shape (Nz, Nx, Ny, 3, 3), where jac[..., i, j] = ∂B_i/∂x_j
    """
    Yc = Y.permute(3, 0, 1, 2)  
    Nz, Nx, Ny = Y.shape[0], Y.shape[1], Y.shape[2]
    
    Y_hat = torch.fft.rfftn(Yc, dim=(1, 2, 3)) 
    
    def get_wavenumbers(n, delta):
        """Returns physical wave numbers k = 2π * f / (n*delta)"""
        f = torch.fft.fftfreq(n, d=1.0).to(Y.device)  
        return f * (2 * math.pi / delta)
    
    kz = get_wavenumbers(Nz, dz).view(1, Nz, 1, 1)      
    kx = get_wavenumbers(Nx, dx).view(1, 1, Nx, 1)      
    ky = get_wavenumbers(Ny, dy)[:Ny//2+1].view(1, 1, 1, -1)  

    gradients = []
    for i in range(3): 
        # Differentiation in frequency domain
        grad_z_hat = 1j * kz * Y_hat[i]  # ∂B_i/∂z
        grad_x_hat = 1j * kx * Y_hat[i]  # ∂B_i/∂x
        grad_y_hat = 1j * ky * Y_hat[i]  # ∂B_i/∂y
        
        # Inverse transform back to spatial domain
        grad_z = torch.fft.irfftn(grad_z_hat, s=(Nz, Nx, Ny), dim=(1, 2, 3))
        grad_x = torch.fft.irfftn(grad_x_hat, s=(Nz, Nx, Ny), dim=(1, 2, 3))
        grad_y = torch.fft.irfftn(grad_y_hat, s=(Nz, Nx, Ny), dim=(1, 2, 3))
        
        gradients.append(torch.stack([grad_z, grad_x, grad_y], dim=-1))
    
    jac = torch.stack(gradients, dim=-2) 
    return jac


def jacobian(output, coords):
    """
    Compute Jacobian matrix using PyTorch autograd.
    """
    jac_matrix = [torch.autograd.grad(output[:, i], coords,
                                      grad_outputs=torch.ones_like(output[:, i]).to(output),
                                      retain_graph=True, create_graph=True, allow_unused=True)[0]
                  for i in range(output.shape[1])]
    jac_matrix = torch.stack(jac_matrix, dim=1)
    return jac_matrix


# ==============================================================================
# Model Architecture
# ==============================================================================

class SpectralConv2d_fast(nn.Module):
    """
    2D Fourier layer. Computes the 2D Fast Fourier Transform, filters 
    higher frequencies, applies a linear transform, and transforms back.
    """
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))

    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        B = x.shape[0]
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(B, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat, device=x.device)
        
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class FNO2d(nn.Module):
    """
    Grid-Constrained U-shaped Neural Operator (GC-UNO).
    A U-Net style architecture integrated with Fourier Neural Operator layers.
    """
    def __init__(self, modes1, modes2, width):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width

        # Input Projection
        self.fcin = nn.Linear(6, width) 

        # -------- Encoder --------
        self.conv_down1 = SpectralConv2d_fast(width, int(width * 1), modes1, modes2) 
        self.w_down1 = nn.Conv2d(width, int(width * 1), 1) 

        self.conv_down2 = SpectralConv2d_fast(int(width * 1), int(width * 1.5), int(modes1 // 1.5), int(modes2 // 1.5))
        self.w_down2 = nn.Conv2d(int(width * 1), int(width * 1.5), 1)

        # -------- Bottleneck --------
        self.conv_bottom = SpectralConv2d_fast(int(width * 1.5), width * 2, modes1 // 3, modes2 // 3)
        self.w_bottom = nn.Conv2d(int(width * 1.5), width * 2, 1)

        # -------- Decoder --------
        self.conv_up2 = SpectralConv2d_fast(int(width * 3.5), int(width * 1.5), int(modes1 // 1.5), int(modes2 // 1.5))
        self.w_up2 = nn.Conv2d(int(width * 3.5), int(width * 1.5), 1)

        self.conv_up1 = SpectralConv2d_fast(int(width * 2.5), width, modes1, modes2)
        self.w_up1 = nn.Conv2d(int(width * 2.5), width, 1)

        # Output Projection
        self.fcout1 = nn.Linear(width, 256)
        self.fcout2 = nn.Linear(256, 3)

        # Pooling operation
        self.pool = nn.AvgPool2d(2)
    
    def forward(self, x):
        # Input Projection: [Batch, Nx, Ny, 6] --> [Batch, Nx, Ny, width]
        x = self.fcin(x)             
        x = x.permute(0, 3, 1, 2)    

        # -------- Encoder --------
        # Stage 1
        x1 = F.gelu(self.conv_down1(x) + self.w_down1(x)) 
        x1_down = self.pool(x1)

        # Stage 2
        x2 = F.gelu(self.conv_down2(x1_down) + self.w_down2(x1_down)) 
        x2_down = self.pool(x2)

        # -------- Bottleneck --------
        xb = F.gelu(self.conv_bottom(x2_down) + self.w_bottom(x2_down)) 

        # -------- Decoder --------
        # Stage 2 Up (with skip connection)
        x2_up = F.interpolate(xb, scale_factor=2, mode='bilinear', align_corners=False)
        x2_cat = torch.cat([x2_up, x2], dim=1)  
        x2_out = F.gelu(self.conv_up2(x2_cat) + self.w_up2(x2_cat))

        # Stage 1 Up (with skip connection)
        x1_up = F.interpolate(x2_out, scale_factor=2, mode='bilinear', align_corners=False)
        x1_cat = torch.cat([x1_up, x1], dim=1)  
        x1_out = F.gelu(self.conv_up1(x1_cat) + self.w_up1(x1_cat))

        # -------- Output Projection --------
        x = x1_out.permute(0, 2, 3, 1)   
        x = F.gelu(self.fcout1(x))
        y = self.fcout2(x)

        return y

    # ==============================================================================
    # Physics-Informed Loss Calculation Methods
    # ==============================================================================

    def loss_pde_fft(self, xx, yy):
        """
        Calculate PDE loss using FFT for spatial derivatives.
        Includes Data loss, Divergence-free loss, and Force-free loss.
        """
        B = self.forward(xx)
        
        # Data loss (MSE/LpLoss)
        loss_fno = myloss(yy.view(par.batch_size_2, -1), B.view(par.batch_size_2, -1))

        B = B.permute(1, 2, 0, 3) 
        bx, by, bz = B[..., 0], B[..., 1], B[..., 2]
        B_norm = torch.sqrt(bx**2 + by**2 + bz**2)

        bx_grad = self.deriv3d(bx)
        by_grad = self.deriv3d(by)
        bz_grad = self.deriv3d(bz)

        # Divergence-free loss
        loss_div = torch.mean(torch.abs(bx_grad[0] + by_grad[1] + bz_grad[2]) / (6 * B_norm + 1e-8)) * 10000

        # Force-free loss
        Jx = bz_grad[1] - by_grad[2]  # ∂Bz/∂y - ∂By/∂z
        Jy = bx_grad[2] - bz_grad[0]  # ∂Bx/∂z - ∂Bz/∂x
        Jz = by_grad[0] - bx_grad[1]  # ∂By/∂x - ∂Bx/∂y

        J_norm = torch.sqrt(Jx**2 + Jy**2 + Jz**2)
        J = torch.stack([Jx, Jy, Jz], dim=-1)
        
        JxB = torch.cross(J.reshape(-1, 3), B.reshape(-1, 3), dim=-1)
        JxB_norm = torch.sqrt(JxB[:, 0]**2 + JxB[:, 1]**2 + JxB[:, 2]**2).reshape(B_norm.shape)

        loss_force = torch.mean(torch.sum(JxB_norm / B_norm) / (torch.sum(J_norm) + 1e-8))

        return loss_fno, loss_div, loss_force

    def pde_forward(self, b_in, x_in):
        """
        Helper function to handle inputs for autograd-based PDE loss.
        """
        x = torch.cat([b_in, x_in], dim=-1)
        x = x.view(par.batch_size_2, par.Nx, par.Ny, 6)
        y = self.forward(x)
        y = y.view(-1, 3)
        return y

    def loss_pde(self, xx, yy):
        """
        Calculate PDE loss using PyTorch autograd for exact spatial derivatives.
        Includes Data loss, Divergence-free loss, and Force-free loss.
        """
        x = xx.view(-1, 6)
        b_in = x[:, :3]
        x_in = x[:, 3:] 
        
        y = self.pde_forward(b_in, x_in) 
        
        # Data loss
        loss_cube = myloss(y.view(par.batch_size_2, -1), yy.view(par.batch_size_2, -1))
        
        # Compute Jacobian via Autograd
        jac = jacobian(y, x_in)
        jac = jac.view(par.batch_size_2, par.Nx, par.Ny, 3, 3)
        jac = jac.permute(1, 2, 0, 3, 4) 
        
        y = y.reshape(par.batch_size_2, par.Nx, par.Ny, 3)
        y = y.permute(1, 2, 0, 3)  
        B_norm = torch.sqrt(y[..., 0]**2 + y[..., 1]**2 + y[..., 2]**2)
        
        # Extract derivative components
        bx_x = jac[..., 0, 0]
        by_y = jac[..., 1, 1]
        bz_z = jac[..., 2, 2]
        
        bx_y = jac[..., 0, 1]
        bx_z = jac[..., 0, 2]
        by_x = jac[..., 1, 0]
        by_z = jac[..., 1, 2]
        bz_x = jac[..., 2, 0]
        bz_y = jac[..., 2, 1]

        # Divergence-free loss
        loss_div = torch.mean(torch.abs(bx_x + by_y + bz_z) / (6 * B_norm + 1e-8)) * 10000

        # Force-free loss
        Jx = bz_y - by_z  # ∂Bz/∂y - ∂By/∂z
        Jy = bx_z - bz_x  # ∂Bx/∂z - ∂Bz/∂x
        Jz = by_x - bx_y  # ∂By/∂x - ∂Bx/∂y

        J_norm = torch.sqrt(Jx**2 + Jy**2 + Jz**2)
        J = torch.stack([Jx, Jy, Jz], dim=-1)
        
        JxB = torch.cross(J.reshape(-1, 3), y.reshape(-1, 3), dim=-1)
        JxB_norm = torch.sqrt(JxB[:, 0]**2 + JxB[:, 1]**2 + JxB[:, 2]**2).reshape(B_norm.shape)

        loss_force = torch.mean(torch.sum(JxB_norm / B_norm) / (torch.sum(J_norm.reshape(B_norm.shape)) + 1e-8))
        
        return loss_cube, loss_div, loss_force