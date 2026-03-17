"""
@author: Hao Yang

This repository contains the implementation for real-time coronal magnetic field extrapolation using the GC-UNO framework.

Please note that these scripts are the original codes developed and utilized directly during my research. 
They have not undergone strict standardization or refactoring, and I appreciate your understanding.

We plan to continuously refine and update this repository. Upcoming improvements include

- Code Refactoring Clean up the codebase, remove redundant scripts, and improve overall readability.

- Standardization Modularize the GC-UNO and physical constraint (PINN) training pipelines for easier out-of-the-box usage.

- Documentation Provide detailed step-by-step tutorials, API references, and sample datasets for the coronal magnetic field extrapolation.

For a detailed description of the model and methodology, please refer to our published paper:

- Application of a Grid-constrained U-shaped Neural Operator in Real-time Solar Corona Magnetic Field Extrapolation 

- DOI [10.3847/1538-4365/ae4025]

"""

import os
import json
import random
import numpy as np
import scipy.io
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

import matplotlib.pyplot as plt

from src.utilities3 import *
from src.Adam import Adam
from src.config import par
from model.net import FNO2d
from data.data_isee import batch_data


# ==============================================================================
# Utility & Physics Functions
# ==============================================================================

def calculate_en_em(B, B_ref):
    """
    Calculate Energy Conservation (en) and Normalized Energy Difference (em).
    
    Args:
        B: Predicted magnetic field tensor
        B_ref: Reference magnetic field tensor
    """
    B = B.permute((3,1,2,0))
    B_ref = B_ref.permute((3,1,2,0))

    norm_B_ref = torch.linalg.norm(B_ref, dim=0)     
    norm_diff = torch.linalg.norm(B - B_ref, dim=0)  
    
    # Calculate en: 1 - sum(|B-B_ref|) / sum(|B_ref|)
    total_norm_diff = torch.sum(norm_diff)
    total_norm_ref = torch.sum(norm_B_ref)
    en = total_norm_diff / total_norm_ref
    
    # Calculate em: mean relative error on valid points
    valid_mask = norm_B_ref > 1e-8
    relative_errors = torch.zeros_like(norm_B_ref)
    relative_errors[valid_mask] = norm_diff[valid_mask] / norm_B_ref[valid_mask]
    
    em = torch.mean(relative_errors[valid_mask])
    return en, em

def Hgradient(f, h=par.h, dims=None):
    """
    High-precision gradient computation supporting arbitrary dimensions.
    Uses 4th-order central difference for inner points and 3rd-order 
    one-sided difference for boundaries.
    """
    device = f.device
    ndim = f.dim()
    dims = list(range(ndim)) if dims is None else dims
    if isinstance(h, (float, int)):
        h = [h] * len(dims)

    grads = []
    for i, d in enumerate(dims):
        n = f.size(d)
        grad = torch.zeros_like(f)

        def make_slice(s):
            return tuple(s if idx == d else slice(None) for idx in range(ndim))

        try:
            if n >= 5:  # 4th-order central + 3rd-order boundary
                # Inner region
                inner_slice = make_slice(slice(2, -2))
                components = [
                    (-1, slice(4, None)),
                    (8, slice(3, -1)),
                    (-8, slice(1, -3)),
                    (1, slice(None, -4))
                ]
                numerator = sum(coeff * f[make_slice(sl)] for coeff, sl in components)
                grad[inner_slice] = numerator / (12 * h[i])

                # Left boundary
                for pos in [0, 1]:
                    coeffs = torch.tensor([-11, 18, -9, 2], device=device)
                    terms = [f[make_slice(slice(pos + k, pos + k + 1))] for k in range(4)]
                    grad[make_slice(pos)] = sum(c * t for c, t in zip(coeffs, terms)) / (6 * h[i])

                # Right boundary
                for pos in [n - 2, n - 1]:
                    coeffs = torch.tensor([2, -9, 18, -11], device=device)
                    terms = [f[make_slice(slice(max(0, pos - 3 + k), max(0, pos - 3 + k) + 1))] for k in range(4)]
                    grad[make_slice(pos)] = sum(c * t for c, t in zip(coeffs, terms)) / (6 * h[i])

            elif 3 <= n < 5:  # 2nd-order central
                inner_slice = make_slice(slice(1, -1))
                grad[inner_slice] = (f[make_slice(slice(2, None))] - f[make_slice(slice(None, -2))]) / (2 * h[i])
                grad[make_slice(0)] = (f[make_slice(1)] - f[make_slice(0)]) / h[i]
                grad[make_slice(-1)] = (f[make_slice(-1)] - f[make_slice(-2)]) / h[i]
            else:  
                grad = torch.gradient(f, dim=d, edge_order=2)[0] / h[i]
        except Exception as e:
            grad = torch.gradient(f, dim=d, edge_order=2)[0] / h[i]

        grads.append(grad)
    return grads

def B_Equ(y_in):
    """
    Compute physics-informed loss components: Divergence-free & Force-free conditions.
    """
    B = y_in.permute((1,2,0,3)) # Shape: (nx, ny, nz, 3)
    Bx = B[...,0]
    By = B[...,1]
    Bz = B[...,2]

    B_norm = torch.sqrt(Bx**2 + By**2 + Bz**2)
    
    if hasattr(par, 'if_H_2') and par.if_H_2:
        bx_grad = Hgradient(Bx)
        by_grad = Hgradient(By)
        bz_grad = Hgradient(Bz)
    else:
        bx_grad = torch.gradient(Bx, dim=(0, 1, 2), edge_order=2)
        by_grad = torch.gradient(By, dim=(0, 1, 2), edge_order=2)
        bz_grad = torch.gradient(Bz, dim=(0, 1, 2), edge_order=2)

    # 1. Divergence-free loss (∇ · B = 0)
    loss_div = torch.mean(torch.abs(bx_grad[0] + by_grad[1] + bz_grad[2]) / (6 * B_norm + 1e-8)) * 10000

    # 2. Force-free loss (J x B = 0)
    Jx = bz_grad[1] - by_grad[2]
    Jy = bx_grad[2] - bz_grad[0]
    Jz = by_grad[0] - bx_grad[1]

    J_norm = torch.sqrt(Jx**2 + Jy**2 + Jz**2)
    J = torch.stack([Jx, Jy, Jz], dim=-1)
    
    JxB = torch.cross(J.reshape(-1,3), B.reshape(-1,3), dim=-1)
    JxB_norm = torch.sqrt(JxB[:, 0]**2 + JxB[:, 1]**2 + JxB[:, 2]**2).reshape(B_norm.shape)

    loss_force = torch.mean(torch.sum(JxB_norm / B_norm) / (torch.sum(J_norm) + 1e-8))

    return loss_div, loss_force

def loss_metrics(y_hat: torch.Tensor, yy: torch.Tensor, eps: float = 1e-8):
    """
    Calculate normalized energy difference loss (loss_em).
    """
    error_vector = y_hat - yy
    error_norm = torch.linalg.norm(error_vector, ord=2, dim=-1)
    nlf_norm = torch.linalg.norm(yy, ord=2, dim=-1)
    
    em_terms = error_norm / (nlf_norm + eps)
    mask = nlf_norm > eps
    valid_em_terms = torch.where(mask, em_terms, torch.tensor(float('nan'), device=em_terms.device))
    
    loss_em = torch.nanmean(valid_em_terms)
    return loss_em

def run():

    # ==============================================================================
    # Initialization & Setup
    # ==============================================================================

    # Set random seeds for reproducibility
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Save configurations
    os.makedirs('par', exist_ok=True)
    os.makedirs('model', exist_ok=True)
    os.makedirs('result', exist_ok=True)

    args_dict = vars(par)
    with open('par/%s.json' % par.info, 'w') as json_file:
        json.dump(args_dict, json_file, indent=4)

    print(f"Starting training process - Run ID: {par.counter}")
    path = '2D-t-%d.pth' % par.counter
    path_model_fno = 'weight/fno-' + path
    path_model_pde = 'weight/pde-' + path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    train_loader, train_loader_order, test_loader = batch_data()

    # Initialize Model
    model = FNO2d(par.modes_1, par.modes_2, par.width).to(device)
    optimizer = Adam(model.parameters(), lr=par.lr_1, weight_decay=par.weight_decay_1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=par.sch_step_1, gamma=par.sch_gamma_1)

    fnoloss = LpLoss(size_average=True)


    # ==============================================================================
    # Step 1: Data-Driven Pre-training (U-NO)
    # ==============================================================================
    print("=== Step 1: Data-Driven Pre-training ===")
    for ep in tqdm(range(par.epochs_1)):
        model.train()
        for ii, (xx, yy) in enumerate(train_loader):

            if ii % 1000 == 0:
                print('Epoch: %d, Iteration: %d' % (ep, ii))

            xx = xx.requires_grad_().to(device)
            yy = yy.to(device)

            im = model(xx)

            loss_fno = fnoloss(im.reshape(par.batch_size_1, -1), yy.reshape(par.batch_size_1, -1))
            loss_em = loss_metrics(im, yy)
            loss = loss_fno + 1.0 * loss_em

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()

    # Save pre-trained state
    torch.save(model, './weight/fno_s1_%s_%s_%s_%s_%s.pth' % (par.sharp_cut, par.modes_1, par.width, par.tag, par.tag_num))


    # ==============================================================================
    # Step 2: Physics-Informed Autoregressive Training (GC-UNO)
    # ==============================================================================
    print("=== Step 2: Physics-Informed Autoregressive Training ===")
    # Adjust learning rate for physics-informed tuning if needed
    optimizer_2 = Adam(model.parameters(), lr=par.lr_2, weight_decay=par.weight_decay_2)
    scheduler_2 = torch.optim.lr_scheduler.StepLR(optimizer_2, step_size=par.sch_step_2, gamma=par.sch_gamma_2)

    # Using 'epochs_1' or 'epochs_2' depending on your config setup
    epochs_phase_2 = getattr(par, 'epochs_2', par.epochs_2)

    for ep in tqdm(range(epochs_phase_2)):
        model.train()

        loss_fno_ep, loss_div_ep, loss_force_ep = 0.0, 0.0, 0.0

        for i, (xx, yy) in enumerate(train_loader_order):
            # Initialize tensors for sequence generation
            nx_jump = par.Nx // par.jump if hasattr(par, 'jump') else par.Nx
            ny_jump = par.Ny // par.jump if hasattr(par, 'jump') else par.Ny

            y_hat = torch.zeros([par.z_cut, nx_jump, ny_jump, 3]).to(device)
            x_hat_temp = torch.zeros([1, nx_jump, ny_jump, 3]).to(device)
            loss_fno = 0.0

            xx = xx.reshape(1, nx_jump, ny_jump, 6).to(device)
            yy = yy.reshape(par.z_cut, nx_jump, ny_jump, 3).to(device)
            xx.requires_grad_(True)

            # Autoregressive generation along the z-axis (height)
            for ii in range(par.z_cut):
                if ii >= par.z_cut:
                    break

                if ii >= 1:
                    xx_new = xx.clone()
                    xx_new[0, :, :, :3] = x_hat_temp.detach()  # Inject previous prediction
                    xx_new[0, :, :, -1] = xx_new[0, :, :, -1] + 1.0  # Increment spatial index
                    xx = xx_new

                # Using gradient checkpointing to save memory
                # im = checkpoint(model, xx, use_reentrant=False)
                im = model(xx)

                y_hat[ii, :, :, :] = im
                x_hat_temp = im

            # 1. Data-driven Loss
            loss_fno = fnoloss(y_hat.reshape(par.z_cut, -1), yy.reshape(par.z_cut, -1))

            # 2. Physics-Informed Loss (∇·B = 0, JxB = 0)
            loss_div, loss_force = B_Equ(y_hat)

            # Combine all loss terms
            loss = (par.w_cube * loss_fno +
                    par.w_div * loss_div +
                    par.w_force * loss_force)

            optimizer_2.zero_grad()
            loss.backward()
            optimizer_2.step()

            # Track losses
            loss_fno_ep += loss_fno.item()
            loss_div_ep += loss_div.item()
            loss_force_ep += loss_force.item()

        scheduler_2.step()

        # num_batches = len(train_loader_order)
        # print('Epoch %d - Cube: %.3f | Force: %.3f | Div: %.3f | Em: %.3f' %
        #       (ep, loss_fno_ep/num_batches, loss_force_ep/num_batches, loss_div_ep/num_batches, loss_em_ep/num_batches))

    # Save the final physically constrained model
    torch.save(model, path_model_fno)

    # ==============================================================================
    # Step 3: Evaluation & Saving Results
    # ==============================================================================
    print("=== Step 3: Inference and Saving Results ===")
    result = torch.zeros([par.Nz - 1, par.Nx, par.Ny, 3])

    with torch.no_grad():
        x_hat = torch.zeros([1, par.Nx, par.Ny, 3]).to(device)
        for t, (x, y) in tqdm(enumerate(test_loader), total=par.Nz - 1):
            x = x.to(device)

            if t >= 1:
                x[0, :, :, :3] = x_hat

            im = model(x)
            result[t, :, :, :] = im.cpu()
            x_hat = im

    # Export predictions to .mat file for downstream analysis
    scipy.io.savemat('result/gc-uno-' + path + '.mat', mdict={'pred': result.numpy()})

    print('Finish!!!')