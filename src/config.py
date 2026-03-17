"""
@author: Hao Yang (Modified)

Configuration file for the GC-UNO coronal magnetic field extrapolation model.
Handles hyperparameter settings, data paths, and physical constraint weights.
"""

import os
import argparse

# ==============================================================================
# Run Counter Management (Auto-increment run IDs)
# ==============================================================================
counter_file = "counter.txt"

# Create counter file if it doesn't exist (useful for fresh git clones)
if not os.path.exists(counter_file):
    with open(counter_file, "w") as f:
        f.write("1")

# Read current run counter
with open(counter_file, "r") as f:
    counter = int(f.read().strip())

parser = argparse.ArgumentParser(description="GC-UNO Configuration Parameters")

# ==============================================================================
# Global & System Settings
# ==============================================================================
group_sys = parser.add_argument_group("System Settings")
group_sys.add_argument('--info', type=str, default='training-%d' % counter, help='Run identifier')
group_sys.add_argument("--counter", type=int, default=counter, help='Current run counter')
group_sys.add_argument("--train_info", type=str, default='fno', help='Training phase info')
group_sys.add_argument("--net_type", type=str, default='Loss_F', help='Network loss type identifier')
group_sys.add_argument("--tag", type=str, default='U', help='Model architecture tag')
group_sys.add_argument("--tag_num", type=str, default='1002', help='Model version tag')

# ==============================================================================
# Data Parameters
# ==============================================================================
group_data = parser.add_argument_group("Data Parameters")
group_data.add_argument("--Nx", type=int, default=512, help='Grid size in X dimension')
group_data.add_argument("--Ny", type=int, default=256, help='Grid size in Y dimension')
group_data.add_argument("--Nz", type=int, default=256, help='Grid size in Z dimension (height)')
group_data.add_argument("--jump", type=int, default=1, help='Spatial downsampling factor')
group_data.add_argument("--z_in", type=int, default=256, help='Input Z height limit')
group_data.add_argument("--z_cut", type=int, default=5, help='Reducing it can save memory and may also lead to better extrapolation performance.')
group_data.add_argument("--sharp_id", type=int, default=11158, help='Primary Active Region SHARP ID')
group_data.add_argument("--sharp_cut", type=int, default=75, help='Number of Active Regions to load for training')

# ==============================================================================
# Model Architecture (GC-UNO / FNO)
# ==============================================================================
group_arch = parser.add_argument_group("Architecture Parameters")
group_arch.add_argument("--if_PINN", type=bool, default=True, help='Enable Physics-Informed Neural Network features')
group_arch.add_argument("--modes_1", type=int, default=84, help='Fourier modes in dimension 1')
group_arch.add_argument("--modes_2", type=int, default=84, help='Fourier modes in dimension 2')
group_arch.add_argument("--width", type=int, default=64, help='Base channel width of the network')
group_arch.add_argument("--h", type=float, default=0.05, help='Grid spacing for gradient calculation')

# ==============================================================================
# Phase 1: Data-Driven Pre-training
# ==============================================================================
group_t1 = parser.add_argument_group("Phase 1: Pre-training")
group_t1.add_argument("--epochs_1", type=int, default=50, help='Epochs for Phase 1')
group_t1.add_argument("--lr_1", type=float, default=1e-3, help='Learning rate for Phase 1')
group_t1.add_argument("--batch_size_1", type=int, default=2, help='Batch size for Phase 1')
group_t1.add_argument("--weight_decay_1", type=float, default=0, help='Weight decay (L2 penalty) for Phase 1')
group_t1.add_argument("--sch_step_1", type=int, default=5, help='Scheduler step size for Phase 1')
group_t1.add_argument("--sch_gamma_1", type=float, default=0.9, help='Scheduler gamma (decay rate) for Phase 1')

# ==============================================================================
# Phase 2: Physics-Informed Autoregressive Training
# ==============================================================================
group_t2 = parser.add_argument_group("Phase 2: Physics-Informed Training")
group_t2.add_argument("--epochs_2", type=int, default=20, help='Epochs for Phase 2')
group_t2.add_argument("--lr_2", type=float, default=1e-5, help='Learning rate for Phase 2')
group_t2.add_argument("--batch_size_2", type=int, default=1, help='Batch size for Phase 2 (Autoregressive)')
group_t2.add_argument("--weight_decay_2", type=float, default=0, help='Weight decay (L2 penalty) for Phase 2')
group_t2.add_argument("--sch_step_2", type=int, default=5, help='Scheduler step size for Phase 2')
group_t2.add_argument("--sch_gamma_2", type=float, default=0.6, help='Scheduler gamma (decay rate) for Phase 2')

# ==============================================================================
# Loss Weights & Physical Constraints
# ==============================================================================
group_loss = parser.add_argument_group("Loss Weights")
group_loss.add_argument("--w_cube", type=float, default=1.0, help='Weight for Data Loss (MSE/LpLoss)')
group_loss.add_argument("--w_div", type=float, default=0.001, help='Weight for Divergence-free Loss (∇·B=0);Our best result: 0.001 * 1.125')
group_loss.add_argument("--w_force", type=float, default=5.0, help='Weight for Force-free Loss (JxB=0);Our best result:4.704')
group_loss.add_argument("--w_cv", type=float, default=0, help='Weight for Vector Correlation Coefficient Loss')
group_loss.add_argument("--w_cc", type=float, default=0, help='Weight for Component Correlation Coefficient Loss')
group_loss.add_argument("--w_en", type=float, default=0, help='Weight for Energy Conservation Loss')
group_loss.add_argument("--w_em", type=float, default=0, help='Weight for Normalized Energy Difference Loss')
group_loss.add_argument("--w_norm", type=float, default=0, help='Weight for Norm Loss')
group_loss.add_argument("--loss_type", type=str, default='force', help='Target physics loss strategy')
group_loss.add_argument("--if_H_PINN", type=bool, default=False, help='Use High-Precision gradient calculation')
group_loss.add_argument("--if_fft_loss", type=bool, default=False, help='Calculate derivatives using FFT')

# ==============================================================================
# Experimental Flags & Ablation Studies
# ==============================================================================
group_exp = parser.add_argument_group("Experimental Flags")
group_exp.add_argument("--if_fno_s", type=bool, default=False, help='Enable weighted short-sequence loss')
group_exp.add_argument("--fno_sindex", type=int, default=5, help='Index limit for short-sequence calculation')
group_exp.add_argument("--fno_sw", type=float, default=1.0, help='Weight multiplier for short-sequence loss')
group_exp.add_argument("--if_add_w", type=bool, default=False, help='Add weighted scaling to height progression')
group_exp.add_argument("--data_in", type=bool, default=False, help='Experimental data flag')
group_exp.add_argument("--more_net", type=bool, default=True, help='Enable extended network branches')
group_exp.add_argument("--if_Pre", type=bool, default=False, help='Pre-processing execution flag')
group_exp.add_argument("--data_T", type=str, default='NLFFF', help='Data target type (e.g., NLFFF)')
group_exp.add_argument("--cut_type", type=str, default='dim', help='Type of tensor slicing')
group_exp.add_argument("--dim_test", type=int, default=1, help='Test dimension parameter')
group_exp.add_argument("--cut_test", type=int, default=60, help='Test slicing parameter')
group_exp.add_argument("--cut_x", type=int, default=93, help='Slice parameter X')
group_exp.add_argument("--cut_y", type=int, default=47, help='Slice parameter Y')

# Parse arguments
par = parser.parse_args()

# Update counter and write back to file
counter += 1
with open(counter_file, "w") as f:
    f.write(str(counter))