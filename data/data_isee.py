"""
This module handles data loading and preprocessing for the ISEE solar coronal magnetic field dataset.
It prepares the 3D grid coordinates and corresponding magnetic field vectors for model training and testing.
"""

import torch
import numpy as np
import scipy.io
from src.config import par

# Set random seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)

def load_data(sharp_id):
    """
    Loads and preprocesses magnetic field data for a specific active region (SHARP ID).
    
    Args:
        sharp_id (str): The specific SHARP region ID to load (e.g., '11158').
        
    Returns:
        train_dataset (TensorDataset): Dataset formatted for standard training (next-step prediction).
        test_dataset (TensorDataset): Dataset formatted for full volume autoregressive testing.
    """
    filename_in = f'./data/ISEE/nlf_{sharp_id}_h32.dat'
    
    # Load .mat file
    B = scipy.io.loadmat(filename_in)['B']
    
    # Rearrange dimensions to (Nz, Nx, Ny, B_xyz) -> e.g., (188, 372, 188, 3)
    B = B.transpose((2, 0, 1, 3))
    B = B.astype(np.float32)
    
    Nx = par.Nx
    Ny = par.Ny
    Nz = par.Nz
    
    # Grid scaling factors
    chap_xy = 2
    chap_z = 1
    
    # Generate 3D spatial coordinates
    x = np.linspace(-512 // chap_xy, 512 // chap_xy - 1, 512)
    y = np.linspace(-256 // chap_xy, 256 // chap_xy - 1, 256)
    z = np.linspace(0, 256 // chap_z - 1, 256)
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    coor = np.stack([X, Y, Z], axis=3)
    coor = coor.transpose((2, 0, 1, 3)).astype(np.float32)  # Shape: (Nz, Nx, Ny, 3)
    
    # Apply vertical cutoff for regions other than the primary test case ('11158')
    if sharp_id == '11158':
        pass # Keep full volume
    else:
        B = B[:par.z_in, :, :, :]
        coor = coor[:par.z_in, :, :, :]
        
    # Concatenate Magnetic Field (3) and Coordinates (3) -> Shape: (Nz, Nx, Ny, 6)
    train_data = np.concatenate([B, coor], axis=-1)
    
    # Formulate inputs (z) and targets (z+1) for sequence prediction
    train_data_xx = train_data[:-1, :, :, :]      # Inputs: B_xyz + coords at height z
    train_data_yy = train_data[1:, :, :, :3]      # Targets: B_xyz at height z+1
    
    train_data_x = torch.Tensor(train_data_xx)
    train_data_y = torch.Tensor(train_data_yy)
    train_dataset = torch.utils.data.TensorDataset(train_data_x, train_data_y)
    
    # Formulate testing/autoregressive dataset 
    # Extract bottom boundary condition (z=0) with optional spatial downsampling
    test_data_xx = train_data_xx[0, ::par.jump, ::par.jump, :].reshape(1, Nx // par.jump, Ny // par.jump, 6)
    test_data_yy = train_data_yy[:par.z_cut, ::par.jump, ::par.jump, :].reshape(1, par.z_cut, Nx // par.jump, Ny // par.jump, 3)
    
    test_data_x = torch.Tensor(test_data_xx)
    test_data_y = torch.Tensor(test_data_yy)
    test_dataset = torch.utils.data.TensorDataset(test_data_x, test_data_y)
    
    return train_dataset, test_dataset


def batch_data():
    """
    Orchestrates the loading of multiple active regions to create 
    batched DataLoaders for training and evaluation.
    
    Returns:
        train_loader_ii (DataLoader): Shuffled loader for individual step training.
        train_loader_j (DataLoader): Loader for sequence/autoregressive training.
        test_loader (DataLoader): Loader for evaluation (using region 11158 as standard).
    """
    
    # List of active regions to use
    sharp_ids = [
        "11158","11078", "11089", "11092", "11093", "11108", "11109", "11117", "11130", "11131", "11133",
        "11140", "11163", "11164", "11165", "11166", "11169", "11176", "11183", "11190", "11195",
        "11196", "11199", "11236", "11257", "11260", "11261", "11263", "11271", "11283", "11289",
        "11301", "11302", "11305", "11312", "11316", "11325", "11327", "11330", "11339", "11354",
        "11358", "11362", "11363", "11374", "11384", "11386", "11387", "11389", "11390", "11391",
        "11393", "11402", "11416", "11422", "11428", "11429", "11455", "11459", "11460", "11465",
        "11471", "11476", "11484", "11486", "11492", "11497", "11512", "11515", "11520", "11543",
        "11555", "11560", "11562", "11564", "11579", "11582", "11585", "11589", "11591", "11596",
        "11598", "11613", "11618", "11620", "11635", "11640", "11652", "11654", "11660", "11665",
        "11682", "11698", "11711", "11718", "11719", "11723", "11726", "11730", "11731", "11736",
        "11745", "11748", "11755", "11765", "11776", "11777", "11793", "11818", "11827", "11835",
        "11877", "11884", "11890", "11936", "11944", "11967", "12089", "12109", "12121", "12135",
        "12144", "12146", "12149", "12152", "12158", "12173", "12175", "12177", "12186", "12192",
        "12203", "12205", "12209", "12216", "12217", "12219", "12221", "12222", "12232", "12242",
        "12297", "12371", "12403", "12422", "12489", "12494", "12544", "12738", "12816", "12936",
        "12975"
    ]
    trainDatasets_ii = []
    trainDatasets_j = []
    
    # Load a subset of regions based on config `par.sharp_cut`
    selected_regions = sharp_ids[-(par.sharp_cut * 2) - 1:]
    
    for ar in selected_regions:
        print(f"Loading Active Region: {ar}")
        train_dataset_ii, train_dataset_j = load_data(ar)
        trainDatasets_ii.append(train_dataset_ii)  
        trainDatasets_j.append(train_dataset_j)

    # Combine datasets
    TrainDataset_ii = torch.utils.data.ConcatDataset(trainDatasets_ii) 
    TrainDataset_j = torch.utils.data.ConcatDataset(trainDatasets_j)

    # Create DataLoaders
    train_loader_ii = torch.utils.data.DataLoader(  
        TrainDataset_ii,
        batch_size=par.batch_size_1,
        shuffle=True,
        drop_last=True
    )

    train_loader_j = torch.utils.data.DataLoader(
        TrainDataset_j,
        batch_size=1,
        shuffle=True
    )

    # Use '11158' strictly as the testing/evaluation loader
    train_dataset_0, _ = load_data('11158')
    test_loader = torch.utils.data.DataLoader(train_dataset_0, batch_size=1, shuffle=False)

    return train_loader_ii, train_loader_j, test_loader