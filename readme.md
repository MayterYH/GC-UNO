# GC-UNO: Real-Time Coronal Magnetic Field Extrapolation

This repository contains the implementation for real-time coronal magnetic field extrapolation using the Grid-Constrained U-shaped Neural Operator (GC-UNO) framework. 

Please note that these scripts are the original codes developed and utilized directly during my research. They have not yet undergone strict software standardization or refactoring, and I appreciate your understanding.

For a detailed description of the model and methodology, please refer to our published paper:
> **Application of a Grid-constrained U-shaped Neural Operator in Real-time Solar Corona Magnetic Field Extrapolation** 
DOI: https://doi.org/10.3847/1538-4365/ae4025

---

## 📂 Directory Structure

The repository is organized as follows:

* **`data/`**: Handles data loading and preprocessing.
  * `data_isee.py`: Scripts to format the 3D grid coordinates and magnetic field vectors.
* **`model/`**: Contains the core neural network architectures.
  * `net.py`: Defines the GC-UNO architecture and physics-informed loss calculations.
* **`train/`**: Contains the training loops and logic.
  * `train_gc_uno.py`: Manages the data-driven pre-training and physics-informed autoregressive training phases.
* **`src/`**: Contains configuration files and utilities.
  * `config.py`: Hyperparameters, data paths, and physical constraint weights.
  * *Other utility scripts.*
* **`weight/`**: Directory used to save and load the trained model weights (`.pth` files).
* **`result/`**: Directory where the final 3D magnetic field inference results are exported (e.g., `.mat` files).

### 🚀 Quick Start
To execute the model, run the main entry script located in the root directory:
```bash
python run_gc_uno.py
```

## 🙏 Acknowledgments

This project is built upon and greatly benefits from the following open-source resources and datasets. We express our sincere gratitude to the original authors and institutions for their contributions to the community:

* **Fourier Neural Operator (FNO)**
  * **Codebase:** [neuraloperator/neuraloperator](https://github.com/neuraloperator/neuraloperator)
  * **Paper:** Z. Li *et al.*, "Fourier Neural Operator for Parametric Partial Differential Equations," 2020, arXiv. DOI: [10.48550/ARXIV.2010.08895](https://doi.org/10.48550/ARXIV.2010.08895)

* **ISEE NLFFF Database**
  * **Dataset:** [ISEE Database for Nonlinear Force-Free Field of Solar Active Regions](https://hinode.isee.nagoya-u.ac.jp/nlfff_database/)
  * **Citation:** K. Kusano, H. Iijima, T. Kaneko, S. Masuda, T. Iju, and S. Inoue, "ISEE Database for Nonlinear Force-Free Field of Solar Active Regions," Hinode Science Center, Institute for Space-Earth Environmental Research, Nagoya University, 2021.