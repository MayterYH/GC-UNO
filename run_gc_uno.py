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
import sys
import os

sys.path.append(os.getcwd())

from train.gc_uno import run

if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        print("\n❌ Training interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        # raise e # Uncomment this to see full traceback for debugging
        sys.exit(1)