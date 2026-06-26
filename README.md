# LiquidLens Artifact Evaluation

This repository contains the artifact for the paper *"Detecting Liquid Food Adulteration with Mobile Hyperspectral Analysis based on RGB Camera"*.

## 📁 Repository Structure
```bash
.
├── dataset/                                    # Pre-split spectral datasets (honey+wine, orange juice) and RGB images datasets
├── model/                                      # Pre-trained model weights (.pth files)
│   ├── best_student_model.pth                  # Flash-adapted VIS reconstructor (RGB → 400-750nm)
│   ├── best_teacher_model.pth                  # Halogen-trained VIS reconstructor (for ECDA distillation)
│   └── best_nir_model.pth                      # NIR predictor (750-1000nm)
├── output/                                     # Output directory
├── LiquidLens_VIS-NIR_reconstruction_test.py   # Evaluation only (uses pre-trained models)
├── LiquidLens_VIS-NIR_reconstruction_train_test.py  # Train + evaluation (optional)
├── LiquidLens_adulteration_detection.py        # Adulteration classification
├── LiquidLens_transferlearning.py              # Transfer learning on orange juice (unknown liquid)
├── requirements.txt                            # Python package dependencies
└── README.md                                   # This file
```


## 🚀 Quick Start

# 1. Set up environment

# Install Python 3.7.8 (using pyenv recommended)
pyenv install 3.7.8

pyenv local 3.7.8

# Create and activate virtual environment
python -m venv venv

source venv/bin/activate  # Linux/macOS

# or

venv\Scripts\activate     # Windows

# Install dependencies

pip install -r requirements.txt


# 2. Run evaluations
Option A: One-click run all evaluations (recommended)

Linux / macOS:

chmod +x run_all.sh

./run_all.sh

Windows:


run_all.bat

This will automatically run all three evaluations in sequence:

a. Spectral reconstruction test

b. Adulteration detection

c. Transfer learning on orange juice


Option B: Run each script individually



a. Run spectral recovery evaluation

python LiquidLens_VIS-NIR_reconstruction_test.py

Expected output: MRAE ≈ 0.075, RMSE ≈ 0.042 (Table 3)

b. Run adulteration detection

python LiquidLens_adulteration_detection.py

Expected output: Honey 96.2%, Wine 94.0% (Figures 15-16)

c. Run transfer learning (orange juice)

python LiquidLens_transferlearning.py

