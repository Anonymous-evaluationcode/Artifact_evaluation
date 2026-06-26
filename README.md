# LiquidLens Artifact Evaluation

This repository contains the artifact for the paper *"Detecting Liquid Food Adulteration with Mobile Hyperspectral Analysis based on RGB Camera"*.

## 📁 Repository Structure
├── dataset/ # Pre-split spectral datasets (honey+wine, orange juice)
├── model/ # Pre-trained model weights (.pth files)
│ ├── best_student_model.pth
│ ├── best_teacher_model.pth
│ └── best_nir_model.pth
├── output/ # Output directory for results
├── LiquidLens_VIS-NIR_reconstruction_test.py
├── LiquidLens_VIS-NIR_reconstruction_train_test.py
├── LiquidLens_adulteration_detection.py
├── LiquidLens_transferlearning.py
├── requirements.txt
└── README.md


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

# 2: Run spectral recovery evaluation
bash
python LiquidLens_VIS-NIR_reconstruction_test.py
Expected output: MRAE ≈ 0.075, RMSE ≈ 0.042 (Table 3)

# 3: Run adulteration detection
bash
python LiquidLens_adulteration_detection.py
Expected output: Honey 96.2%, Wine 94.0% (Figures 15-16)

# 4: Run transfer learning (orange juice)
bash
python LiquidLens_transferlearning.py

