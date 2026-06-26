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


# 🚀 Quick Start

## 1. Set Up the Environment

### Requirements

- Python 3.7
- pip

### Create and activate a virtual environment

Create a virtual environment using **Python 3.7**.

**Linux / macOS**

```bash
python -m venv venv
source venv/bin/activate
```

**Windows**

```powershell
python -m venv venv
venv\Scripts\activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## 2. Run Evaluations

The repository includes pre-trained model weights.

You can directly reproduce the reported results without retraining.

### Option A: Run All Evaluations (Recommended)

#### Linux / macOS

```bash
chmod +x run_all.sh
./run_all.sh
```

#### Windows

```bat
run_all.bat
```

This script sequentially performs the following evaluations:

1. Spectral reconstruction
2. Adulteration detection
3. Transfer learning on orange juice

---

### Retrain the VIS–NIR Spectral Recovery Models

If you would like to retrain the VIS–NIR spectral recovery models instead of using the provided pre-trained weights:

#### Linux / macOS

```bash
chmod +x run_train_and_test.sh
./run_train_and_test.sh
```

#### Windows

```bat
run_train_and_test.bat
```

---

### Option B: Run Each Script Individually

#### 1. Spectral Reconstruction Evaluation

```bash
python LiquidLens_VIS-NIR_reconstruction_test.py
```

#### 2. Adulteration Detection

```bash
python LiquidLens_adulteration_detection.py
```

#### 3. Transfer Learning on Orange Juice

```bash
python LiquidLens_transferlearning.py
```

#### 4. Retrain the VIS–NIR Spectral Recovery Models

```bash
python LiquidLens_VIS-NIR_reconstruction_train_test.py
```




