import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import math
import os
import matplotlib.pyplot as plt
import seaborn as sns

TARGET_SPECTRUM_START_COL_NAME = '454nm'
TARGET_SPECTRUM_END_COL_NAME = '998nm'
HALO_RGB_COLS = ['R_Ori', 'G_Ori', 'B_Ori']
LED_RGB_COLS = ['R', 'G', 'B']
CLASS_PREFIX = ''
TEST_RATIO = 0.3
RANDOM_SEED = 42
BATCH_SIZE = 32
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 1e-4
EPOCHS = 200
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if DEVICE.type == 'cuda':
    torch.cuda.manual_seed_all(RANDOM_SEED)


class MRAELoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, predictions, targets):
        abs_error = torch.abs(predictions - targets)
        relative_error = abs_error / (torch.abs(targets) + self.eps)
        return torch.mean(relative_error) * 100


class FeatureAlignmentLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, feat_led, feat_halo):
        return torch.mean((feat_led - feat_halo) ** 2)


def load_and_prepare_data(file_paths):
    all_dfs = []

    for i, file_path in enumerate(file_paths):
        try:
            df = pd.read_csv(file_path)
            print(f"File {i + 1}: {file_path}")
            print(f"  Original samples: {len(df)}")

            df['file_source'] = f'file_{i + 1}'
            mask = df['Class'].str.startswith(CLASS_PREFIX)
            df_filtered = df[mask].reset_index(drop=True)

            if len(df_filtered) == 0:
                print(f"  Warning: No classes starting with '{CLASS_PREFIX}' found in file {i + 1}, using all data.")
                df_filtered = df.copy()
            else:
                print(f"  Filtered samples: {len(df_filtered)}")

            print(f"  Classes: {df_filtered['Class'].unique()}")
            all_dfs.append(df_filtered)

        except Exception as e:
            print(f"Error loading file {file_path}: {str(e)}")
            continue

    if not all_dfs:
        raise ValueError("No data files loaded successfully")

    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"\nTotal merged samples: {len(combined_df)}")
    print(f"Merged class distribution: {combined_df['Class'].value_counts()}")

    try:
        start_idx = combined_df.columns.get_loc(TARGET_SPECTRUM_START_COL_NAME)
        end_idx = combined_df.columns.get_loc(TARGET_SPECTRUM_END_COL_NAME)
        spectrum_cols = combined_df.columns[start_idx:end_idx + 1]

        if len(spectrum_cols) == 0:
            raise ValueError("No spectral columns found in specified range")

        print(f"Number of spectral bands: {len(spectrum_cols)}")
        return combined_df, spectrum_cols

    except Exception as e:
        print(f"Error processing spectral columns: {str(e)}")
        spectrum_cols = [col for col in combined_df.columns if 'nm' in col]
        if spectrum_cols:
            print(f"Auto-detected spectral columns: {len(spectrum_cols)} bands")
            return combined_df, spectrum_cols
        else:
            raise ValueError("Cannot find spectral columns")


def stratified_class_split(df, test_ratio=0.3, random_state=42):
    class_counts = df['Class'].value_counts()
    valid_classes = class_counts[class_counts >= 2].index
    df_valid = df[df['Class'].isin(valid_classes)].copy()
    if len(df_valid) < len(df):
        print(f"Warning: Removed {len(df) - len(df_valid)} samples (class count <2), cannot be used in stratified split")

    train_idx, test_idx = train_test_split(
        df_valid.index,
        test_size=test_ratio,
        stratify=df_valid['Class'],
        random_state=random_state
    )

    print(f"Stratified split complete: train samples = {len(train_idx)}, test samples = {len(test_idx)}")
    print(f"Train class distribution:\n{df_valid.loc[train_idx, 'Class'].value_counts()}")
    print(f"Test class distribution:\n{df_valid.loc[test_idx, 'Class'].value_counts()}")
    return train_idx, test_idx


def prepare_datasets(df, spectrum_cols):
    train_idx, test_idx = stratified_class_split(df, test_ratio=TEST_RATIO, random_state=RANDOM_SEED)
    wavelengths = [float(col.replace('nm', '')) for col in spectrum_cols]

    vis_end_idx = next((i for i, w in enumerate(wavelengths) if w > 750), len(wavelengths))
    nir_start_idx = vis_end_idx

    X_halo_train = df.loc[train_idx, HALO_RGB_COLS].values.astype(np.float32)
    X_halo_test = df.loc[test_idx, HALO_RGB_COLS].values.astype(np.float32)

    X_led_train = df.loc[train_idx, LED_RGB_COLS].values.astype(np.float32)
    X_led_test = df.loc[test_idx, LED_RGB_COLS].values.astype(np.float32)

    y_vis_train = df.loc[train_idx, spectrum_cols[:vis_end_idx]].values.astype(np.float32)
    y_vis_test = df.loc[test_idx, spectrum_cols[:vis_end_idx]].values.astype(np.float32)

    y_nir_train = df.loc[train_idx, spectrum_cols[nir_start_idx:]].values.astype(np.float32)
    y_nir_test = df.loc[test_idx, spectrum_cols[nir_start_idx:]].values.astype(np.float32)

    halo_scaler = StandardScaler()
    led_scaler = StandardScaler()

    X_halo_train = halo_scaler.fit_transform(X_halo_train)
    X_halo_test = halo_scaler.transform(X_halo_test)
    X_led_train = led_scaler.fit_transform(X_led_train)
    X_led_test = led_scaler.transform(X_led_test)

    return (X_halo_train, X_led_train, y_vis_train, y_nir_train,
            X_halo_test, X_led_test, y_vis_test, y_nir_test), (train_idx, test_idx), (vis_end_idx, nir_start_idx)


class ECDADataset(Dataset):
    def __init__(self, halo_rgb, led_rgb, vis_targets, nir_targets=None):
        self.halo_rgb = torch.tensor(halo_rgb, dtype=torch.float32)
        self.led_rgb = torch.tensor(led_rgb, dtype=torch.float32)
        self.vis_targets = torch.tensor(vis_targets, dtype=torch.float32)
        self.nir_targets = torch.tensor(nir_targets, dtype=torch.float32) if nir_targets is not None else None

    def __len__(self):
        return len(self.halo_rgb)

    def __getitem__(self, idx):
        if self.nir_targets is not None:
            return (self.halo_rgb[idx], self.led_rgb[idx],
                    self.vis_targets[idx], self.nir_targets[idx])
        else:
            return (self.halo_rgb[idx], self.led_rgb[idx],
                    self.vis_targets[idx])


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class SpectralAttentionBlock(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        self.spectral_weight = nn.Parameter(torch.ones(1))

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        attn_output, _ = self.self_attn(x, x, x)
        attn_output = self.spectral_weight * attn_output
        x = residual + self.dropout(attn_output)

        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + self.dropout(x)
        return x


class SpectralFusionModule(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.gmp = nn.AdaptiveMaxPool1d(1)

        self.interaction = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )

        self.fusion_proj = nn.Linear(4 * d_model, d_model)

    def forward(self, enc_feat, dec_feat):
        enc_feat_perm = enc_feat.permute(0, 2, 1)
        enc_gap = self.gap(enc_feat_perm).squeeze(-1)
        enc_gmp = self.gmp(enc_feat_perm).squeeze(-1)

        dec_feat_perm = dec_feat.permute(0, 2, 1)
        dec_gap = self.gap(dec_feat_perm).squeeze(-1)
        dec_gmp = self.gmp(dec_feat_perm).squeeze(-1)

        enc_mask = self.interaction(enc_gap)
        dec_mask = self.interaction(dec_gap)

        enc_enhanced = enc_mask.unsqueeze(1) * enc_feat
        dec_enhanced = dec_mask.unsqueeze(1) * dec_feat

        fused_feat = torch.cat([enc_feat, dec_feat, enc_enhanced, dec_enhanced], dim=-1)
        fused_feat = self.fusion_proj(fused_feat)
        return fused_feat


class MultistageSpectralReconstructor(nn.Module):
    def __init__(self, input_dim, output_dim, d_model=64, nhead=4, num_stages=2):
        super().__init__()
        self.d_model = d_model
        self.num_stages = num_stages

        self.input_proj = nn.Linear(input_dim, d_model)

        self.stages = nn.ModuleList([
            self._build_stage(d_model, nhead)
            for _ in range(num_stages)
        ])

        self.fusion_modules = nn.ModuleList([
            SpectralFusionModule(d_model)
            for _ in range(num_stages)
        ])

        self.pos_encoder = PositionalEncoding(d_model)

        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, output_dim)
        )

    def _build_stage(self, d_model, nhead):
        return nn.Sequential(
            SpectralAttentionBlock(d_model, nhead),
            SpectralAttentionBlock(d_model, nhead),
            SpectralAttentionBlock(d_model, nhead)
        )

    def forward(self, x, return_feature=False):
        x = self.input_proj(x)
        x = x.unsqueeze(1)
        x = self.pos_encoder(x)

        intermediate_features = []

        x = self.stages[0](x)
        intermediate_features.append(x)

        x = self.stages[1](x)
        intermediate_features.append(x)

        fused_feat = x
        for i in range(self.num_stages - 1, -1, -1):
            fused_feat = self.fusion_modules[i](intermediate_features[i], fused_feat)

        output = self.decoder(fused_feat.squeeze(1))

        if return_feature:
            feature = intermediate_features[0].squeeze(1)
            return output, feature
        return output


class SparseGatedNIRPredictor(nn.Module):
    def __init__(self, input_dim, output_dim, d_model=64, nhead=4, num_layers=3):
        super().__init__()
        self.d_model = d_model

        self.sparse_gate = nn.Parameter(torch.ones(1, input_dim))
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=256,
            dropout=0.1, activation='gelu', batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.decoder = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.constant_(self.sparse_gate, 1.0)

    def forward(self, x):
        gate_weights = torch.sigmoid(self.sparse_gate)
        x = x * gate_weights
        x = self.input_proj(x)
        x = x.unsqueeze(1)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.squeeze(1)
        return self.decoder(x)


def calculate_fair_metrics(predictions, targets):
    preds = predictions.numpy() if isinstance(predictions, torch.Tensor) else predictions
    trues = targets.numpy() if isinstance(targets, torch.Tensor) else targets

    rmse = np.sqrt(np.mean((preds - trues) ** 2))
    abs_error = np.abs(preds - trues)
    relative_error = abs_error / (np.abs(trues) + 1e-8)
    mrae = np.mean(relative_error) * 100

    return {'rmse': rmse, 'mrae': mrae}


def validate_vis(model, loader, criterion, input_type='halo'):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for halo_rgb, led_rgb, vis_targets, _ in loader:
            inputs = halo_rgb if input_type == 'halo' else led_rgb
            inputs, targets = inputs.to(DEVICE), vis_targets.to(DEVICE)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            all_preds.append(outputs.cpu())
            all_targets.append(targets.cpu())

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    metrics = calculate_fair_metrics(all_preds, all_targets)
    return total_loss / len(loader), metrics


def validate_nir(model, loader, criterion, vis_model):
    model.eval()
    vis_model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for halo_rgb, led_rgb, vis_targets, nir_targets in loader:
            led_rgb, nir_targets = led_rgb.to(DEVICE), nir_targets.to(DEVICE)

            vis_reconstructed = vis_model(led_rgb)
            outputs = model(vis_reconstructed)
            loss = criterion(outputs, nir_targets)
            total_loss += loss.item()

            all_preds.append(outputs.cpu())
            all_targets.append(nir_targets.cpu())

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    metrics = calculate_fair_metrics(all_preds, all_targets)
    return total_loss / len(loader), metrics


if __name__ == "__main__":
    TRAIN_FILE = "dataset/LiquidLens_honey_wine_spectra_train.csv"
    TEST_FILE = "dataset/LiquidLens_honey_wine_spectra_test.csv"

    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)

    if CLASS_PREFIX:
        train_df = train_df[train_df['Class'].str.startswith(CLASS_PREFIX)].reset_index(drop=True)
        test_df = test_df[test_df['Class'].str.startswith(CLASS_PREFIX)].reset_index(drop=True)

    try:
        start_idx = train_df.columns.get_loc(TARGET_SPECTRUM_START_COL_NAME)
        end_idx = train_df.columns.get_loc(TARGET_SPECTRUM_END_COL_NAME)
        spectrum_cols = train_df.columns[start_idx:end_idx + 1]
    except:
        spectrum_cols = [col for col in train_df.columns if 'nm' in col]

    wavelengths = [float(col.replace('nm', '')) for col in spectrum_cols]
    vis_end_idx = next((i for i, w in enumerate(wavelengths) if w > 750), len(wavelengths))
    nir_start_idx = vis_end_idx

    X_halo_train_for_scaler = train_df[HALO_RGB_COLS].values.astype(np.float32)
    X_led_train_for_scaler = train_df[LED_RGB_COLS].values.astype(np.float32)

    X_halo_test = test_df[HALO_RGB_COLS].values.astype(np.float32)
    X_led_test = test_df[LED_RGB_COLS].values.astype(np.float32)
    y_vis_test = test_df[spectrum_cols[:vis_end_idx]].values.astype(np.float32)
    y_nir_test = test_df[spectrum_cols[nir_start_idx:]].values.astype(np.float32)

    halo_scaler = StandardScaler()
    led_scaler = StandardScaler()
    halo_scaler.fit(X_halo_train_for_scaler)
    led_scaler.fit(X_led_train_for_scaler)

    X_halo_test = halo_scaler.transform(X_halo_test)
    X_led_test = led_scaler.transform(X_led_test)

    vis_wavelengths = wavelengths[:vis_end_idx]
    nir_wavelengths = wavelengths[nir_start_idx:]

    vis_wavelengths = wavelengths[:vis_end_idx]
    nir_wavelengths = wavelengths[nir_start_idx:]


    test_dataset = ECDADataset(X_halo_test, X_led_test, y_vis_test, y_nir_test)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    teacher_model = MultistageSpectralReconstructor(
        input_dim=len(HALO_RGB_COLS), output_dim=len(vis_wavelengths)
    ).to(DEVICE)

    student_model = MultistageSpectralReconstructor(
        input_dim=len(LED_RGB_COLS), output_dim=len(vis_wavelengths)
    ).to(DEVICE)

    nir_model = SparseGatedNIRPredictor(
        input_dim=len(vis_wavelengths), output_dim=len(nir_wavelengths)
    ).to(DEVICE)

    teacher_model.load_state_dict(torch.load('model/best_teacher_model.pth', map_location=DEVICE))
    student_model.load_state_dict(torch.load('model/best_student_model.pth', map_location=DEVICE))
    nir_model.load_state_dict(torch.load('model/best_nir_model.pth', map_location=DEVICE))

    teacher_model.eval()
    student_model.eval()
    nir_model.eval()
    print("\nPretrained models loaded successfully.")

    criterion = MRAELoss()

    _, teacher_metrics = validate_vis(teacher_model, test_loader, criterion, 'halo')
    _, student_metrics = validate_vis(student_model, test_loader, criterion, 'led')
    _, nir_metrics = validate_nir(nir_model, test_loader, criterion, student_model)

    print("\n=== Final Test Results ===")
    print("Teacher Model (Halo):")
    print(f"  RMSE: {teacher_metrics['rmse']:.4f}, MRAE: {teacher_metrics['mrae']:.2f}%")

    print("Student Model (LED):")
    print(f"  RMSE: {student_metrics['rmse']:.4f}, MRAE: {student_metrics['mrae']:.2f}%")

    print("NIR Prediction:")
    print(f"  RMSE: {nir_metrics['rmse']:.4f}, MRAE: {nir_metrics['mrae']:.2f}%")

    print("\n=== Visible + NIR Full Spectrum Evaluation ===")

    with torch.no_grad():
        test_halo_rgb = torch.tensor(X_halo_test, dtype=torch.float32).to(DEVICE)
        test_led_rgb = torch.tensor(X_led_test, dtype=torch.float32).to(DEVICE)

        teacher_vis_pred = teacher_model(test_halo_rgb).cpu().numpy()
        student_vis_pred = student_model(test_led_rgb).cpu().numpy()
        nir_pred = nir_model(torch.tensor(student_vis_pred, dtype=torch.float32).to(DEVICE)).cpu().numpy()

        full_spectrum_pred = np.concatenate([student_vis_pred, nir_pred], axis=1)
        full_spectrum_true = np.concatenate([y_vis_test, y_nir_test], axis=1)

        full_rmse = np.sqrt(np.mean((full_spectrum_pred - full_spectrum_true) ** 2))
        full_abs_error = np.abs(full_spectrum_pred - full_spectrum_true)
        full_relative_error = full_abs_error / (np.abs(full_spectrum_true) + 1e-8)
        full_mrae = np.mean(full_relative_error) * 100

        band_rmse = np.sqrt(np.mean((full_spectrum_pred - full_spectrum_true) ** 2, axis=0))
        band_mrae = np.mean(full_relative_error, axis=0) * 100

        print(f"Full spectrum evaluation metrics:")
        print(f"  RMSE: {full_rmse:.4f}")
        print(f"  MRAE: {full_mrae:.2f}%")

        # ==========================================================
        # Visualization of RGB -> VIS -> NIR Reconstruction Pipeline
        # ==========================================================

        sample_idx = np.random.randint(0, len(test_df))

        true_vis = y_vis_test[sample_idx]
        true_nir = y_nir_test[sample_idx]

        pred_vis = student_vis_pred[sample_idx]
        pred_nir = nir_pred[sample_idx]

        true_full = np.concatenate([true_vis, true_nir])
        pred_full = np.concatenate([pred_vis, pred_nir])

        rgb_input = test_df.iloc[sample_idx][LED_RGB_COLS].values.astype(float)

        plt.style.use('seaborn-whitegrid')

        fig = plt.figure(figsize=(18, 10))

        gs = fig.add_gridspec(
            2, 3,
            height_ratios=[1, 1.2],
            hspace=0.35,
            wspace=0.3
        )

        # ======================================================
        # RGB Input
        # ======================================================

        ax0 = fig.add_subplot(gs[0, 0])

        rgb_norm = rgb_input / max(rgb_input.max(), 1)

        ax0.bar(
            ['R', 'G', 'B'],
            rgb_input,
            color=[
                (1, 0.3, 0.3),
                (0.3, 0.8, 0.3),
                (0.3, 0.5, 1)
            ],
            edgecolor='black',
            linewidth=1.5
        )

        ax0.set_title(
            "Input RGB",
            fontsize=16,
            fontweight='bold'
        )

        ax0.set_ylabel("Intensity")

        # ======================================================
        # VIS Reconstruction
        # ======================================================

        ax1 = fig.add_subplot(gs[0, 1])

        ax1.plot(
            vis_wavelengths,
            true_vis,
            '--',
            color='black',
            linewidth=2,
            label='Ground Truth'
        )

        ax1.plot(
            vis_wavelengths,
            pred_vis,
            color='#1f77b4',
            linewidth=3,
            label='Reconstructed VIS'
        )

        ax1.fill_between(
            vis_wavelengths,
            pred_vis,
            alpha=0.25,
            color='#1f77b4'
        )

        ax1.set_title(
            "Stage 1: VIS Reconstruction",
            fontsize=16,
            fontweight='bold'
        )

        ax1.set_xlabel("Wavelength (nm)")
        ax1.set_ylabel("Reflectance")
        ax1.legend()

        # ======================================================
        # NIR Prediction
        # ======================================================

        ax2 = fig.add_subplot(gs[0, 2])

        ax2.plot(
            nir_wavelengths,
            true_nir,
            '--',
            color='black',
            linewidth=2,
            label='Ground Truth'
        )

        ax2.plot(
            nir_wavelengths,
            pred_nir,
            color='#d62728',
            linewidth=3,
            label='Predicted NIR'
        )

        ax2.fill_between(
            nir_wavelengths,
            pred_nir,
            alpha=0.25,
            color='#d62728'
        )

        ax2.set_title(
            "Stage 2: NIR Prediction",
            fontsize=16,
            fontweight='bold'
        )

        ax2.set_xlabel("Wavelength (nm)")
        ax2.set_ylabel("Reflectance")
        ax2.legend()

        # ======================================================
        # Full Spectrum
        # ======================================================

        ax3 = fig.add_subplot(gs[1, :])

        ax3.axvspan(
            wavelengths[0],
            750,
            alpha=0.12,
            color='royalblue',
            label='Visible'
        )

        ax3.axvspan(
            750,
            wavelengths[-1],
            alpha=0.12,
            color='crimson',
            label='NIR'
        )

        ax3.plot(
            wavelengths,
            true_full,
            '--',
            color='black',
            linewidth=2.5,
            label='Ground Truth Spectrum'
        )

        ax3.plot(
            wavelengths,
            pred_full,
            color='#2ca02c',
            linewidth=3,
            label='Predicted Full Spectrum'
        )

        ax3.axvline(
            750,
            color='gray',
            linestyle=':',
            linewidth=2
        )

        ax3.text(
            620,
            np.max(true_full) * 0.95,
            "VIS",
            fontsize=14,
            fontweight='bold',
            color='navy'
        )

        ax3.text(
            830,
            np.max(true_full) * 0.95,
            "NIR",
            fontsize=14,
            fontweight='bold',
            color='darkred'
        )

        ax3.set_title(
            "Complete Spectral Reconstruction Pipeline",
            fontsize=18,
            fontweight='bold'
        )

        ax3.set_xlabel("Wavelength (nm)", fontsize=13)
        ax3.set_ylabel("Reflectance", fontsize=13)

        ax3.legend(
            loc='lower right',
            fontsize=11
        )

        # ======================================================
        # Flow arrows
        # ======================================================

        fig.text(
            0.34,
            0.7,
            "→",
            fontsize=40,
            fontweight='bold'
        )

        fig.text(
            0.62,
            0.7,
            "→",
            fontsize=40,
            fontweight='bold'
        )

        plt.suptitle(
            f"Example Spectral Reconstruction (Sample #{sample_idx})",
            fontsize=20,
            fontweight='bold'
        )

        plt.savefig(
            "output/pipeline_visualization.png",
            dpi=600,
            bbox_inches='tight'
        )

        plt.show()

        # ==========================================================
        # Per-Class Full Spectrum Reconstruction Visualization
        # ==========================================================

        # 全谱误差
        sample_errors = np.mean(
            np.abs(full_spectrum_pred - full_spectrum_true),
            axis=1
        )

        class_names = test_df['Class'].unique()

        selected_indices = []

        for cls in class_names:
            cls_idx = np.where(test_df['Class'].values == cls)[0]

            cls_errors = sample_errors[cls_idx]

            median_local = np.argsort(cls_errors)[len(cls_errors) // 2]

            selected_indices.append(cls_idx[median_local])

