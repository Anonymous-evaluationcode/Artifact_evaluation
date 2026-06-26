import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import math
import os

# --- Fine-tuning configuration ---
NEW_DATA_FILE_PATHS = ['dataset/LiquidLens_juice_spectra.csv']  # path to new dataset
FINE_TUNE_EPOCHS = 1000
FINE_TUNE_LEARNING_RATE = 2.5e-5
FINE_TUNE_BATCH_SIZE = 16
FINE_TUNE_TEST_RATIO = 0.5
FINE_TUNE_WEIGHT_DECAY = 1e-5

# Early stopping patience based on training loss
EARLY_STOP_PATIENCE = 80

# Reuse configurations from original code
TARGET_SPECTRUM_START_COL_NAME = '454nm'
TARGET_SPECTRUM_END_COL_NAME = '998nm'
HALO_RGB_COLS = ['R_Ori', 'G_Ori', 'B_Ori']
LED_RGB_COLS = ['R', 'G', 'B']
CLASS_PREFIX = ''
RANDOM_SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Number of training samples per adulteration class
SAMPLES_PER_CLASS = 4

# --- Set random seeds ---
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if DEVICE.type == 'cuda':
    torch.cuda.manual_seed_all(RANDOM_SEED)


# --- Reused classes and functions from original code ---
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

        self.stages = nn.ModuleList()
        for _ in range(num_stages):
            stage = nn.Sequential(
                SpectralAttentionBlock(d_model, nhead),
                SpectralAttentionBlock(d_model, nhead),
                SpectralAttentionBlock(d_model, nhead)
            )
            self.stages.append(stage)

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

    def forward(self, x, return_feature=False):
        x = self.input_proj(x)
        x = x.unsqueeze(1)
        x = self.pos_encoder(x)

        intermediate_features = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
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


def load_and_prepare_fine_tune_data(file_paths):
    """Load fine-tuning data from CSV files."""
    all_dfs = []
    for i, file_path in enumerate(file_paths):
        try:
            df = pd.read_csv(file_path)
            print(f"Fine-tune file {i+1}: {file_path}")
            print(f"  Number of samples: {len(df)}")
            df['file_source'] = f'fine_tune_file_{i+1}'
            if 'Class' in df.columns:
                if CLASS_PREFIX:
                    mask = df['Class'].str.split('_').str[0].str.startswith(CLASS_PREFIX)
                else:
                    mask = pd.Series([True] * len(df))
            else:
                mask = pd.Series([True] * len(df))
            df_filtered = df[mask].reset_index(drop=True)
            if len(df_filtered) == 0:
                print(f"  Warning: using all data")
                df_filtered = df.copy()
            print(f"  Included classes: {df_filtered['Class'].unique() if 'Class' in df_filtered.columns else 'N/A'}")
            all_dfs.append(df_filtered)
        except Exception as e:
            print(f"Error loading file {file_path}: {str(e)}")
            continue
    if not all_dfs:
        raise ValueError("No fine-tuning data files loaded successfully")
    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"\nTotal fine-tuning samples: {len(combined_df)}")
    spectrum_cols = [col for col in combined_df.columns if
                     'nm' in col and col.replace('nm', '').replace('.', '').isdigit()]
    if spectrum_cols:
        spectrum_cols = sorted(spectrum_cols, key=lambda x: float(x.replace('nm', '')))
        print(f"Detected spectral columns: {len(spectrum_cols)} bands")
        return combined_df, spectrum_cols
    else:
        raise ValueError("Could not find spectral columns")


def prepare_fine_tune_datasets(df, spectrum_cols):
    """Prepare training and test sets with stratified sampling per class."""
    if 'Class' not in df.columns:
        raise ValueError("Dataset must contain 'Class' column for stratified sampling")
    classes = df['Class'].unique()
    print(f"Found {len(classes)} classes: {classes}")
    train_idx_list = []
    for cls in classes:
        cls_indices = df.index[df['Class'] == cls].tolist()
        n_available = len(cls_indices)
        n_select = min(SAMPLES_PER_CLASS, n_available)
        if n_select < SAMPLES_PER_CLASS:
            print(f"Warning: class '{cls}' has only {n_available} samples, using all of them.")
        selected = np.random.choice(cls_indices, size=n_select, replace=False)
        train_idx_list.extend(selected)
    train_idx = np.array(train_idx_list)
    test_idx = df.index[~df.index.isin(train_idx)]
    print(f"Training set: {len(train_idx)} samples (stratified per class)")
    print(f"Test set: {len(test_idx)} samples (will NOT be used during training)")
    if set(train_idx).intersection(set(test_idx)):
        print("Error: training and test sets overlap!")
    else:
        print("Training and test sets are disjoint")
    wavelengths = [float(col.replace('nm', '')) for col in spectrum_cols]
    vis_end_idx = next((i for i, w in enumerate(wavelengths) if w > 750), len(wavelengths))
    nir_start_idx = vis_end_idx
    print(f"Visible bands: {wavelengths[0]:.1f}nm - {wavelengths[vis_end_idx-1]:.1f}nm ({vis_end_idx} bands)")
    print(f"NIR bands: {wavelengths[nir_start_idx]:.1f}nm - {wavelengths[-1]:.1f}nm ({len(wavelengths)-nir_start_idx} bands)")
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


def calculate_fair_metrics(predictions, targets):
    """Calculate RMSE and MRAE (no R-value)."""
    preds = predictions.numpy() if isinstance(predictions, torch.Tensor) else predictions
    trues = targets.numpy() if isinstance(targets, torch.Tensor) else targets
    rmse = np.sqrt(np.mean((preds - trues) ** 2))
    abs_error = np.abs(preds - trues)
    relative_error = abs_error / (np.abs(trues) + 1e-8)
    mrae = np.mean(relative_error) * 100
    std_dev = np.std(trues)
    nrmse = rmse / std_dev if std_dev > 0 else rmse
    return {'rmse': rmse, 'mrae': mrae, 'nrmse': nrmse}


def validate_vis(model, loader, criterion, input_type='halo'):
    """Evaluate visible spectrum reconstruction on a given loader (used only for final test)."""
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
    """Evaluate NIR prediction on a given loader (used only for final test)."""
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


def load_pretrained_model_with_adaptation(model, checkpoint_path, target_output_dim):
    """Load pretrained model and adapt output dimension if needed."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        original_output_dim = checkpoint['decoder.2.weight'].shape[0]
        print(f"Pretrained model output dim: {original_output_dim}, current need: {target_output_dim}")
        if original_output_dim != target_output_dim:
            print("Output dimension mismatch, adapting...")
            model_state_dict = model.state_dict()
            adapted_state_dict = {}
            for key, value in checkpoint.items():
                if key in model_state_dict:
                    if 'decoder.2' in key:
                        if 'weight' in key:
                            original_weight = value
                            target_weight = model_state_dict[key]
                            if original_weight.shape[0] < target_weight.shape[0]:
                                adapted_weight = torch.zeros_like(target_weight)
                                min_dim = min(original_weight.shape[0], target_weight.shape[0])
                                adapted_weight[:min_dim] = original_weight[:min_dim]
                                if original_weight.shape[0] < target_weight.shape[0]:
                                    mean_weights = original_weight.mean(dim=0, keepdim=True)
                                    adapted_weight[original_weight.shape[0]:] = mean_weights.repeat(
                                        target_weight.shape[0] - original_weight.shape[0], 1)
                            else:
                                adapted_weight = original_weight[:target_weight.shape[0], :target_weight.shape[1]]
                            adapted_state_dict[key] = adapted_weight
                        elif 'bias' in key:
                            original_bias = value
                            target_bias = model_state_dict[key]
                            if original_bias.shape[0] < target_bias.shape[0]:
                                adapted_bias = torch.zeros_like(target_bias)
                                min_dim = min(original_bias.shape[0], target_bias.shape[0])
                                adapted_bias[:min_dim] = original_bias[:min_dim]
                                if original_bias.shape[0] < target_bias.shape[0]:
                                    mean_bias = original_bias.mean()
                                    adapted_bias[original_bias.shape[0]:] = mean_bias
                            else:
                                adapted_bias = original_bias[:target_bias.shape[0]]
                            adapted_state_dict[key] = adapted_bias
                    else:
                        adapted_state_dict[key] = value
                else:
                    print(f"Skipping mismatched key: {key}")
            model.load_state_dict(adapted_state_dict, strict=False)
            print("Successfully loaded and adapted pretrained model")
        else:
            model.load_state_dict(checkpoint)
            print("Successfully loaded pretrained model")
    except Exception as e:
        print(f"Warning: Could not load pretrained model: {e}")
        print("Training from scratch")


def load_pretrained_nir_with_adaptation(model, checkpoint_path, target_input_dim):
    """Load pretrained NIR model and adapt input dimension if needed."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        original_input_dim = checkpoint['sparse_gate'].shape[1]
        print(f"Pretrained NIR model input dim: {original_input_dim}, current need: {target_input_dim}")
        if original_input_dim != target_input_dim:
            print("Input dimension mismatch, adapting...")
            model_state_dict = model.state_dict()
            adapted_state_dict = {}
            for key, value in checkpoint.items():
                if key in model_state_dict:
                    if 'sparse_gate' in key:
                        original_gate = value
                        target_gate = model_state_dict[key]
                        if original_gate.shape[1] < target_gate.shape[1]:
                            adapted_gate = torch.ones_like(target_gate)
                            min_dim = min(original_gate.shape[1], target_gate.shape[1])
                            adapted_gate[:, :min_dim] = original_gate[:, :min_dim]
                        else:
                            adapted_gate = original_gate[:, :target_gate.shape[1]]
                        adapted_state_dict[key] = adapted_gate
                    elif 'input_proj.weight' in key:
                        original_weight = value
                        target_weight = model_state_dict[key]
                        if original_weight.shape[1] < target_weight.shape[1]:
                            adapted_weight = torch.zeros_like(target_weight)
                            min_dim = min(original_weight.shape[1], target_weight.shape[1])
                            adapted_weight[:, :min_dim] = original_weight[:, :min_dim]
                            if original_weight.shape[1] < target_weight.shape[1]:
                                mean_weights = original_weight.mean(dim=1, keepdim=True)
                                adapted_weight[:, original_weight.shape[1]:] = mean_weights.repeat(
                                    1, target_weight.shape[1] - original_weight.shape[1])
                        else:
                            adapted_weight = original_weight[:, :target_weight.shape[1]]
                        adapted_state_dict[key] = adapted_weight
                    elif 'input_proj.bias' in key:
                        adapted_state_dict[key] = value
                    else:
                        adapted_state_dict[key] = value
                else:
                    print(f"Skipping mismatched key: {key}")
            model.load_state_dict(adapted_state_dict, strict=False)
            print("Successfully loaded and adapted pretrained NIR model")
        else:
            model.load_state_dict(checkpoint)
            print("Successfully loaded pretrained NIR model")
    except Exception as e:
        print(f"Warning: Could not load pretrained NIR model: {e}")
        print("Training NIR from scratch")


def apply_student_freezing_scheme(model):
    """Freeze all params, then unfreeze stages.1, fusion_modules, decoder."""
    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if name.startswith('stages.1.') or name.startswith('fusion_modules.') or name.startswith('decoder.'):
            param.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Student model: {trainable:,}/{total:,} parameters trainable ({trainable/total*100:.1f}%)")
    return trainable


def apply_nir_freezing_scheme(model):
    """Freeze all params, then unfreeze input_proj, transformer_encoder.layers.2, decoder."""
    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if (name.startswith('input_proj.') or
            name.startswith('transformer_encoder.layers.2.') or
            name.startswith('decoder.')):
            param.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"NIR model: {trainable:,}/{total:,} parameters trainable ({trainable/total*100:.1f}%)")
    return trainable


def fine_tune_student_model(student_model, train_loader, criterion):
    """Fine-tune the student model with the new freezing scheme and early stopping."""
    print("\n=== Fine-tuning LED student model (new freezing scheme) ===")
    apply_student_freezing_scheme(student_model)
    trainable_params = [p for p in student_model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=FINE_TUNE_LEARNING_RATE, weight_decay=FINE_TUNE_WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
    best_train_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    best_model_state = None
    for epoch in range(1, FINE_TUNE_EPOCHS + 1):
        student_model.train()
        epoch_train_loss = 0.0
        for halo_rgb, led_rgb, vis_targets, _ in train_loader:
            led_rgb = led_rgb.to(DEVICE)
            vis_targets = vis_targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = student_model(led_rgb)
            loss = criterion(outputs, vis_targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_train_loss += loss.item()
        avg_train_loss = epoch_train_loss / len(train_loader)
        scheduler.step(avg_train_loss)
        if avg_train_loss < best_train_loss:
            best_train_loss = avg_train_loss
            best_epoch = epoch
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in student_model.state_dict().items()}
        else:
            patience_counter += 1
        if epoch % 10 == 0 or epoch == 1:
            lr = optimizer.param_groups[0]['lr']
            print(f"Fine-tune Epoch {epoch:03d}/{FINE_TUNE_EPOCHS} | Train Loss: {avg_train_loss:.4f} | LR: {lr:.6f}")
        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"Early stopping triggered at epoch {epoch}. Best train loss: {best_train_loss:.4f} at epoch {best_epoch}")
            break
    if best_model_state is not None:
        student_model.load_state_dict(best_model_state)
    torch.save(student_model.state_dict(), 'model/fine_tuned_student_model_new.pth')
    print("Best student model saved to fine_tuned_student_model_new.pth")
    return best_train_loss


def fine_tune_nir_model(nir_model, student_model, train_loader, criterion):
    """Fine-tune the NIR model with the new freezing scheme and early stopping."""
    print("\n=== Fine-tuning NIR prediction model (new freezing scheme) ===")
    apply_nir_freezing_scheme(nir_model)
    trainable_params = [p for p in nir_model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=FINE_TUNE_LEARNING_RATE, weight_decay=FINE_TUNE_WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
    best_train_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    best_model_state = None
    for epoch in range(1, FINE_TUNE_EPOCHS + 1):
        nir_model.train()
        student_model.eval()
        epoch_train_loss = 0.0
        for halo_rgb, led_rgb, vis_targets, nir_targets in train_loader:
            led_rgb = led_rgb.to(DEVICE)
            nir_targets = nir_targets.to(DEVICE)
            optimizer.zero_grad()
            with torch.no_grad():
                vis_reconstructed = student_model(led_rgb)
            outputs = nir_model(vis_reconstructed)
            loss = criterion(outputs, nir_targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(nir_model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_train_loss += loss.item()
        avg_train_loss = epoch_train_loss / len(train_loader)
        scheduler.step(avg_train_loss)
        if avg_train_loss < best_train_loss:
            best_train_loss = avg_train_loss
            best_epoch = epoch
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in nir_model.state_dict().items()}
        else:
            patience_counter += 1
        if epoch % 10 == 0 or epoch == 1:
            lr = optimizer.param_groups[0]['lr']
            print(f"NIR Fine-tune Epoch {epoch:03d}/{FINE_TUNE_EPOCHS} | Train Loss: {avg_train_loss:.4f} | LR: {lr:.6f}")
        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"Early stopping triggered at epoch {epoch}. Best NIR train loss: {best_train_loss:.4f} at epoch {best_epoch}")
            break
    if best_model_state is not None:
        nir_model.load_state_dict(best_model_state)
    torch.save(nir_model.state_dict(), 'model/fine_tuned_nir_model_new.pth')
    print("Best NIR model saved to fine_tuned_nir_model_new.pth")
    return best_train_loss


def calculate_full_spectrum_metrics(student_model, nir_model, X_led_test, y_vis_test, y_nir_test):
    """Calculate overall metrics for full spectrum, visible, and NIR on test set."""
    with torch.no_grad():
        test_led_rgb = torch.tensor(X_led_test, dtype=torch.float32).to(DEVICE)
        student_vis_pred = student_model(test_led_rgb).cpu().numpy()
        nir_pred = nir_model(torch.tensor(student_vis_pred, dtype=torch.float32).to(DEVICE)).cpu().numpy()
        full_spectrum_pred = np.concatenate([student_vis_pred, nir_pred], axis=1)
        full_spectrum_true = np.concatenate([y_vis_test, y_nir_test], axis=1)
    full_rmse = np.sqrt(np.mean((full_spectrum_pred - full_spectrum_true) ** 2))
    full_abs_error = np.abs(full_spectrum_pred - full_spectrum_true)
    full_relative_error = full_abs_error / (np.abs(full_spectrum_true) + 1e-8)
    full_mrae = np.mean(full_relative_error) * 100
    vis_rmse = np.sqrt(np.mean((student_vis_pred - y_vis_test) ** 2))
    vis_abs_error = np.abs(student_vis_pred - y_vis_test)
    vis_relative_error = vis_abs_error / (np.abs(y_vis_test) + 1e-8)
    vis_mrae = np.mean(vis_relative_error) * 100
    nir_rmse = np.sqrt(np.mean((nir_pred - y_nir_test) ** 2))
    nir_abs_error = np.abs(nir_pred - y_nir_test)
    nir_relative_error = nir_abs_error / (np.abs(y_nir_test) + 1e-8)
    nir_mrae = np.mean(nir_relative_error) * 100
    return {
        'full_spectrum': {'rmse': full_rmse, 'mrae': full_mrae},
        'visible': {'rmse': vis_rmse, 'mrae': vis_mrae},
        'nir': {'rmse': nir_rmse, 'mrae': nir_mrae}
    }


def main():
    """Main function: load pretrained models, fine-tune ONLY on training set, then evaluate on test set."""
    print("=== Loading fine-tuning data ===")
    df, spectrum_cols = load_and_prepare_fine_tune_data(NEW_DATA_FILE_PATHS)
    wavelengths = [float(col.replace('nm', '')) for col in spectrum_cols]
    (X_halo_train, X_led_train, y_vis_train, y_nir_train,
     X_halo_test, X_led_test, y_vis_test, y_nir_test), (train_idx, test_idx), (
        vis_end_idx, nir_start_idx) = prepare_fine_tune_datasets(df, spectrum_cols)
    vis_wavelengths = wavelengths[:vis_end_idx]
    nir_wavelengths = wavelengths[nir_start_idx:]
    print(f"\n=== Fine-tuning data info ===")
    print(f"Training samples: {len(X_halo_train)} (used for training)")
    print(f"Test samples: {len(X_halo_test)} (held out for final evaluation only)")
    print(f"Number of visible bands: {len(vis_wavelengths)}")
    print(f"Number of NIR bands: {len(nir_wavelengths)}")
    train_dataset = ECDADataset(X_halo_train, X_led_train, y_vis_train, y_nir_train)
    test_dataset = ECDADataset(X_halo_test, X_led_test, y_vis_test, y_nir_test)
    train_loader = DataLoader(train_dataset, batch_size=FINE_TUNE_BATCH_SIZE, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=FINE_TUNE_BATCH_SIZE, shuffle=False, pin_memory=True)
    print("\n=== Loading pretrained models ===")
    student_model = MultistageSpectralReconstructor(
        input_dim=len(LED_RGB_COLS), output_dim=len(vis_wavelengths),
        d_model=64, nhead=4, num_stages=2
    ).to(DEVICE)
    nir_model = SparseGatedNIRPredictor(
        input_dim=len(vis_wavelengths), output_dim=len(nir_wavelengths),
        d_model=64, nhead=4, num_layers=3
    ).to(DEVICE)
    print("=== Checking model files ===")
    print(f"Student model file exists: {os.path.exists('model/best_student_model.pth')}")
    print(f"NIR model file exists: {os.path.exists('model/best_nir_model.pth')}")
    if os.path.exists('model/best_student_model.pth'):
        load_pretrained_model_with_adaptation(student_model, 'model/best_student_model.pth', len(vis_wavelengths))
    else:
        print("Pretrained student model not found, training from scratch")
    if os.path.exists('model/best_nir_model.pth'):
        load_pretrained_nir_with_adaptation(nir_model, 'model/best_nir_model.pth', len(vis_wavelengths))
    else:
        print("Pretrained NIR model not found, training from scratch")
    criterion = MRAELoss()
    student_best_loss = fine_tune_student_model(student_model, train_loader, criterion)
    nir_best_loss = fine_tune_nir_model(nir_model, student_model, train_loader, criterion)
    print("\n=== Final evaluation on test set (never used during training) ===")
    student_model.load_state_dict(torch.load('model/fine_tuned_student_model_new.pth', map_location=DEVICE))
    nir_model.load_state_dict(torch.load('model/fine_tuned_nir_model_new.pth', map_location=DEVICE))
    student_val_loss, student_metrics = validate_vis(student_model, test_loader, criterion, 'led')
    nir_val_loss, nir_metrics = validate_nir(nir_model, test_loader, criterion, student_model)
    print("Student model (after fine-tuning):")
    print(f"  RMSE: {student_metrics['rmse']:.4f}, MRAE: {student_metrics['mrae']:.2f}%")
    print("NIR model (after fine-tuning):")
    print(f"  RMSE: {nir_metrics['rmse']:.4f}, MRAE: {nir_metrics['mrae']:.2f}%")
    print("\n=== Full spectrum overall metrics after fine-tuning ===")
    post_fine_tune_metrics = calculate_full_spectrum_metrics(
        student_model, nir_model, X_led_test, y_vis_test, y_nir_test)
    print(f"Full spectrum overall:")
    print(f"  RMSE: {post_fine_tune_metrics['full_spectrum']['rmse']:.4f}")
    print(f"  MRAE: {post_fine_tune_metrics['full_spectrum']['mrae']:.2f}%")


if __name__ == "__main__":
    main()