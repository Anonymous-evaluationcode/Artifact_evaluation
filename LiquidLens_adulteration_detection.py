import numpy as np
import os
import cv2
import torch
import torch.nn as nn
import torch.utils.data as data
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import joblib
import math
import glob
import random
import warnings

warnings.filterwarnings('ignore')


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
        return residual + self.dropout(x)


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
        return self.fusion_proj(fused_feat)


class MultistageSpectralReconstructor(nn.Module):
    def __init__(self, input_dim=3, output_dim=128, d_model=64, nhead=4, num_stages=2):
        super().__init__()
        self.d_model = d_model
        self.num_stages = num_stages
        self.input_proj = nn.Linear(input_dim, d_model)
        self.stages = nn.ModuleList([
            nn.Sequential(
                SpectralAttentionBlock(d_model, nhead),
                SpectralAttentionBlock(d_model, nhead),
                SpectralAttentionBlock(d_model, nhead)
            ) for _ in range(num_stages)
        ])
        self.fusion_modules = nn.ModuleList([
            SpectralFusionModule(d_model) for _ in range(num_stages)
        ])
        self.pos_encoder = PositionalEncoding(d_model)
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, output_dim)
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = x.unsqueeze(1)
        x = self.pos_encoder(x)

        stage_outputs = []
        for i in range(self.num_stages):
            x = self.stages[i](x)
            stage_outputs.append(x)

        fused_feat = x
        for i in range(self.num_stages - 1, -1, -1):
            fused_feat = self.fusion_modules[i](stage_outputs[i], fused_feat)

        return self.decoder(fused_feat.squeeze(1))


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


def normalize_pixel_brightness(pixels):
    max_vals = np.max(pixels, axis=1, keepdims=True)
    max_vals = np.where(max_vals == 0, 1, max_vals)
    return pixels.astype(np.float32) / max_vals


def load_tiles_from_directory(tiles_dir, normalize=True, use_all_pixels=True, max_pixels_per_tile=400):
    all_data = []
    all_labels = []
    all_file_paths = []
    all_file_ids = []

    class_dirs = [d for d in os.listdir(tiles_dir)
                  if os.path.isdir(os.path.join(tiles_dir, d))]

    if not class_dirs:
        # No subdirectories: treat all jpg files in the main folder as one class per filename prefix
        tile_files = glob.glob(os.path.join(tiles_dir, "*.jpg"))
        if tile_files:
            class_names = set()
            for file_path in tile_files:
                filename = os.path.basename(file_path)
                class_name = filename.split('_')[0]
                class_names.add(class_name)
            class_dirs = list(class_names)

    class_dirs = sorted(class_dirs)
    class_to_label = {name: idx for idx, name in enumerate(class_dirs)}

    print(f"Found {len(class_dirs)} classes: {class_dirs}")

    file_id_counter = 0

    for class_name in tqdm(class_dirs, desc="Loading class data"):
        class_label = class_to_label[class_name]

        class_dir = os.path.join(tiles_dir, class_name)
        if os.path.isdir(class_dir):
            tile_files = glob.glob(os.path.join(class_dir, "*.jpg"))
        else:
            tile_files = glob.glob(os.path.join(tiles_dir, f"{class_name}_*.jpg"))

        if not tile_files:
            print(f"Warning: no jpg files found for class {class_name}")
            continue

        print(f"  Class {class_name}: found {len(tile_files)} jpg files")

        for tile_file in tile_files:
            img = cv2.imread(tile_file)
            if img is None:
                print(f"Error: cannot read file {tile_file}")
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            height, width, channels = img.shape
            pixels = img.reshape(-1, channels)

            if normalize:
                pixels = normalize_pixel_brightness(pixels)

            n_pixels = pixels.shape[0]
            if use_all_pixels:
                selected_pixels = pixels
            else:
                n_samples = min(max_pixels_per_tile, n_pixels)
                if n_pixels > n_samples:
                    indices = np.random.choice(n_pixels, n_samples, replace=False)
                    selected_pixels = pixels[indices]
                else:
                    selected_pixels = pixels

            labels_arr = np.full(selected_pixels.shape[0], class_label)
            file_ids_arr = np.full(selected_pixels.shape[0], file_id_counter)

            all_data.append(selected_pixels)
            all_labels.append(labels_arr)
            all_file_paths.extend([tile_file] * selected_pixels.shape[0])
            all_file_ids.append(file_ids_arr)

            file_id_counter += 1

    if all_data:
        all_data = np.vstack(all_data)
        all_labels = np.concatenate(all_labels)
        all_file_ids = np.concatenate(all_file_ids)
        all_file_paths = np.array(all_file_paths)
    else:
        all_data = np.array([])
        all_labels = np.array([])
        all_file_ids = np.array([])
        all_file_paths = np.array([])

    return all_data, all_labels, all_file_paths, all_file_ids, class_dirs


def predict_spectral_features(rgb_data, vis_model, nir_model, device='cpu', batch_size=4096):
    dataset = data.TensorDataset(torch.tensor(rgb_data).float())
    dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    vis_features = []
    nir_features = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting spectral features"):
            inputs = batch[0].to(device)
            vis_pred = vis_model(inputs)
            vis_features.append(vis_pred.cpu().numpy())
            nir_pred = nir_model(vis_pred)
            nir_features.append(nir_pred.cpu().numpy())

    return np.vstack(vis_features), np.vstack(nir_features)


def train_rf(X_train, y_train, X_val, y_val, class_names, fold_idx=None):
    from sklearn.ensemble import RandomForestClassifier

    clf = RandomForestClassifier(
        n_estimators=500,
        max_depth=50,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        bootstrap=True,
        n_jobs=-1,
        random_state=42,
        verbose=0
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_val)
    acc = accuracy_score(y_val, y_pred)

    if fold_idx is not None:
        print(f"Fold {fold_idx}: validation accuracy = {acc:.4f}")

    return clf, y_pred, acc


def visualize_tile_classification(tiles, y_true, y_pred, class_names, class_colors,
                                  output_prefix="tile_classification"):
    n_tiles = len(tiles)
    if n_tiles == 0:
        print("Error: no tiles to visualize")
        return
    cols = min(4, n_tiles)
    rows = (n_tiles + cols - 1) // cols
    tile_size = tiles[0].shape[0]
    spacing = 10
    total_h = rows * tile_size + (rows + 1) * spacing
    total_w = cols * (tile_size * 2 + spacing * 3)
    gt_image = np.ones((total_h, total_w, 3), dtype=np.uint8) * 240
    pred_image = np.ones((total_h, total_w, 3), dtype=np.uint8) * 240

    for i, (tile, true_class, pred_class) in enumerate(zip(tiles, y_true, y_pred)):
        row = i // cols
        col = i % cols
        y_start = row * tile_size + (row + 1) * spacing
        y_end = y_start + tile_size
        x_start = col * (tile_size * 2 + spacing * 3) + spacing
        x_end = x_start + tile_size
        if y_end <= total_h and x_end <= total_w:
            gt_image[y_start:y_end, x_start:x_end] = tile
            pred_image[y_start:y_end, x_start:x_end] = tile
        x_color = x_end + spacing
        gt_image[y_start:y_end, x_color:x_color + tile_size] = class_colors[true_class]
        x_pred_color = x_color + tile_size + spacing
        pred_image[y_start:y_end, x_pred_color:x_pred_color + tile_size] = class_colors[pred_class]

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    axes[0].imshow(cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB))
    axes[0].set_title("True classification", fontsize=12)
    axes[0].axis('off')
    axes[1].imshow(cv2.cvtColor(pred_image, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Predicted classification", fontsize=12)
    axes[1].axis('off')

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=tuple(color / 255),
                             label=class_names[i]) for i, color in enumerate(class_colors)]
    fig.legend(handles=legend_elements, loc='lower center',
               bbox_to_anchor=(0.5, 0.02), ncol=min(6, len(class_names)),
               frameon=True, fontsize=9)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(f"{output_prefix}.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_spectral_distribution(X_spectral, y, class_names, class_colors, output_path="spectral_distribution.png"):
    plt.figure(figsize=(12, 8))
    visible_bands = 75
    nir_bands = 62
    total_bands = visible_bands + nir_bands
    class_spectra = []
    for class_idx in range(len(class_names)):
        class_mask = (y == class_idx)
        if np.any(class_mask):
            class_mean_spectrum = np.mean(X_spectral[class_mask], axis=0)
            class_spectra.append(class_mean_spectrum)
        else:
            class_spectra.append(np.zeros(total_bands))
    wavelengths = np.linspace(400, 1000, total_bands)
    for i, spectrum in enumerate(class_spectra):
        plt.plot(wavelengths, spectrum, label=class_names[i],
                 color=np.array(class_colors[i]) / 255, linewidth=2)
    plt.axvline(x=700, color='gray', linestyle='--', alpha=0.7)
    plt.text(500, np.max(X_spectral) * 0.95, 'Visible region', fontsize=12, ha='center')
    plt.text(850, np.max(X_spectral) * 0.95, 'NIR region', fontsize=12, ha='center')
    plt.title('Average spectral distribution by class', fontsize=16)
    plt.xlabel('Wavelength (nm)', fontsize=14)
    plt.ylabel('Reflectance', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Spectral distribution plot saved to: {output_path}")


def visualize_sample_tiles_per_class(tiles_dir, class_names, class_colors, output_path="sample_tiles_per_class.png"):
    n_classes = len(class_names)
    samples_per_class = 4
    fig, axes = plt.subplots(n_classes, samples_per_class, figsize=(samples_per_class * 3, n_classes * 3))
    if n_classes == 1:
        axes = axes.reshape(1, -1)
    for i, class_name in enumerate(class_names):
        class_dir = os.path.join(tiles_dir, class_name)
        if os.path.isdir(class_dir):
            tile_files = glob.glob(os.path.join(class_dir, "*.jpg"))
        else:
            tile_files = glob.glob(os.path.join(tiles_dir, f"{class_name}_*.jpg"))
        if not tile_files:
            continue
        n_samples = min(samples_per_class, len(tile_files))
        sample_files = random.sample(tile_files, n_samples)
        for j, tile_file in enumerate(sample_files):
            img = cv2.imread(tile_file)
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                axes[i, j].imshow(img_rgb)
                axes[i, j].axis('off')
                if j == 0:
                    axes[i, j].set_title(f"{class_name}", fontsize=12, color=np.array(class_colors[i]) / 255)
    plt.suptitle("Sample tiles per class (20x20)", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Sample tiles plot saved to: {output_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tiles_dir = 'dataset/LiquidLens_roi_of_flash_only'
    if not os.path.exists(tiles_dir):
        print(f"Error: tiles directory does not exist: {tiles_dir}")
        return

    print("\nLoading data from tiles directory...")
    X_rgb, y, file_paths, file_ids, class_names = load_tiles_from_directory(
        tiles_dir, normalize=True, use_all_pixels=True, max_pixels_per_tile=400
    )
    if X_rgb.size == 0:
        print("Error: no data loaded")
        return

    print(f"Loaded {X_rgb.shape[0]} pixels")

    colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))
    class_colors = [(colors[i][:3] * 255).astype(int) for i in range(len(class_names))]

    print("\nLoading spectral prediction models...")
    VISIBLE_BANDS = 75
    NIR_BANDS = 62

    vis_model = MultistageSpectralReconstructor(
        input_dim=3,
        output_dim=VISIBLE_BANDS,
        d_model=64,
        nhead=8
    ).to(device)
    vis_model.load_state_dict(torch.load('model/best_student_model.pth', map_location=device))
    vis_model.eval()

    nir_model = SparseGatedNIRPredictor(
        input_dim=VISIBLE_BANDS,
        output_dim=NIR_BANDS,
        d_model=64,
        nhead=8
    ).to(device)
    nir_model.load_state_dict(torch.load('model/best_nir_model.pth', map_location=device))
    nir_model.eval()

    print("\nPredicting spectral features...")
    vis_all, nir_all = predict_spectral_features(
        X_rgb, vis_model, nir_model, device=device, batch_size=16384
    )
    X_spectral = np.hstack((vis_all, nir_all))
    print(f"Spectral feature dimension: {X_spectral.shape}")

    print("\nPerforming 8-fold cross-validation (grouped by file)...")
    n_splits = 8
    gkf = GroupKFold(n_splits=n_splits)

    fold_accuracies = []
    all_y_true = []
    all_y_pred = []

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_spectral, y, groups=file_ids)):
        print(f"\n--- Fold {fold_idx+1}/{n_splits} ---")
        X_train, X_val = X_spectral[train_idx], X_spectral[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        _, y_pred, acc = train_rf(X_train, y_train, X_val, y_val, class_names, fold_idx+1)

        fold_accuracies.append(acc)
        all_y_true.append(y_val)
        all_y_pred.append(y_pred)

    all_y_true = np.concatenate(all_y_true)
    all_y_pred = np.concatenate(all_y_pred)

    print("\n" + "=" * 60)
    print("8-fold cross-validation results")
    print("=" * 60)
    for i, acc in enumerate(fold_accuracies):
        print(f"Fold {i+1}: {acc:.4f}")
    print(f"Average accuracy (8-fold CV): {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}")

    # Compute group-specific accuracies
    wine_class_indices = [i for i, name in enumerate(class_names) if name.startswith('wine_')]
    honey_class_indices = [i for i, name in enumerate(class_names) if name.startswith('honey_')]

    is_wine = np.isin(all_y_true, wine_class_indices)
    is_honey = np.isin(all_y_true, honey_class_indices)

    wine_acc = np.mean(all_y_pred[is_wine] == all_y_true[is_wine]) if is_wine.sum() > 0 else 0.0
    honey_acc = np.mean(all_y_pred[is_honey] == all_y_true[is_honey]) if is_honey.sum() > 0 else 0.0

    print(f"Wine group average accuracy: {wine_acc:.4f}")
    print(f"Honey group average accuracy: {honey_acc:.4f}")

    print("\nOverall classification report:")
    print(classification_report(all_y_true, all_y_pred, target_names=class_names))



    print("\n" + "=" * 60)
    print("Experiment summary")
    print("=" * 60)
    print(f"Total samples: {X_spectral.shape[0]}")
    print(f"Feature dimension: {X_spectral.shape[1]}")
    print(f"Number of classes: {len(class_names)}")
    print(f"Average accuracy (8-fold CV): {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}")
    print(f"Wine group average accuracy: {wine_acc:.4f}")
    print(f"Honey group average accuracy: {honey_acc:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()