"""
TrainSolubilityGP.py

Training a Gaussian Process on features extracted by a pre-trained GNN.
Output: continuous prediction [0, 1] + 95% confidence interval
"""

import os
import torch
import gpytorch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from torch_geometric.data import Batch
import matplotlib.pyplot as plt

from GnnClass2 import CombinedModel, FeatureExtractor, ExactGPLayer
from helpers import load_solubility_excel


# =============================================================================
# CONFIGURATION
# =============================================================================

device = "cpu"  # Use CPU to avoid MPS memory issues
print(f"Using device: {device}\n")

current_dir = os.getcwd()


# =============================================================================
# DATASET
# =============================================================================

class TripletDataset(Dataset):
    """Dataset for triplets (anion, ligand, solvent) with continuous labels."""
    
    def __init__(self, triplets, labels):
        self.triplets = triplets
        self.labels = labels

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        a, l, s = self.triplets[idx]
        # Convert {0, 1, 2} → {0.0, 0.5, 1.0}
        y = float(self.labels[idx]) / 2.0
        return a, l, s, y


def collate_triplets(batch):
    """Collate function to create PyG batches."""
    anions = [item[0] for item in batch]
    ligands = [item[1] for item in batch]
    solvents = [item[2] for item in batch]
    labels = torch.tensor([item[3] for item in batch], dtype=torch.float)
    
    return Batch.from_data_list(anions), Batch.from_data_list(ligands), Batch.from_data_list(solvents), labels


# =============================================================================
# DATA LOADING
# =============================================================================

print("=" * 60)
print("Loading data...")
print("=" * 60)

ligands_dict = {
    '2.1': os.path.join(current_dir, 'ligand/xyz/Ag_Pillarplex-Br.xyz'),
    '3.1': os.path.join(current_dir, 'ligand/xyz/Au_Pillarplex-Br.xyz')
}

# Charger Ag_Pillarplex
ag_triplets, ag_labels = [], []
try:
    ag_triplets, ag_labels = load_solubility_excel(
        excel_path=os.path.join(current_dir, 'solubility', 'Ag_Pillarplex.xlsx'),
        anions_dir='anions', ligands_dict=ligands_dict, solvents_dir='solvents',
        pillarplex_id='2.1', current_dir=current_dir
    )
    print(f"✓ Ag_Pillarplex: {len(ag_triplets)} samples")
except Exception as e:
    print(f"✗ Error Ag_Pillarplex: {e}")

# Charger Au_Pillarplex
au_triplets, au_labels = [], []
try:
    au_triplets, au_labels = load_solubility_excel(
        excel_path=os.path.join(current_dir, 'solubility', 'Au_Pillarplex.xlsx'),
        anions_dir='anions', ligands_dict=ligands_dict, solvents_dir='solvents',
        pillarplex_id='3.1', current_dir=current_dir
    )
    print(f"✓ Au_Pillarplex: {len(au_triplets)} samples")
except Exception as e:
    print(f"✗ Error Au_Pillarplex: {e}")

# Combine
all_triplets = ag_triplets + au_triplets
all_labels = ag_labels + au_labels
print(f"\nTotal: {len(all_triplets)} samples")

if len(all_triplets) == 0:
    exit("Error: no data loaded!")

# Split train/val
dataset = TripletDataset(all_triplets, all_labels)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size], 
                                           generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_triplets)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_triplets)

print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")


# =============================================================================
# STEP 1 : LOAD THE PRE-TRAINED GNN
# =============================================================================

print("\n" + "=" * 60)
print("Loading pre-trained GNN...")
print("=" * 60)

# Feature dimensions
nodefeat_num = all_triplets[0][0].x.shape[-1]
edgefeat_num = all_triplets[0][0].edge_attr.shape[-1]

# Create and load model
base_model = CombinedModel(
    nodefeat_num=nodefeat_num,
    edgefeat_num=edgefeat_num,
    nodeembed_to=64,
    edgeembed_to=32,
    num_classes=3
).to(device)

base_model.load_state_dict(torch.load('best_model.pth', map_location=device))
print("✓ Weights loaded from best_model.pth")

# Create FeatureExtractor (FROZEN)
feature_extractor = FeatureExtractor(base_model).to(device)
feature_extractor.eval()
for param in feature_extractor.parameters():
    param.requires_grad = False
print("✓ Feature Extractor frozen")


# =============================================================================
# STEP 2 : FEATURE EXTRACTION
# =============================================================================

print("\n" + "=" * 60)
print("Extracting features...")
print("=" * 60)

# Training features
train_features_list = []
train_labels_list = []

with torch.no_grad():
    for anions_b, ligands_b, solvents_b, labels_b in train_loader:
        anions_b = anions_b.to(device)
        ligands_b = ligands_b.to(device)
        solvents_b = solvents_b.to(device)
        
        features = feature_extractor(anions_b, ligands_b, solvents_b)
        train_features_list.append(features)
        train_labels_list.append(labels_b)

train_x = torch.cat(train_features_list, dim=0)
train_y = torch.cat(train_labels_list, dim=0)

print(f"✓ Train features: {train_x.shape}")

# Validation features
val_features_list = []
val_labels_list = []

with torch.no_grad():
    for anions_b, ligands_b, solvents_b, labels_b in val_loader:
        anions_b = anions_b.to(device)
        ligands_b = ligands_b.to(device)
        solvents_b = solvents_b.to(device)
        
        features = feature_extractor(anions_b, ligands_b, solvents_b)
        val_features_list.append(features)
        val_labels_list.append(labels_b)

val_x = torch.cat(val_features_list, dim=0)
val_y = torch.cat(val_labels_list, dim=0)

print(f"✓ Val features: {val_x.shape}")


# =============================================================================
# STEP 3 : GP CREATION
# =============================================================================

print("\n" + "=" * 60)
print("Creating Gaussian Process...")
print("=" * 60)

likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
gp_model = ExactGPLayer(train_x, train_y, likelihood).to(device)

print(f"✓ GP created with {train_x.shape[0]} training points")


# =============================================================================
# STEP 4 : GP TRAINING
# =============================================================================

print("\n" + "=" * 60)
print("Training GP...")
print("=" * 60)

# Optimizer
optimizer = torch.optim.Adam([
    {'params': gp_model.covar_module.parameters(), 'lr': 0.1},
    {'params': gp_model.mean_module.parameters(), 'lr': 0.1},
    {'params': likelihood.parameters(), 'lr': 0.1},
])

# Loss
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp_model)

# Training
epochs = 400
gp_model.train()
likelihood.train()

train_losses = []

for epoch in range(epochs):
    optimizer.zero_grad()
    output = gp_model(train_x)
    loss = -mll(output, train_y)
    loss.backward()
    optimizer.step()
    
    train_losses.append(loss.item())
    
    if epoch % 20 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch:03d}  Loss: {loss.item():.4f}")

print("✓ Training completed!")


# =============================================================================
# STEP 5 : EVALUATION
# =============================================================================

print("\n" + "=" * 60)
print("Evaluation...")
print("=" * 60)

gp_model.eval()
likelihood.eval()

with torch.no_grad(), gpytorch.settings.fast_pred_var():
    # Predictions on validation
    pred_dist = likelihood(gp_model(val_x))
    
    val_mean = pred_dist.mean.numpy()
    val_lower, val_upper = pred_dist.confidence_region()
    val_lower = val_lower.numpy()
    val_upper = val_upper.numpy()
    val_true = val_y.numpy()

val_mean = np.clip(val_mean, 0, 1)
val_lower = np.clip(val_lower, 0, 1)
val_upper = np.clip(val_upper, 0, 1)

# Metrics
mse = np.mean((val_mean - val_true) ** 2)
mae = np.mean(np.abs(val_mean - val_true))
coverage = np.mean((val_true >= val_lower) & (val_true <= val_upper))

print(f"\nMSE: {mse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"95% CI Coverage: {coverage:.1%}")


# =============================================================================
# STEP 6 : RESULTS DISPLAY
# =============================================================================

print("\n" + "=" * 60)
print("Prediction examples:")
print("=" * 60)

for i in range(min(10, len(val_mean))):
    pred = val_mean[i]
    lower = val_lower[i]
    upper = val_upper[i]
    true = val_true[i]
    
    # Interpretation
    if pred < 0.25:
        interp = "Insoluble"
    elif pred < 0.75:
        interp = "Partial"
    else:
        interp = "Soluble"
    
    print(f"  [{i+1}] True: {true:.2f} | Pred: {pred:.2f} [{lower:.2f}, {upper:.2f}] → {interp}")


# =============================================================================
# STEP 7 : VISUALISATION
# =============================================================================

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Plot 1: Loss
axes[0].plot(train_losses)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Negative Log Likelihood')
axes[0].set_title('Training Curve')
axes[0].grid(True, alpha=0.3)

# Plot 2: Predictions vs True
axes[1].errorbar(range(len(val_mean)), val_mean, 
                  yerr=[val_mean - val_lower, val_upper - val_mean],
                  fmt='o', capsize=3, alpha=0.7, label='Predictions ± 95% CI')
axes[1].scatter(range(len(val_true)), val_true, c='red', marker='x', 
                s=50, zorder=5, label='True values')
axes[1].set_xlabel('Sample')
axes[1].set_ylabel('Solubility')
axes[1].set_title('Predictions with Uncertainty')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Plot 3: Scatter predicted vs true
axes[2].scatter(val_true, val_mean, alpha=0.7)
axes[2].plot([0, 1], [0, 1], 'r--', label='Perfect')
axes[2].set_xlabel('True value')
axes[2].set_ylabel('Prediction')
axes[2].set_title(f'Prediction vs Reality (MSE={mse:.4f})')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('resultats_gp.png', dpi=150)
plt.show()

print("\n" + "=" * 60)
print("✓ Plots saved: resultats_gp.png")
print("=" * 60)


# =============================================================================
# ÉTAPE 8 : SAUVEGARDE DU MODÈLE
# =============================================================================

torch.save({
    'gp_model': gp_model.state_dict(),
    'likelihood': likelihood.state_dict(),
    'train_x': train_x,
    'train_y': train_y,
}, 'best_model_gp.pth')

print("✓ GP model saved: best_model_gp.pth")
