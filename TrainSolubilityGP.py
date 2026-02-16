"""
TrainSolubilityGP.py

Training a Gaussian Process on features extracted by a pre-trained GNN using K-Fold Cross-Validation.
Output: continuous prediction [0, 1] + 95% confidence interval for each fold
"""

import os
import torch
import gpytorch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

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


# =============================================================================
# K-FOLD SETUP
# =============================================================================

n_splits = 5
# Convert continuous labels back to discrete for stratification {0.0, 0.5, 1.0} -> {0, 1, 2}
discrete_labels = [int(label * 2) for label in all_labels]
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Store results for each fold
gp_results = {
    'fold': [],
    'mse': [],
    'mae': [],
    'coverage': [],
    'rmse': []
}

print(f"\nStarting {n_splits}-Fold Cross-Validation for GP Training...")
print(f"{'='*80}\n")

# Get feature dimensions (same for all folds)
first_triplet = all_triplets[0]
nodefeat_num = first_triplet[0].x.shape[-1]
edgefeat_num = first_triplet[0].edge_attr.shape[-1]

# ============================================================================
# K-FOLD GP TRAINING LOOP
# ============================================================================

for fold, (train_idx, val_idx) in enumerate(skf.split(all_triplets, discrete_labels)):
    print(f"{'='*80}")
    print(f"GP FOLD {fold+1}/{n_splits}")
    print(f"{'='*80}")
    
    # Create fold datasets
    fold_train_triplets = [all_triplets[i] for i in train_idx]
    fold_train_labels = [all_labels[i] for i in train_idx]
    fold_val_triplets = [all_triplets[i] for i in val_idx]
    fold_val_labels = [all_labels[i] for i in val_idx]
    
    fold_train_dataset = TripletDataset(fold_train_triplets, fold_train_labels)
    fold_val_dataset = TripletDataset(fold_val_triplets, fold_val_labels)
    
    fold_train_loader = DataLoader(fold_train_dataset, batch_size=4, shuffle=True, collate_fn=collate_triplets)
    fold_val_loader = DataLoader(fold_val_dataset, batch_size=4, shuffle=False, collate_fn=collate_triplets)
    
    print(f"Train: {len(fold_train_dataset)} | Val: {len(fold_val_dataset)}")
    
    # =============================================================================
    # LOAD PRE-TRAINED GNN
    # =============================================================================
    
    base_model = CombinedModel(
        nodefeat_num=nodefeat_num,
        edgefeat_num=edgefeat_num,
        nodeembed_to=64,
        edgeembed_to=32,
        num_classes=3
    ).to(device)
    
    # Load the best model from classification training
    # Note: Using the first fold's best model (best_model_fold0.pth)
    # For production, should average or fine-tune with fold-specific models
    try:
        base_model.load_state_dict(torch.load('best_model_fold0.pth', map_location=device))
        print("✓ Loaded pre-trained GNN from best_model_fold0.pth")
    except FileNotFoundError:
        print("✗ Warning: best_model_fold0.pth not found. Using last available model.")
        # Try to find an available model
        for i in range(n_splits):
            try:
                base_model.load_state_dict(torch.load(f'best_model_fold{i}.pth', map_location=device))
                print(f"✓ Loaded pre-trained GNN from best_model_fold{i}.pth")
                break
            except FileNotFoundError:
                continue
    
    # Create feature extractor (frozen)
    feature_extractor = FeatureExtractor(base_model).to(device)
    feature_extractor.eval()
    for param in feature_extractor.parameters():
        param.requires_grad = False
    
    # =============================================================================
    # EXTRACT FEATURES FOR THIS FOLD
    # =============================================================================
    
    print(f"\nExtracting features...")
    
    fold_train_features = []
    fold_train_y = []
    
    with torch.no_grad():
        for anions_b, ligands_b, solvents_b, labels_b in fold_train_loader:
            anions_b = anions_b.to(device)
            ligands_b = ligands_b.to(device)
            solvents_b = solvents_b.to(device)
            features = feature_extractor(anions_b, ligands_b, solvents_b)
            fold_train_features.append(features)
            fold_train_y.append(labels_b)
    
    fold_train_x = torch.cat(fold_train_features, dim=0)
    fold_train_y = torch.cat(fold_train_y, dim=0)
    
    print(f"✓ Train features: {fold_train_x.shape}")
    
    fold_val_features = []
    fold_val_y = []
    
    with torch.no_grad():
        for anions_b, ligands_b, solvents_b, labels_b in fold_val_loader:
            anions_b = anions_b.to(device)
            ligands_b = ligands_b.to(device)
            solvents_b = solvents_b.to(device)
            features = feature_extractor(anions_b, ligands_b, solvents_b)
            fold_val_features.append(features)
            fold_val_y.append(labels_b)
    
    fold_val_x = torch.cat(fold_val_features, dim=0)
    fold_val_y = torch.cat(fold_val_y, dim=0)
    
    print(f"✓ Val features: {fold_val_x.shape}")
    
    # =============================================================================
    # CREATE AND TRAIN GP FOR THIS FOLD
    # =============================================================================
    
    print(f"\nTraining GP for fold {fold+1}...")
    
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    gp_model = ExactGPLayer(fold_train_x, fold_train_y, likelihood).to(device)
    
    optimizer = torch.optim.Adam([
        {'params': gp_model.covar_module.parameters(), 'lr': 0.1},
        {'params': gp_model.mean_module.parameters(), 'lr': 0.1},
        {'params': likelihood.parameters(), 'lr': 0.1},
    ])
    
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp_model)
    
    epochs = 400
    gp_model.train()
    likelihood.train()
    
    train_losses = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = gp_model(fold_train_x)
        loss = -mll(output, fold_train_y)
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        
        if epoch % 100 == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch:03d}  Loss: {loss.item():.4f}")
    
    print("✓ GP Training completed!")
    
    # =============================================================================
    # EVALUATE GP ON VALIDATION SET
    # =============================================================================
    
    print(f"\nEvaluating fold {fold+1}...")
    
    gp_model.eval()
    likelihood.eval()
    
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred_dist = likelihood(gp_model(fold_val_x))
        
        fold_val_mean = pred_dist.mean.numpy()
        fold_val_lower, fold_val_upper = pred_dist.confidence_region()
        fold_val_lower = fold_val_lower.numpy()
        fold_val_upper = fold_val_upper.numpy()
        fold_val_true = fold_val_y.numpy()
    
    fold_val_mean = np.clip(fold_val_mean, 0, 1)
    fold_val_lower = np.clip(fold_val_lower, 0, 1)
    fold_val_upper = np.clip(fold_val_upper, 0, 1)
    
    # Calculate metrics for this fold
    mse = np.mean((fold_val_mean - fold_val_true) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(fold_val_mean - fold_val_true))
    coverage = np.mean((fold_val_true >= fold_val_lower) & (fold_val_true <= fold_val_upper))
    
    gp_results['fold'].append(fold+1)
    gp_results['mse'].append(mse)
    gp_results['rmse'].append(rmse)
    gp_results['mae'].append(mae)
    gp_results['coverage'].append(coverage)
    
    print(f"  MSE:      {mse:.4f}")
    print(f"  RMSE:     {rmse:.4f}")
    print(f"  MAE:      {mae:.4f}")
    print(f"  Coverage: {coverage:.1%}")
    
    # Save GP model for this fold
    torch.save({
        'gp_model': gp_model.state_dict(),
        'likelihood': likelihood.state_dict(),
        'train_x': fold_train_x,
        'train_y': fold_train_y,
    }, f'best_model_gp_fold{fold}.pth')
    
    print(f"✓ Saved fold {fold+1} GP model to best_model_gp_fold{fold}.pth")
    print(f"{'='*80}\n")

# =============================================================================
# CROSS-VALIDATION SUMMARY
# =============================================================================

print(f"\n{'='*80}")
print("GP CROSS-VALIDATION SUMMARY")
print(f"{'='*80}")

mses = np.array(gp_results['mse'])
rmses = np.array(gp_results['rmse'])
maes = np.array(gp_results['mae'])
coverages = np.array(gp_results['coverage'])

print(f"\nMSE:      {np.mean(mses):.4f} ± {np.std(mses):.4f}")
print(f"  Range: {np.min(mses):.4f} - {np.max(mses):.4f}")
print(f"\nRMSE:     {np.mean(rmses):.4f} ± {np.std(rmses):.4f}")
print(f"  Range: {np.min(rmses):.4f} - {np.max(rmses):.4f}")
print(f"\nMAE:      {np.mean(maes):.4f} ± {np.std(maes):.4f}")
print(f"  Range: {np.min(maes):.4f} - {np.max(maes):.4f}")
print(f"\n95% CI Coverage: {np.mean(coverages):.1%} ± {np.std(coverages):.1%}")
print(f"  Range: {np.min(coverages):.1%} - {np.max(coverages):.1%}")

print(f"\nPer-Fold Results:")
print(f"{'Fold':<6} {'MSE':<10} {'RMSE':<10} {'MAE':<10} {'Coverage':<12}")
print(f"{'-'*55}")
for i, fold_num in enumerate(gp_results['fold']):
    print(f"{fold_num:<6} {gp_results['mse'][i]:<10.4f} {gp_results['rmse'][i]:<10.4f} "
          f"{gp_results['mae'][i]:<10.4f} {gp_results['coverage'][i]:<12.1%}")

# =============================================================================
# AGGREGATED PREDICTIONS ACROSS ALL FOLDS
# =============================================================================

print(f"\n{'='*80}")
print("AGGREGATED PREDICTIONS (All Folds)")
print(f"{'='*80}\n")

all_predictions = []
all_true_values = []

for fold, (train_idx, val_idx) in enumerate(skf.split(all_triplets, discrete_labels)):
    fold_val_triplets = [all_triplets[i] for i in val_idx]
    fold_val_labels = [all_labels[i] for i in val_idx]
    fold_val_dataset = TripletDataset(fold_val_triplets, fold_val_labels)
    fold_val_loader = DataLoader(fold_val_dataset, batch_size=4, shuffle=False, collate_fn=collate_triplets)
    
    # Load GP model and likelihood for this fold
    checkpoint = torch.load(f'best_model_gp_fold{fold}.pth', map_location=device)
    
    gp_model = ExactGPLayer(checkpoint['train_x'], checkpoint['train_y'], 
                           gpytorch.likelihoods.GaussianLikelihood()).to(device)
    gp_model.load_state_dict(checkpoint['gp_model'])
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    likelihood.load_state_dict(checkpoint['likelihood'])
    
    # Extract features for validation
    base_model = CombinedModel(
        nodefeat_num=nodefeat_num,
        edgefeat_num=edgefeat_num,
        nodeembed_to=64,
        edgeembed_to=32,
        num_classes=3
    ).to(device)
    
    try:
        base_model.load_state_dict(torch.load('best_model_fold0.pth', map_location=device))
    except:
        for i in range(n_splits):
            try:
                base_model.load_state_dict(torch.load(f'best_model_fold{i}.pth', map_location=device))
                break
            except:
                continue
    
    feature_extractor = FeatureExtractor(base_model).to(device)
    feature_extractor.eval()
    
    # Get predictions
    gp_model.eval()
    likelihood.eval()
    
    fold_preds = []
    fold_trues = []
    
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        for anions_b, ligands_b, solvents_b, labels_b in fold_val_loader:
            anions_b = anions_b.to(device)
            ligands_b = ligands_b.to(device)
            solvents_b = solvents_b.to(device)
            
            features = feature_extractor(anions_b, ligands_b, solvents_b)
            pred_dist = likelihood(gp_model(features))
            
            fold_preds.extend(pred_dist.mean.cpu().numpy())
            fold_trues.extend(labels_b.numpy())
    
    all_predictions.extend(fold_preds)
    all_true_values.extend(fold_trues)

# Plot aggregated predictions
all_predictions = np.array(all_predictions)
all_true_values = np.array(all_true_values)

plt.figure(figsize=(8, 6))
plt.scatter(all_true_values, all_predictions, alpha=0.7, s=50)
plt.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Prediction')
plt.xlabel('True Solubility', fontsize=12)
plt.ylabel('Predicted Solubility', fontsize=12)
plt.title(f'GP Predictions vs True Values (All Folds, Overall MAE={np.mean(np.abs(all_predictions - all_true_values)):.4f})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.tight_layout()
plt.savefig('cv_gp_predictions_all_folds.png', dpi=150)
plt.show()

print("✓ Saved plot: cv_gp_predictions_all_folds.png")
print(f"{'='*80}")
print("✓ K-Fold Cross-Validation for GP Training Completed!")
print(f"Model weights saved as: best_model_gp_fold0.pth to best_model_gp_fold{n_splits-1}.pth")
print(f"{'='*80}")
