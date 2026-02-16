"""
Training script using real solubility data from Excel files with K-Fold Cross-Validation.

Loads data from solubility/Ag_Pillarplex.xlsx and solubility/Au_Pillarplex.xlsx
and trains the CombinedModel with proper labels (yes/slightly/no) using StratifiedKFold.
"""

import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import StratifiedKFold

# Ajout pour les plots seaborn
import seaborn as sns
import matplotlib.pyplot as plt

from GnnClass2 import CombinedModel
from helpers import load_solubility_excel
from read_xyz import extract_last_snapshot

# Device
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using device: {device}\n")

current_dir = os.getcwd()

# ============================================================================
# DATASET AND DATALOADER SETUP
# ============================================================================

class TripletDataset(Dataset):
    """Dataset for triplet graphs (anion, ligand, solvent) with labels."""
    
    def __init__(self, triplets, labels):
        assert len(triplets) == len(labels), "Triplets and labels must have same length"
        self.triplets = triplets
        self.labels = labels

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        a, l, s = self.triplets[idx]
        y = int(self.labels[idx])
        return a, l, s, y


def collate_triplets(batch):
    """
    Collate function for TripletDataset.
    Converts list of (anion, ligand, solvent, label) into batched graphs.
    
    Args:
        batch: list of (anion_graph, ligand_graph, solvent_graph, label)
        
    Returns:
        anions_batch, ligands_batch, solvents_batch, labels_tensor
    """
    anions = [item[0] for item in batch]
    ligands = [item[1] for item in batch]
    solvents = [item[2] for item in batch]
    labels = torch.tensor([item[3] for item in batch], dtype=torch.long)
    
    anions_batch = Batch.from_data_list(anions)
    ligands_batch = Batch.from_data_list(ligands)
    solvents_batch = Batch.from_data_list(solvents)
    
    return anions_batch, ligands_batch, solvents_batch, labels


# ============================================================================
# LOAD DATA FROM EXCEL FILES
# ============================================================================

print("="*80)
print("Loading solubility data from Excel files...")
print("="*80)

# Define ligand paths (full paths)
ligands_dict = {
    '2.1': os.path.join(current_dir, 'ligand/xyz/Ag_Pillarplex-Br.xyz'),
    '3.1': os.path.join(current_dir, 'ligand/xyz/Au_Pillarplex-Br.xyz')
}

anions_dir = 'anions'
solvents_dir = 'solvents'

# Load Ag_Pillarplex data
print("\nLoading Ag_Pillarplex (2.1)...")
ag_excel_path = os.path.join(current_dir, 'solubility', 'Ag_Pillarplex.xlsx')
try:
    ag_triplets, ag_labels = load_solubility_excel(
        excel_path=ag_excel_path,
        anions_dir=anions_dir,
        ligands_dict=ligands_dict,
        solvents_dir=solvents_dir,
        pillarplex_id='2.1',
        current_dir=current_dir
    )
    print(f"✓ Loaded {len(ag_triplets)} samples from Ag_Pillarplex")
except Exception as e:
    print(f"✗ Error loading Ag_Pillarplex: {e}")
    ag_triplets, ag_labels = [], []

# Load Au_Pillarplex data
print("\nLoading Au_Pillarplex (3.1)...")
au_excel_path = os.path.join(current_dir, 'solubility', 'Au_Pillarplex.xlsx')
try:
    au_triplets, au_labels = load_solubility_excel(
        excel_path=au_excel_path,
        anions_dir=anions_dir,
        ligands_dict=ligands_dict,
        solvents_dir=solvents_dir,
        pillarplex_id='3.1',
        current_dir=current_dir
    )
    print(f"✓ Loaded {len(au_triplets)} samples from Au_Pillarplex")
except Exception as e:
    print(f"✗ Error loading Au_Pillarplex: {e}")
    au_triplets, au_labels = [], []

# Combine datasets
all_triplets = ag_triplets + au_triplets
all_labels = ag_labels + au_labels

print(f"\n{'='*80}")
print(f"Total dataset: {len(all_triplets)} samples")
print(f"Label distribution: no={sum(1 for l in all_labels if l==0)}, "
      f"slightly={sum(1 for l in all_labels if l==1)}, "
      f"yes={sum(1 for l in all_labels if l==2)}")
print(f"{'='*80}\n")

if len(all_triplets) == 0:
    print("ERROR: No data loaded. Please check:")
    print(f"  - Excel files exist in {current_dir}/solubility/")
    print(f"  - Anion files exist in {current_dir}/{anions_dir}/")
    print(f"  - Solvent files exist in {current_dir}/{solvents_dir}/")
    print(f"  - Ligand files exist in {current_dir}/mof_solubility/ligand/xyz/")
    exit(1)

# Create dataset
dataset = TripletDataset(all_triplets, all_labels)

# Setup k-fold cross-validation
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Store results for each fold
fold_results = {
    'fold': [],
    'train_acc': [],
    'val_acc': [],
    'train_loss': [],
    'val_loss': [],
    'val_confusion_matrices': [],
    'train_confusion_matrices': [],
    'train_losses_per_epoch': [],  # New: list of lists for each fold
    'val_losses_per_epoch': [],    # New
    'train_accs_per_epoch': [],    # New
    'val_accs_per_epoch': []       # New
}

print(f"Starting {n_splits}-Fold Stratified Cross-Validation...")
print(f"{'='*80}\n")

# Get feature dimensions from first sample (same for all folds)
first_triplet = all_triplets[0]
nodefeat_num = first_triplet[0].x.shape[-1]
edgefeat_num = first_triplet[0].edge_attr.shape[-1]

print(f"Node features: {nodefeat_num}")
print(f"Edge features: {edgefeat_num}\n")

# ============================================================================
# K-FOLD TRAINING LOOP
# ============================================================================

for fold, (train_idx, val_idx) in enumerate(skf.split(all_triplets, all_labels)):
    print(f"{'='*80}")
    print(f"FOLD {fold+1}/{n_splits}")
    print(f"{'='*80}")
    
    # Create fold datasets
    fold_train_triplets = [all_triplets[i] for i in train_idx]
    fold_train_labels = [all_labels[i] for i in train_idx]
    fold_val_triplets = [all_triplets[i] for i in val_idx]
    fold_val_labels = [all_labels[i] for i in val_idx]
    
    # Create TripletDataset for this fold
    train_dataset = TripletDataset(fold_train_triplets, fold_train_labels)
    val_dataset = TripletDataset(fold_val_triplets, fold_val_labels)
    
    # Create DataLoaders
    batch_size = 4
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_triplets)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_triplets)
    
    print(f"Train set: {len(train_dataset)} samples")
    print(f"Validation set: {len(val_dataset)} samples")
    print(f"Batch size: {batch_size}\n")
    
    # ============================================================================
    # MODEL SETUP FOR THIS FOLD
    # ============================================================================
    
    model = CombinedModel(
        nodefeat_num=nodefeat_num,
        edgefeat_num=edgefeat_num,
        nodeembed_to=64,
        edgeembed_to=32,
        num_classes=3
    )
    model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params}")
    
    # ============================================================================
    # TRAINING SETUP FOR THIS FOLD
    # ============================================================================
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    epochs = 100
    
    # ============================================================================
    # TRAINING LOOP FOR THIS FOLD
    # ============================================================================
    
    best_val_acc = 0
    patience = 45
    patience_counter = 0
    
    print(f"\nTraining fold {fold+1}...")
    
    # Initialize per-epoch tracking
    fold_train_losses = []
    fold_val_losses = []
    fold_train_accs = []
    fold_val_accs = []
    
    for epoch in range(epochs):
        # ========== TRAINING ==========
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (anions_b, ligands_b, solvents_b, labels_b) in enumerate(train_loader):
            anions_b = anions_b.to(device)
            ligands_b = ligands_b.to(device)
            solvents_b = solvents_b.to(device)
            labels_b = labels_b.to(device)

            optimizer.zero_grad()
            logits = model(anions_b, ligands_b, solvents_b)
            loss = criterion(logits, labels_b)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * labels_b.size(0)
            preds = logits.argmax(dim=1)
            train_correct += (preds == labels_b).sum().item()
            train_total += labels_b.size(0)

        avg_train_loss = train_loss / len(train_dataset)
        train_acc = train_correct / train_total
        
        # Store per-epoch data
        fold_train_losses.append(avg_train_loss)
        fold_train_accs.append(train_acc)

        # ========== VALIDATION ==========
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for anions_b, ligands_b, solvents_b, labels_b in val_loader:
                anions_b = anions_b.to(device)
                ligands_b = ligands_b.to(device)
                solvents_b = solvents_b.to(device)
                labels_b = labels_b.to(device)
                
                logits = model(anions_b, ligands_b, solvents_b)
                loss = criterion(logits, labels_b)
                val_loss += loss.item() * labels_b.size(0)
                
                preds = logits.argmax(dim=1)
                val_correct += (preds == labels_b).sum().item()
                val_total += labels_b.size(0)
        
        avg_val_loss = val_loss / len(val_dataset)
        val_acc = val_correct / val_total
        
        # Store per-epoch data
        fold_val_losses.append(avg_val_loss)
        fold_val_accs.append(val_acc)
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Save best model for this fold
            torch.save(model.state_dict(), f'best_model_fold{fold}.pth')
        else:
            patience_counter += 1
        
        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch:03d}  train_loss={avg_train_loss:.4f}  train_acc={train_acc:.3f}  "
                  f"val_loss={avg_val_loss:.4f}  val_acc={val_acc:.3f}")
        
        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch} (patience={patience})")
            break
    
    # ============================================================================
    # FOLD EVALUATION
    # ============================================================================
    
    print(f"\nEvaluating fold {fold+1}...")
    
    # Load best model for this fold
    model.load_state_dict(torch.load(f'best_model_fold{fold}.pth', map_location=device))
    model.eval()
    
    label_names = {0: 'no', 1: 'slightly', 2: 'yes'}
    
    # Validation evaluation
    val_preds = []
    val_labels_list = []
    
    with torch.no_grad():
        for anions_b, ligands_b, solvents_b, labels_b in val_loader:
            anions_b = anions_b.to(device)
            ligands_b = ligands_b.to(device)
            solvents_b = solvents_b.to(device)
            labels_b = labels_b.to(device)
            
            logits = model(anions_b, ligands_b, solvents_b)
            preds = logits.argmax(dim=1)
            
            val_preds.extend(preds.cpu().numpy())
            val_labels_list.extend(labels_b.cpu().numpy())
    
    val_acc_final = np.mean(np.array(val_preds) == np.array(val_labels_list))
    val_cm = confusion_matrix(val_labels_list, val_preds, labels=[0, 1, 2])
    
    # Training evaluation
    train_preds = []
    train_labels_list = []
    
    with torch.no_grad():
        for anions_b, ligands_b, solvents_b, labels_b in train_loader:
            anions_b = anions_b.to(device)
            ligands_b = ligands_b.to(device)
            solvents_b = solvents_b.to(device)
            labels_b = labels_b.to(device)
            
            logits = model(anions_b, ligands_b, solvents_b)
            preds = logits.argmax(dim=1)
            
            train_preds.extend(preds.cpu().numpy())
            train_labels_list.extend(labels_b.cpu().numpy())
    
    train_acc_final = np.mean(np.array(train_preds) == np.array(train_labels_list))
    train_cm = confusion_matrix(train_labels_list, train_preds, labels=[0, 1, 2])
    
    # Store fold results
    fold_results['fold'].append(fold+1)
    fold_results['train_acc'].append(train_acc_final)
    fold_results['val_acc'].append(val_acc_final)
    fold_results['train_loss'].append(avg_train_loss)
    fold_results['val_loss'].append(avg_val_loss)
    fold_results['val_confusion_matrices'].append(val_cm)
    fold_results['train_confusion_matrices'].append(train_cm)
    fold_results['train_losses_per_epoch'].append(fold_train_losses)
    fold_results['val_losses_per_epoch'].append(fold_val_losses)
    fold_results['train_accs_per_epoch'].append(fold_train_accs)
    fold_results['val_accs_per_epoch'].append(fold_val_accs)
    
    print(f"  Train Accuracy: {train_acc_final:.3f}")
    print(f"  Val Accuracy:   {val_acc_final:.3f}")
    print(f"{'='*80}\n")

# ============================================================================
# CROSS-VALIDATION SUMMARY
# ============================================================================

print(f"\n{'='*80}")
print("CROSS-VALIDATION SUMMARY")
print(f"{'='*80}")

train_accs = np.array(fold_results['train_acc'])
val_accs = np.array(fold_results['val_acc'])
train_losses = np.array(fold_results['train_loss'])
val_losses = np.array(fold_results['val_loss'])

print(f"\nTrain Accuracy: {np.mean(train_accs):.3f} ± {np.std(train_accs):.3f}")
print(f"  Range: {np.min(train_accs):.3f} - {np.max(train_accs):.3f}")
print(f"\nVal Accuracy:   {np.mean(val_accs):.3f} ± {np.std(val_accs):.3f}")
print(f"  Range: {np.min(val_accs):.3f} - {np.max(val_accs):.3f}")
print(f"\nTrain Loss:     {np.mean(train_losses):.4f} ± {np.std(train_losses):.4f}")
print(f"Val Loss:       {np.mean(val_losses):.4f} ± {np.std(val_losses):.4f}")

print(f"\nPer-Fold Results:")
print(f"{'Fold':<6} {'Train Acc':<12} {'Val Acc':<12}")
print(f"{'-'*30}")
for i, fold_num in enumerate(fold_results['fold']):
    print(f"{fold_num:<6} {fold_results['train_acc'][i]:<12.3f} {fold_results['val_acc'][i]:<12.3f}")

# Aggregate confusion matrices
print(f"\n{'='*80}")
print("AGGREGATED CONFUSION MATRICES ACROSS ALL FOLDS")
print(f"{'='*80}")

all_val_preds = []
all_val_labels = []
all_train_preds = []
all_train_labels = []

for fold, (train_idx, val_idx) in enumerate(skf.split(all_triplets, all_labels)):
    fold_val_triplets = [all_triplets[i] for i in val_idx]
    fold_val_labels = [all_labels[i] for i in val_idx]
    fold_train_triplets = [all_triplets[i] for i in train_idx]
    fold_train_labels = [all_labels[i] for i in train_idx]
    
    val_dataset = TripletDataset(fold_val_triplets, fold_val_labels)
    train_dataset = TripletDataset(fold_train_triplets, fold_train_labels)
    
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_triplets)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False, collate_fn=collate_triplets)
    
    model.load_state_dict(torch.load(f'best_model_fold{fold}.pth', map_location=device))
    model.eval()
    
    # Val predictions
    with torch.no_grad():
        for anions_b, ligands_b, solvents_b, labels_b in val_loader:
            anions_b = anions_b.to(device)
            ligands_b = ligands_b.to(device)
            solvents_b = solvents_b.to(device)
            logits = model(anions_b, ligands_b, solvents_b)
            preds = logits.argmax(dim=1)
            all_val_preds.extend(preds.cpu().numpy())
            all_val_labels.extend(labels_b.numpy())
    
    # Train predictions
    with torch.no_grad():
        for anions_b, ligands_b, solvents_b, labels_b in train_loader:
            anions_b = anions_b.to(device)
            ligands_b = ligands_b.to(device)
            solvents_b = solvents_b.to(device)
            logits = model(anions_b, ligands_b, solvents_b)
            preds = logits.argmax(dim=1)
            all_train_preds.extend(preds.cpu().numpy())
            all_train_labels.extend(labels_b.numpy())

# Aggregated validation confusion matrix
agg_val_cm = confusion_matrix(all_val_labels, all_val_preds, labels=[0, 1, 2])
print("\nAggregated Validation Confusion Matrix:")
print(f"{'':15} Pred_no  Pred_slightly  Pred_yes")
for i, class_name in enumerate(['no', 'slightly', 'yes']):
    print(f"True_{class_name:8s}:  {agg_val_cm[i, 0]:5d}    {agg_val_cm[i, 1]:5d}        {agg_val_cm[i, 2]:5d}")

# Plot aggregated validation confusion matrix
plt.figure(figsize=(6,5))
sns.heatmap(agg_val_cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['no', 'slightly', 'yes'], 
            yticklabels=['no', 'slightly', 'yes'])
plt.title('Aggregated Confusion Matrix (Validation Set - All Folds)')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.tight_layout()
plt.savefig('cv_confusion_matrix_validation.png', dpi=150)
plt.show()

# Aggregated training confusion matrix
agg_train_cm = confusion_matrix(all_train_labels, all_train_preds, labels=[0, 1, 2])
print("\nAggregated Training Confusion Matrix:")
print(f"{'':15} Pred_no  Pred_slightly  Pred_yes")
for i, class_name in enumerate(['no', 'slightly', 'yes']):
    print(f"True_{class_name:8s}:  {agg_train_cm[i, 0]:5d}    {agg_train_cm[i, 1]:5d}        {agg_train_cm[i, 2]:5d}")

# Plot aggregated training confusion matrix
plt.figure(figsize=(6,5))
sns.heatmap(agg_train_cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=['no', 'slightly', 'yes'],
            yticklabels=['no', 'slightly', 'yes'])
plt.title('Aggregated Confusion Matrix (Training Set - All Folds)')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.tight_layout()
plt.savefig('cv_confusion_matrix_training.png', dpi=150)
plt.show()

# ============================================================================
# ADDITIONAL PLOTS FOR PRESENTATION
# ============================================================================

# 1. Training Loss and Accuracy Curves (averaged across folds)
plt.figure(figsize=(12, 5))

# Loss plot
plt.subplot(1, 2, 1)
for fold in range(n_splits):
    epochs_range = range(1, len(fold_results['train_losses_per_epoch'][fold]) + 1)
    plt.plot(epochs_range, fold_results['train_losses_per_epoch'][fold], label=f'Fold {fold+1} Train', alpha=0.7)
    plt.plot(epochs_range, fold_results['val_losses_per_epoch'][fold], label=f'Fold {fold+1} Val', linestyle='--', alpha=0.7)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss per Fold')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)

# Accuracy plot
plt.subplot(1, 2, 2)
for fold in range(n_splits):
    epochs_range = range(1, len(fold_results['train_accs_per_epoch'][fold]) + 1)
    plt.plot(epochs_range, fold_results['train_accs_per_epoch'][fold], label=f'Fold {fold+1} Train', alpha=0.7)
    plt.plot(epochs_range, fold_results['val_accs_per_epoch'][fold], label=f'Fold {fold+1} Val', linestyle='--', alpha=0.7)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy per Fold')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)

plt.tight_layout()
plt.savefig('training_curves_per_fold.png', dpi=150, bbox_inches='tight')
plt.show()

# 2. Per-Fold Accuracies Bar Plot
plt.figure(figsize=(8, 5))
folds = fold_results['fold']
train_accs = fold_results['train_acc']
val_accs = fold_results['val_acc']
x = np.arange(len(folds))
width = 0.35
plt.bar(x - width/2, train_accs, width, label='Train Accuracy', alpha=0.8)
plt.bar(x + width/2, val_accs, width, label='Val Accuracy', alpha=0.8)
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.title('Accuracy per Fold')
plt.xticks(x, folds)
plt.legend()
plt.grid(True, axis='y')
plt.tight_layout()
plt.savefig('per_fold_accuracies.png', dpi=150)
plt.show()

# 3. ROC Curves for Multi-Class (One-vs-Rest)
# Convert labels to binary for each class
from sklearn.preprocessing import label_binarize
classes = [0, 1, 2]
class_names = ['no', 'slightly', 'yes']
y_test_bin = label_binarize(all_val_labels, classes=classes)
y_score = []  # Need probabilities

# Get probabilities for validation set
model.load_state_dict(torch.load('best_model_fold0.pth', map_location=device))  # Use first fold model for simplicity
model.eval()
val_probs = []
with torch.no_grad():
    for anions_b, ligands_b, solvents_b, labels_b in DataLoader(TripletDataset(all_triplets, all_labels), batch_size=4, shuffle=False, collate_fn=collate_triplets):
        anions_b = anions_b.to(device)
        ligands_b = ligands_b.to(device)
        solvents_b = solvents_b.to(device)
        logits = model(anions_b, ligands_b, solvents_b)
        probs = F.softmax(logits, dim=1)
        val_probs.extend(probs.cpu().numpy())
val_probs = np.array(val_probs)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(len(classes)):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], val_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curves
plt.figure(figsize=(8, 6))
colors = ['blue', 'red', 'green']
for i, color in zip(range(len(classes)), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'ROC curve of class {class_names[i]} (area = {roc_auc[i]:.2f})')
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-Class ROC Curves (One-vs-Rest)')
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig('roc_curves.png', dpi=150)
plt.show()

# 4. Label Distribution Pie Chart
label_counts = [sum(1 for l in all_labels if l==0), sum(1 for l in all_labels if l==1), sum(1 for l in all_labels if l==2)]
plt.figure(figsize=(6, 6))
plt.pie(label_counts, labels=class_names, autopct='%1.1f%%', startangle=90, colors=['lightcoral', 'lightblue', 'lightgreen'])
plt.title('Label Distribution in Dataset')
plt.axis('equal')
plt.tight_layout()
plt.savefig('label_distribution.png', dpi=150)
plt.show()

# Classification reports
print("\n" + "="*80)
print("AGGREGATED CLASSIFICATION REPORTS")
print("="*80)

print("\nValidation Classification Report:")
print(classification_report(all_val_labels, all_val_preds, 
                          target_names=['no', 'slightly', 'yes'], digits=3))

print("\nTraining Classification Report:")
print(classification_report(all_train_labels, all_train_preds,
                          target_names=['no', 'slightly', 'yes'], digits=3))

print("\n" + "="*80)
print("✓ K-Fold Cross-Validation Completed!")
print(f"Best model weights saved as: best_model_fold0.pth to best_model_fold{n_splits-1}.pth")
print("="*80)
