"""
Training script using real solubility data from Excel files.

Loads data from solubility/Ag_Pillarplex.xlsx and solubility/Au_Pillarplex.xlsx
and trains the CombinedModel with proper labels (yes/slightly/no).
"""

import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch_geometric.data import Batch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

from GnnClass2 import CombinedModel
from helpers import load_solubility_excel
from read_xyz import extract_last_snapshot

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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

# Create dataset and dataloader
dataset = TripletDataset(all_triplets, all_labels)

# Train/Validation split (80/20)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

batch_size = 4
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_triplets)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_triplets)

print(f"Train set: {len(train_dataset)} samples")
print(f"Validation set: {len(val_dataset)} samples")
print(f"Batch size: {batch_size}")
print(f"Number of batches per epoch: {len(train_loader)}\n")

# ============================================================================
# MODEL SETUP
# ============================================================================

print("="*80)
print("Creating model...")
print("="*80)

# Get feature dimensions from first sample
first_triplet = all_triplets[0]
nodefeat_num = first_triplet[0].x.shape[-1]
edgefeat_num = first_triplet[0].edge_attr.shape[-1]

print(f"Node features: {nodefeat_num}")
print(f"Edge features: {edgefeat_num}")

model = CombinedModel(
    nodefeat_num=nodefeat_num,
    edgefeat_num=edgefeat_num,
    nodeembed_to=64,
    edgeembed_to=32,
    num_classes=3  # 0='no', 1='slightly', 2='yes'
)
model.to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total model parameters: {total_params}\n")

# ============================================================================
# TRAINING SETUP
# ============================================================================

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()
epochs = 100

print("="*80)
print("Starting training...")
print("="*80)

# ============================================================================
# TRAINING LOOP
# ============================================================================

best_val_acc = 0
patience = 15
patience_counter = 0

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
        logits = model(anions_b, ligands_b, solvents_b)  # shape [batch_size, 3]
        loss = criterion(logits, labels_b)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * labels_b.size(0)
        preds = logits.argmax(dim=1)
        train_correct += (preds == labels_b).sum().item()
        train_total += labels_b.size(0)

    avg_train_loss = train_loss / len(train_dataset)
    train_acc = train_correct / train_total

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
    
    # Early stopping
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        # Save best model
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        patience_counter += 1
    
    if epoch % 10 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch:03d}  train_loss={avg_train_loss:.4f}  train_acc={train_acc:.3f}  "
              f"val_loss={avg_val_loss:.4f}  val_acc={val_acc:.3f}")
    
    if patience_counter >= patience:
        print(f"\nEarly stopping at epoch {epoch} (validation accuracy not improving)")
        break

print("\n" + "="*80)
print("Training completed!")
print(f"Best validation accuracy: {best_val_acc:.3f}")
print("="*80)

# ============================================================================
# EVALUATION ON VALIDATION SET
# ============================================================================

print("\nLoading best model...")
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.eval()

label_names = {0: 'no', 1: 'slightly', 2: 'yes'}

# Evaluate on validation set
print("\n" + "="*80)
print("VALIDATION SET RESULTS")
print("="*80)

correct_per_class = [0, 0, 0]
total_per_class = [0, 0, 0]
overall_correct = 0
overall_total = 0
val_preds = []
val_labels = []

with torch.no_grad():
    for anions_b, ligands_b, solvents_b, labels_b in val_loader:
        anions_b = anions_b.to(device)
        ligands_b = ligands_b.to(device)
        solvents_b = solvents_b.to(device)
        labels_b = labels_b.to(device)
        
        logits = model(anions_b, ligands_b, solvents_b)
        preds = logits.argmax(dim=1)
        
        # Store for confusion matrix
        val_preds.extend(preds.cpu().numpy())
        val_labels.extend(labels_b.cpu().numpy())
        
        # Overall accuracy
        overall_correct += (preds == labels_b).sum().item()
        overall_total += labels_b.size(0)
        
        # Per-class accuracy
        for class_idx in range(3):
            class_mask = labels_b == class_idx
            if class_mask.sum() > 0:
                class_correct = (preds[class_mask] == labels_b[class_mask]).sum().item()
                correct_per_class[class_idx] += class_correct
                total_per_class[class_idx] += class_mask.sum().item()

print(f"\nOverall Accuracy: {overall_correct/overall_total:.3f}")
print("\nPer-class Accuracy:")
for class_idx in range(3):
    if total_per_class[class_idx] > 0:
        class_acc = correct_per_class[class_idx] / total_per_class[class_idx]
        print(f"  {label_names[class_idx]:10s}: {class_acc:.3f} ({correct_per_class[class_idx]}/{total_per_class[class_idx]})")
    else:
        print(f"  {label_names[class_idx]:10s}: No samples")

# Confusion Matrix (Validation)
print("\nConfusion Matrix (Validation Set):")
val_cm = confusion_matrix(val_labels, val_preds, labels=[0, 1, 2])
print(f"{'':15} Pred_no  Pred_slightly  Pred_yes")
for i, class_name in enumerate(['no', 'slightly', 'yes']):
    print(f"True_{class_name:8s}:  {val_cm[i, 0]:5d}    {val_cm[i, 1]:5d}        {val_cm[i, 2]:5d}")

print("\nClassification Report (Validation Set):")
print(classification_report(val_labels, val_preds, target_names=['no', 'slightly', 'yes'], digits=3))

# Also evaluate on training set for comparison
print("\n" + "="*80)
print("TRAINING SET RESULTS (for reference)")
print("="*80)

correct_per_class = [0, 0, 0]
total_per_class = [0, 0, 0]
overall_correct = 0
overall_total = 0
train_preds = []
train_labels = []

with torch.no_grad():
    for anions_b, ligands_b, solvents_b, labels_b in train_loader:
        anions_b = anions_b.to(device)
        ligands_b = ligands_b.to(device)
        solvents_b = solvents_b.to(device)
        labels_b = labels_b.to(device)
        
        logits = model(anions_b, ligands_b, solvents_b)
        preds = logits.argmax(dim=1)
        
        # Store for confusion matrix
        train_preds.extend(preds.cpu().numpy())
        train_labels.extend(labels_b.cpu().numpy())
        
        # Overall accuracy
        overall_correct += (preds == labels_b).sum().item()
        overall_total += labels_b.size(0)
        
        # Per-class accuracy
        for class_idx in range(3):
            class_mask = labels_b == class_idx
            if class_mask.sum() > 0:
                class_correct = (preds[class_mask] == labels_b[class_mask]).sum().item()
                correct_per_class[class_idx] += class_correct
                total_per_class[class_idx] += class_mask.sum().item()

print(f"\nOverall Accuracy: {overall_correct/overall_total:.3f}")
print("\nPer-class Accuracy:")
for class_idx in range(3):
    if total_per_class[class_idx] > 0:
        class_acc = correct_per_class[class_idx] / total_per_class[class_idx]
        print(f"  {label_names[class_idx]:10s}: {class_acc:.3f} ({correct_per_class[class_idx]}/{total_per_class[class_idx]})")
    else:
        print(f"  {label_names[class_idx]:10s}: No samples")

# Confusion Matrix (Training)
print("\nConfusion Matrix (Training Set):")
train_cm = confusion_matrix(train_labels, train_preds, labels=[0, 1, 2])
print(f"{'':15} Pred_no  Pred_slightly  Pred_yes")
for i, class_name in enumerate(['no', 'slightly', 'yes']):
    print(f"True_{class_name:8s}:  {train_cm[i, 0]:5d}    {train_cm[i, 1]:5d}        {train_cm[i, 2]:5d}")

print("\nClassification Report (Training Set):")
print(classification_report(train_labels, train_preds, target_names=['no', 'slightly', 'yes'], digits=3))

print("\n" + "="*80)
