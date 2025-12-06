import os
import torch
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch
from torch_geometric.data import Data
from GnnClass2 import CombinedModel
from helpers import extract_elements_and_positions, get_graph, create_dataset_from_solubility
from read_xyz import extract_last_snapshot

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

current_dir = os.getcwd()

# Dataset de triplets (structure simple : remplacer par lecture réelle)
class TripletDataset(Dataset):
    def __init__(self, triplets, labels):
        """
        triplets: list of tuples (anion_data, ligand_data, solvent_data)
        labels: list/array of int (0,1,2)
        """
        assert len(triplets) == len(labels)
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
    batch: list of (a,l,s,y)
    returns: batched_anions, batched_ligand, batched_solvent, labels_tensor
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
# LOAD DATASET FROM SOLUBILITY CSV
# ============================================================================

solubility_csv = current_dir + '/pillarplex_salts_solubility 1(Sheet1).csv'
anions_dir = 'mof_solubility/anions'
solvents_dir = 'mof_solubility/solvent'

# Define ligands for each Pillarplex ID
ligands_dict = {
    '2.1': 'mof_solubility/ligand/xyz/Ag_Pillarplex-Br.xyz',
    '3.1': 'mof_solubility/ligand/xyz/Au_Pillarplex-Br.xyz'  # Adjust path if needed
}

# Load complete dataset
print("Loading dataset from CSV...")
triplets_full, labels_full = create_dataset_from_solubility(
    solubility_csv_path=solubility_csv,
    anions_dir=anions_dir,
    ligands_dict=ligands_dict,
    solvents_dir=solvents_dir,
    pillarplex_ids=['2.1'],  # Can also use ['2.1', '3.1'] for multiple
    current_dir=current_dir
)

print(f"Total dataset size: {len(triplets_full)} samples")

# ============================================================================
# OPTION 1: USE A FRACTION OF THE DATASET FOR TESTING
# ============================================================================

# Set the fraction of data to use (e.g., 0.5 = 50% of data)
data_fraction = 1.0  # Change this to 0.5 for 50%, 0.2 for 20%, etc.

n_samples = max(1, int(len(triplets_full) * data_fraction))
print(f"Using {n_samples}/{len(triplets_full)} samples ({data_fraction*100:.0f}% of data)")

triplets = triplets_full[:n_samples]
labels = labels_full[:n_samples]

# ============================================================================
# CREATE DATASET AND DATALOADER
# ============================================================================

dataset = TripletDataset(triplets, labels)
batch_size = 4
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_triplets)

print(f"Number of batches per epoch: {len(loader)}")

# ============================================================================
# CREATE MODEL
# ============================================================================

# Get feature dimensions from first sample
first_triplet = triplets[0]
nodefeat_num = first_triplet[0].x.shape[-1]
edgefeat_num = first_triplet[0].edge_attr.shape[-1]

print(f"Node features: {nodefeat_num}, Edge features: {edgefeat_num}")

model = CombinedModel(
    nodefeat_num=nodefeat_num, 
    edgefeat_num=edgefeat_num,
    nodeembed_to=64,
    edgeembed_to=32,
    num_classes=3
)
model.to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total model parameters: {total_params}")

# ============================================================================
# TRAINING SETUP
# ============================================================================

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()
epochs = 50

# ============================================================================
# TRAINING LOOP
# ============================================================================

print("\nStarting training...")
for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (anions_b, ligands_b, solvents_b, labels_b) in enumerate(loader):
        anions_b = anions_b.to(device)
        ligands_b = ligands_b.to(device)
        solvents_b = solvents_b.to(device)
        labels_b = labels_b.to(device)

        optimizer.zero_grad()
        logits = model(anions_b, ligands_b, solvents_b)
        loss = criterion(logits, labels_b)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * labels_b.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels_b).sum().item()
        total += labels_b.size(0)

    avg_loss = epoch_loss / len(dataset)
    acc = correct / total
    if epoch % 5 == 0:
        print(f"Epoch {epoch:03d}  loss={avg_loss:.4f}  acc={acc:.3f}")

print("\nTraining completed!")

# ============================================================================
# EVALUATION
# ============================================================================

print("\nRunning evaluation...")
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for anions_b, ligands_b, solvents_b, labels_b in loader:
        anions_b = anions_b.to(device)
        ligands_b = ligands_b.to(device)
        solvents_b = solvents_b.to(device)
        labels_b = labels_b.to(device)
        
        logits = model(anions_b, ligands_b, solvents_b)
        preds = logits.argmax(dim=1)
        correct += (preds == labels_b).sum().item()
        total += labels_b.size(0)
    
    eval_acc = correct / total
    print(f"Evaluation accuracy: {eval_acc:.3f}")