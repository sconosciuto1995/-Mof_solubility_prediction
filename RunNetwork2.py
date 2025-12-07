# ...existing code...
import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch
from torch_geometric.data import Data
from GnnClass2 import CombinedModel
from helpers import extract_elements_and_positions, get_graph
from read_xyz import extract_last_snapshot

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Charger un exemple de triplet (remplacer par le chargement complet du dataset)
current_dir = os.getcwd()
datafile_anions = current_dir + '/anions/ClO4.csv'
atoms_anions_list, atoms_anions_pos_list = extract_elements_and_positions(datafile_anions)

datafile_ligand = current_dir + '/ligand/xyz/Ag_Pillarplex-Br.xyz'
atoms_ligand_list, atom_ligand_pos_list = extract_last_snapshot(datafile_ligand)

datafile_solvent = current_dir + '/solvents/Acetone.csv'
atoms_solvent_list, atoms_solvent_pos_list = extract_elements_and_positions(datafile_solvent)

graph_anions_data = get_graph(atoms_anions_list, atoms_anions_pos_list)   # torch_geometric.data.Data
graph_ligand_data = get_graph(atoms_ligand_list, atom_ligand_pos_list)
graph_solvent_data = get_graph(atoms_solvent_list, atoms_solvent_pos_list)

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
    retourne: batched_anions, batched_ligand, batched_solvent, labels_tensor
    """
    anions = [item[0] for item in batch]
    ligands = [item[1] for item in batch]
    solvents = [item[2] for item in batch]
    labels = torch.tensor([item[3] for item in batch], dtype=torch.long)
    anions_batch = Batch.from_data_list(anions)
    ligands_batch = Batch.from_data_list(ligands)
    solvents_batch = Batch.from_data_list(solvents)
    return anions_batch, ligands_batch, solvents_batch, labels

# Exemple de dataset minimal (dupliquer l'exemple pour simuler plusieurs échantillons)
triplets = [(graph_anions_data, graph_ligand_data, graph_solvent_data)] * 8
labels = [0, 1, 2, 0, 1, 2, 0, 1]  # placeholder : 0=yes,1=slightly,2=no
dataset = TripletDataset(triplets, labels)
loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_triplets)

# Créer le modèle
nodefeat_num = graph_anions_data.x.shape[-1]
edgefeat_num = graph_anions_data.edge_attr.shape[-1]
model = CombinedModel(nodefeat_num=nodefeat_num, edgefeat_num=edgefeat_num, nodeembed_to=64, edgeembed_to=32, num_classes=3)
model.to(device)

# Optimizer & loss (classification)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()

# Training loop (exemple)
epochs = 30
for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0
    correct = 0
    total = 0
    for anions_b, ligands_b, solvents_b, labels_b in loader:
        anions_b = anions_b.to(device)
        ligands_b = ligands_b.to(device)
        solvents_b = solvents_b.to(device)
        labels_b = labels_b.to(device)

        optimizer.zero_grad()
        logits = model(anions_b, ligands_b, solvents_b)  # shape [batch_size, 3]
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

# Évaluation simple
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
    print(f"Eval acc: {correct/total:.3f}")

# Note: remplacer la construction du dataset par la lecture réelle de vos fichiers et labels.