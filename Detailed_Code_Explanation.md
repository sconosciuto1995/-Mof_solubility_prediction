# Detailed Code Explanation: MOF Solubility Prediction Project

This document provides a comprehensive, step-by-step explanation of the code in the MOF solubility prediction project. The project uses Graph Neural Networks (GNNs) and Gaussian Processes to predict the solubility of Metal-Organic Frameworks (MOFs) based on their anion, ligand, and solvent components.

## Project Overview

The project aims to predict MOF solubility by modeling the interactions between three molecular components:
- **Anions**: Negatively charged ions (e.g., ClO₄⁻, BF₄⁻)
- **Ligands**: Organic molecules that coordinate with metal centers (e.g., Pillarplex structures)
- **Solvents**: Liquid media where solubility is tested (e.g., water, acetone)

The approach uses graph-based machine learning where each molecule is represented as a graph, and GNNs learn to extract features from these graphs.

## 1. Data Structure and Input Files

### 1.1 Directory Structure
```
project/
├── anions/          # CSV files with anion atomic coordinates
├── ligand/xyz/      # XYZ files with ligand structures
├── solvents/        # CSV files with solvent atomic coordinates
├── solubility/      # Excel files with experimental solubility data
└── *.py files       # Python scripts for training and prediction
```

### 1.2 Data Formats

#### Anion and Solvent CSV Files
Each CSV file contains atomic coordinates in the format:
```csv
// Comment line (optional)
Element,X,Y,Z
C,1.234,2.345,3.456
O,4.567,5.678,6.789
...
```

**Step-by-step parsing in `extract_elements_and_positions()` (helpers.py:87-125):**
1. Open the CSV file and read the first line
2. Check if it's a comment line (starts with '//')
3. Read the header line to identify column structure
4. Initialize empty lists for elements and positions
5. For each subsequent line:
   - Split by comma to get [element, x, y, z]
   - Strip whitespace from each field
   - Convert x, y, z to float coordinates
   - Append element to elements list
   - Append [x, y, z] to positions list
6. Convert positions list to numpy array of shape (n_atoms, 3)
7. Return (elements_list, positions_array)

#### Ligand XYZ Files
XYZ files contain molecular structures in standard chemical format:
```
<number_of_atoms>
Comment line with energy (optional)
Element X Y Z
Element X Y Z
...
```

**Step-by-step parsing in `extract_last_snapshot()` (read_xyz.py:4-25):**
1. Use ASE (Atomic Simulation Environment) library to read all snapshots: `snapshots = read(file_path, index=':')`
2. Select the last snapshot: `last_snapshot = snapshots[-1]`
3. Extract chemical symbols: `atoms = last_snapshot.get_chemical_symbols()`
4. Extract 3D coordinates: `coordinates = last_snapshot.get_positions()`
5. If `return_energy=True`, search backwards through file for energy pattern `E <number>`
6. Return atoms list and coordinates array

#### Solubility Excel Files
Excel files contain experimental solubility data organized by Pillarplex type:
- First column: Anion IDs
- Subsequent columns: Solvent names
- Data cells: "yes", "slightly", "no", or "-" for missing data

## 2. Graph Representation Creation

### 2.1 Atomic Feature Engineering

**In `get_graph()` function (helpers.py:8-75):**

#### Step 1: Create Fully Connected Edge Index
```python
a = np.arange(len(atoms))  # [0, 1, 2, ..., n-1]
edges = np.array(np.meshgrid(a, a)).T.reshape(-1, 2).T
edges = torch.tensor(edges, dtype=torch.int64)
```
This creates a complete graph where every atom connects to every other atom, resulting in n² edges for n atoms.

#### Step 2: Atomic Number Mapping
```python
atom_to_num = {
    'C': 6, 'O': 8, 'Zn': 30, 'Pt': 78,
    'H': 1, 'Br': 35, 'I': 53,
    'F': 9, 'Cl': 17, 'S': 16, 'N': 7, 'B': 5, 'Ag': 47, 'P': 15, 'Au': 79
}
```
Maps element symbols to their atomic numbers.

#### Step 3: Electronegativity Mapping
```python
atom_to_en = {
    'C': 2.55, 'O': 3.44, 'Zn': 1.65, 'Pt': 2.28,
    'H': 2.20, 'Br': 2.96, 'I': 2.66,
    'F': 3.98, 'Cl': 3.16, 'S': 2.58, 'N': 3.04, 'B': 2.04, 'Ag': 1.93, 'P': 2.19, 'Au': 2.54
}
```
Pauling electronegativity values for each element.

#### Step 4: Atomic Radius Mapping
```python
atom_to_r = {
    'C': 70, 'O': 60, 'Zn': 135, 'Pt': 135,
    'H': 25, 'Br': 114, 'I': 133,
    'F': 50, 'Cl': 99, 'S': 105, 'N': 65, 'B': 85, 'Ag': 165, 'P': 107, 'Au': 144
}
```
Approximate covalent radii in picometers.

#### Step 5: Feature Array Creation
```python
atomic_nums = np.asarray([atom_to_num[atom] for atom in atoms])[:, np.newaxis]
electroneg = torch.tensor(np.asarray([atom_to_en[atom] for atom in atoms])[:, np.newaxis], dtype=torch.float)
atomic_radius = torch.tensor(np.asarray([atom_to_r[atom] for atom in atoms])[:, np.newaxis], dtype=torch.float)
```
Creates separate feature arrays for each atomic property.

### 2.2 Edge Feature Engineering

#### Coulomb Matrix Calculation
```python
pair_dist = pairwise_distances(pos)  # Shape: (n_atoms, n_atoms)
cm = (atomic_nums * atomic_nums.T) / pair_dist
np.fill_diagonal(cm, 0.5 * atomic_nums**2.4)
cm = cm.flatten()[:, np.newaxis]
```
- `pairwise_distances(pos)` computes Euclidean distances between all atom pairs
- Coulomb matrix: C_ij = (Z_i * Z_j) / r_ij for i≠j
- Diagonal: C_ii = 0.5 * Z_i^2.4 (standard approximation)

#### Edge Features Concatenation
```python
edge_attr = torch.tensor(cm, dtype=torch.float)
edge_attr = torch.cat([torch.tensor(cm, dtype=torch.float), 
                      torch.tensor(pair_dist.flatten()[:, np.newaxis], dtype=torch.float)], dim=1)
```
Combines Coulomb matrix values and pairwise distances into edge feature matrix.

### 2.3 Node Feature Concatenation
```python
node_attrs = torch.cat([torch.tensor(atomic_nums, dtype=torch.float), 
                       electroneg, atomic_radius], dim=1)
```
Creates final node feature matrix with shape (n_atoms, 3): [atomic_number, electronegativity, atomic_radius]

### 2.4 PyTorch Geometric Data Object
```python
graph = Data(x=node_attrs, edge_index=edges, edge_attr=edge_attr, y=target)
```
Creates a PyG Data object containing:
- `x`: Node features (n_atoms, 3)
- `edge_index`: Edge connectivity (2, n_edges)
- `edge_attr`: Edge features (n_edges, 2)
- `y`: Target value (optional)

## 3. Graph Neural Network Architecture

### 3.1 SingleGNN Class (GnnClass2.py:10-54)

#### Node Embedding Layer
```python
self._node_embedding = nn.Sequential(
    nn.Linear(nodefeat_num, nodeembed_to),  # e.g., 3 → 64
    nn.ReLU()
)
self._node_embednorm = nn.BatchNorm1d(nodeembed_to)
```
- Linear transformation from input features (3) to embedding dimension (64)
- ReLU activation and batch normalization

#### Edge Embedding Layer
```python
self._edge_embedding = nn.Sequential(
    nn.Linear(edgefeat_num, edgeembed_to),  # e.g., 2 → 32
    nn.ReLU()
)
self._edge_embednorm = nn.BatchNorm1d(edgeembed_to)
```
Similar embedding for edge features.

#### Graph Convolution Layers
```python
self._first_conv = NNConv(
    nodeembed_to,  # Input node features: 64
    nodeembed_to,  # Output node features: 64
    nn.Sequential(nn.Linear(edgeembed_to, nodeembed_to**2), nn.Tanh()),  # 32 → 4096 → Tanh
    aggr='mean'  # Mean aggregation of neighbor messages
)
```
- NNConv: Neural Network Convolution layer
- Learns how to combine node features with edge features
- Uses mean aggregation from neighboring nodes

#### Second Convolution Layer
```python
self._second_conv = NNConv(
    nodeembed_to,
    nodeembed_to,
    nn.Sequential(nn.Linear(edgeembed_to, nodeembed_to**2), nn.ReLU()),
    aggr='mean'
)
```
Similar structure but with ReLU activation.

#### Global Pooling
```python
self._pooling = [global_mean_pool, global_max_pool]
pooled = torch.cat([p(x, batch_vec) for p in self._pooling], dim=1)
```
- Applies both mean and max pooling across all nodes in the graph
- Concatenates results: shape becomes (batch_size, 2 * nodeembed_to)

### 3.2 CombinedModel Class (GnnClass2.py:56-89)

#### Three Separate GNNs
```python
self.anion_gnn = SingleGNN(nodefeat_num, edgefeat_num, nodeembed_to, edgeembed_to)
self.ligand_gnn = SingleGNN(nodefeat_num, edgefeat_num, nodeembed_to, edgeembed_to)
self.solvent_gnn = SingleGNN(nodefeat_num, edgefeat_num, nodeembed_to, edgeembed_to)
```
Each component gets its own GNN to learn component-specific features.

#### Feature Concatenation
```python
total_feat = (self.anion_gnn.output_size + 
              self.ligand_gnn.output_size + 
              self.solvent_gnn.output_size)  # 3 * 128 = 384
```

#### Prediction Head
```python
self.predictor = nn.Sequential(
    nn.Linear(total_feat, 128),    # 384 → 128
    nn.ReLU(),
    nn.Dropout(0.2),               # Regularization
    nn.Linear(128, 64),            # 128 → 64
    nn.ReLU(),
    nn.Dropout(0.2),               # Regularization
    nn.Linear(64, num_classes)     # 64 → 3 (no/slightly/yes)
)
```

#### Forward Pass
```python
a_repr = self.anion_gnn(anion_graph)
l_repr = self.ligand_gnn(ligand_graph)
s_repr = self.solvent_gnn(solvent_graph)
combined = torch.cat([a_repr, l_repr, s_repr], dim=1)
logits = self.predictor(combined)
```

## 4. Data Loading and Dataset Creation

### 4.1 Solubility Data Parsing

**`parse_solubility_csv()` (helpers.py:127-200):**
1. Read CSV with pandas
2. Iterate through rows to find Pillarplex sections (marked by ";Solvent:")
3. Extract solvent names from the row after section header
4. Parse anion IDs and solubility labels from subsequent rows
5. Map text labels ("no"/"slightly"/"yes") to numeric (0/1/2)
6. Return list of (anion_id, solvent, label, pillarplex_id) tuples

**`load_solubility_excel()` (helpers.py:355-539):**
1. Load anion mapping from Excel (ID → name)
2. Read main Excel file with pandas
3. Extract solvent names from first row
4. Parse anion data starting from row 14
5. For each anion-solvent combination:
   - Load anion graph from CSV file
   - Load solvent graph from CSV file
   - Load ligand graph from XYZ file (cached)
   - Create triplet: (anion_graph, ligand_graph, solvent_graph)
   - Store with corresponding label

### 4.2 Dataset Class

**`TripletDataset` (TrainSolubility.py:25-35):**
- Stores list of triplets and corresponding labels
- `__getitem__` returns individual (anion, ligand, solvent, label) tuples
- Handles conversion of labels to appropriate tensor types

### 4.3 Collate Function

**`collate_triplets()` (TrainSolubility.py:37-50):**
1. Extract individual components from batch
2. Create separate lists for anions, ligands, solvents, labels
3. Use `Batch.from_data_list()` to create batched graphs
4. Convert labels to tensor
5. Return (batched_anions, batched_ligands, batched_solvents, labels_tensor)

## 5. Training Process

### 5.1 Training Setup (TrainSolubility.py:151-190)

#### Data Splitting
```python
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size], 
                                         generator=torch.Generator().manual_seed(42))
```
- 80/20 train/validation split
- Fixed random seed for reproducibility

#### Model Initialization
```python
model = CombinedModel(
    nodefeat_num=nodefeat_num,  # 3
    edgefeat_num=edgefeat_num,  # 2
    nodeembed_to=64,
    edgeembed_to=32,
    num_classes=3
)
```

#### Optimizer and Loss
```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()
```

### 5.2 Training Loop (TrainSolubility.py:195-235)

#### Training Phase
For each epoch:
1. Set model to train mode: `model.train()`
2. Initialize epoch metrics: `train_loss = 0.0`, `train_correct = 0`, `train_total = 0`
3. For each batch:
   - Move batch to device (GPU/CPU)
   - Zero gradients: `optimizer.zero_grad()`
   - Forward pass: `logits = model(anions_b, ligands_b, solvents_b)`
   - Compute loss: `loss = criterion(logits, labels_b)`
   - Backward pass: `loss.backward()`
   - Update weights: `optimizer.step()`
   - Accumulate metrics

#### Validation Phase
1. Set model to eval mode: `model.eval()`
2. Disable gradient computation: `with torch.no_grad():`
3. Compute validation loss and accuracy
4. Check for best model and early stopping

#### Early Stopping
```python
if val_acc > best_val_acc:
    best_val_acc = val_acc
    patience_counter = 0
    torch.save(model.state_dict(), 'best_model.pth')
else:
    patience_counter += 1

if patience_counter >= patience:
    print("Early stopping...")
    break
```

### 5.3 Evaluation (TrainSolubility.py:237-429)

#### Confusion Matrix and Classification Report
- Compute predictions on validation set
- Generate confusion matrix and per-class metrics
- Display results with seaborn heatmap

## 6. Gaussian Process Extension

### 6.1 Feature Extraction (TrainSolubilityGP.py:115-170)

#### Load Pre-trained GNN
```python
base_model = CombinedModel(...)
base_model.load_state_dict(torch.load('best_model.pth'))
feature_extractor = FeatureExtractor(base_model)
```
- Load the trained classification model
- Create FeatureExtractor that outputs intermediate representations

#### Freeze Feature Extractor
```python
feature_extractor.eval()
for param in feature_extractor.parameters():
    param.requires_grad = False
```

#### Extract Features
```python
with torch.no_grad():
    for anions_b, ligands_b, solvents_b, labels_b in train_loader:
        features = feature_extractor(anions_b, ligands_b, solvents_b)
        train_features_list.append(features)
        train_labels_list.append(labels_b)

train_x = torch.cat(train_features_list, dim=0)
train_y = torch.cat(train_labels_list, dim=0)
```
- Convert discrete labels {0,1,2} to continuous {0.0, 0.5, 1.0}
- Extract 384-dimensional feature vectors for each sample

### 6.2 GP Model Definition (GnnClass2.py:120-169)

#### ExactGPLayer
```python
class ExactGPLayer(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
```
- Constant mean function
- RBF (Gaussian) kernel with scale parameter

#### ExactGNNGP
```python
class ExactGNNGP(nn.Module):
    def __init__(self, feature_extractor, train_x, train_y):
        self.feature_extractor = feature_extractor
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.gp_layer = ExactGPLayer(train_x, train_y, self.likelihood)
```

### 6.3 GP Training (TrainSolubilityGP.py:201-240)

#### Marginal Log Likelihood
```python
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp_model)
```

#### Training Loop
```python
for epoch in range(epochs):
    optimizer.zero_grad()
    output = gp_model(train_x)
    loss = -mll(output, train_y)  # Negative because we maximize likelihood
    loss.backward()
    optimizer.step()
```

### 6.4 GP Prediction with Uncertainty (TrainSolubilityGP.py:242-290)

#### Prediction
```python
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    pred_dist = likelihood(gp_model(val_x))
    mean = pred_dist.mean
    lower, upper = pred_dist.confidence_region()
```

#### Uncertainty Quantification
- Mean prediction
- 95% confidence intervals
- Coverage statistics: percentage of true values within predicted intervals

## 7. Prediction and Inference

### 7.1 Loading Pre-trained Models

#### Classification Model
```python
model = CombinedModel(...)
model.load_state_dict(torch.load('best_model.pth'))
model.eval()
```

#### GP Model
```python
checkpoint = torch.load('best_model_gp.pth')
gp_model.load_state_dict(checkpoint['gp_model'])
likelihood.load_state_dict(checkpoint['likelihood'])
```

### 7.2 Inference Pipeline

1. **Load molecular data**: Parse anion, ligand, solvent files
2. **Create graphs**: Use `get_graph()` for each component
3. **Batch graphs**: Use `Batch.from_data_list()` for batched inference
4. **Forward pass**: Get predictions from model
5. **Post-processing**: Convert logits to probabilities/classes

## 8. Key Technical Details

### 8.1 Graph Construction Details
- **Fully connected graphs**: Every atom connected to every other atom
- **Node features**: [atomic_number, electronegativity, covalent_radius]
- **Edge features**: [coulomb_interaction, euclidean_distance]
- **No 3D information preservation**: Only pairwise distances encoded

### 8.2 GNN Architecture Choices
- **NNConv**: Learns edge-dependent message passing
- **Two convolution layers**: Balances expressiveness and overfitting
- **Mean aggregation**: Simple but effective for molecular graphs
- **Dual pooling**: Mean + Max pooling captures different aspects

### 8.3 Training Considerations
- **Batch size 4**: Limited by memory and small dataset
- **Early stopping**: Prevents overfitting on small datasets
- **Adam optimizer**: Standard choice with lr=1e-3
- **Dropout regularization**: 0.2 rate in classifier head

### 8.4 GP Advantages
- **Uncertainty quantification**: Provides confidence intervals
- **Continuous predictions**: More nuanced than discrete classes
- **Non-parametric**: Flexible function approximation
- **Exact inference**: Tractable for moderate dataset sizes

## 9. Limitations and Considerations

### 9.1 Data Limitations
- Small dataset size limits model complexity
- Limited chemical diversity in training data
- Experimental solubility labels may have measurement uncertainty

### 9.2 Model Limitations
- Fully connected graphs ignore chemical bonding patterns
- No explicit 3D spatial encoding beyond pairwise distances
- Limited to three-component systems

### 9.3 Computational Considerations
- GNN training requires significant GPU memory
- GP scales as O(n³) for exact inference
- Feature extraction is computationally expensive

## 10. File Dependencies and Execution Order

1. **Data preparation**: `helpers.py` functions
2. **Model definition**: `GnnClass2.py`
3. **Training classification**: `TrainSolubility.py`
4. **Training GP**: `TrainSolubilityGP.py` (requires `best_model.pth`)
5. **Inference**: `RunNetwork2.py` or custom scripts

This comprehensive pipeline enables both accurate classification and uncertainty-aware continuous predictions for MOF solubility, with detailed molecular feature engineering and advanced machine learning techniques.</content>
<parameter name="filePath">c:\Users\Anwender\Desktop\gitrepo\-Mof_solubility_prediction\Detailed_Code_Explanation.md