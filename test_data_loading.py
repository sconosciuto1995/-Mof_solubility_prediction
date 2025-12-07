"""
Test data loading from Excel to verify structure matches experiments.
"""

import os
import pandas as pd
from helpers import load_solubility_excel

current_dir = os.getcwd()

print("="*80)
print("TESTING DATA LOADING FROM EXCEL")
print("="*80)

# Test structure
ligands_dict = {
    '2.1': 'ligand/xyz/Ag_Pillarplex-Br.xyz',
    '3.1': 'ligand/xyz/Au_Pillarplex-Br.xyz'
}

anions_dir = 'anions'
solvents_dir = 'solvents'

# First, let's see raw Excel structure
print("\n1. Raw Excel Structure (Ag_Pillarplex.xlsx):")
print("-" * 80)

excel_path = os.path.join(current_dir, 'solubility', 'Ag_Pillarplex.xlsx')
df = pd.read_excel(excel_path, sheet_name='Sheet1')

print(f"Shape: {df.shape}")
print(f"First column (anion IDs): {df.iloc[14:, 0].tolist()}")  # Starting from row 14
print(f"Solvents (header): {df.iloc[0, 1:].tolist()}")  # Solvents from row 0, col 1+

print("\nSample data (rows 14-17, showing some cells):")
for row_idx in range(14, 18):
    anion_id = df.iloc[row_idx, 0]
    acetone_solubility = df.iloc[row_idx, 1]
    h2o_solubility = df.iloc[row_idx, 2]
    print(f"  Anion {anion_id}: Acetone={acetone_solubility}, H2O={h2o_solubility}")

# Now test the loading function
print("\n" + "="*80)
print("2. Testing load_solubility_excel() function:")
print("-" * 80)

try:
    triplets, labels = load_solubility_excel(
        excel_path=excel_path,
        anions_dir=anions_dir,
        ligands_dict=ligands_dict,
        solvents_dir=solvents_dir,
        pillarplex_id='2.1',
        current_dir=current_dir
    )
    
    print(f"\n✓ Successfully loaded {len(triplets)} samples")
    print(f"  Labels breakdown:")
    print(f"    - no (0): {sum(1 for l in labels if l==0)}")
    print(f"    - slightly (1): {sum(1 for l in labels if l==1)}")
    print(f"    - yes (2): {sum(1 for l in labels if l==2)}")
    
    if len(triplets) > 0:
        print(f"\nFirst sample structure:")
        anion_g, ligand_g, solvent_g = triplets[0]
        print(f"  Anion graph: {anion_g.x.shape[0]} nodes, {anion_g.edge_index.shape[1]} edges")
        print(f"  Ligand graph: {ligand_g.x.shape[0]} nodes, {ligand_g.edge_index.shape[1]} edges")
        print(f"  Solvent graph: {solvent_g.x.shape[0]} nodes, {solvent_g.edge_index.shape[1]} edges")
        print(f"  Label: {labels[0]} (mapping: 0=no, 1=slightly, 2=yes)")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("SUMMARY:")
print("="*80)
print("""
Each experiment consists of:
  - 1 ANION (from anions/*.csv) - encoded as graph 1
  - 1 SOLVENT (from solvents/*.csv) - encoded as graph 2  
  - 1 LIGAND (from Excel filename, e.g., Ag_Pillarplex or Au_Pillarplex) - encoded as graph 3
  - 1 OUTCOME (yes/slightly/no) - the target label

The GNN will:
  1. Process each graph separately (anion_GNN, solvent_GNN, ligand_GNN)
  2. Concatenate the 3 representations
  3. Pass through final MLP to predict dissolution outcome
""")
