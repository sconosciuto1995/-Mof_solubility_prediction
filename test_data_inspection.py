"""
Simple script to inspect and print the return values of data extraction functions.
"""

import os
from helpers import extract_elements_and_positions
from read_xyz import extract_last_snapshot

current_dir = os.getcwd()

print("=" * 80)
print("TESTING extract_elements_and_positions (CSV files)")
print("=" * 80)

# Test CSV loading
csv_path = os.path.join(current_dir, 'anions', 'BF4.csv')
if os.path.exists(csv_path):
    print(f"\nLoading: {csv_path}")
    elements, positions = extract_elements_and_positions(csv_path)
    print(f"Type of elements: {type(elements)}")
    print(f"Elements: {elements}")
    print(f"\nType of positions: {type(positions)}")
    print(f"Positions shape: {positions.shape}")
    print(f"Positions dtype: {positions.dtype}")
    print(f"Positions:\n{positions}")
else:
    print(f"File not found: {csv_path}")

print("\n" + "=" * 80)
print("TESTING extract_last_snapshot (XYZ files)")
print("=" * 80)

# Test XYZ loading
xyz_path = os.path.join(current_dir, 'ligand', 'xyz', 'Ag_Pillarplex-Br.xyz')
if os.path.exists(xyz_path):
    print(f"\nLoading: {xyz_path}")
    atoms, coords = extract_last_snapshot(xyz_path)
    print(f"Type of atoms: {type(atoms)}")
    print(f"Number of atoms: {len(atoms)}")
    print(f"Atoms: {atoms}")
    print(f"\nType of coords: {type(coords)}")
    print(f"Coords shape: {coords.shape}")
    print(f"Coords dtype: {coords.dtype}")
    print(f"First 5 coordinates:\n{coords[:5]}")
else:
    print(f"File not found: {xyz_path}")

print("\n" + "=" * 80)
print("TESTING extract_last_snapshot with energy extraction")
print("=" * 80)

if os.path.exists(xyz_path):
    print(f"\nLoading: {xyz_path} (with energy)")
    atoms, coords, energy = extract_last_snapshot(xyz_path, return_energy=True)
    print(f"Type of atoms: {type(atoms)}")
    print(f"Atoms: {atoms}")
    print(f"\nType of coords: {type(coords)}")
    print(f"Coords shape: {coords.shape}")
    print(f"\nType of energy: {type(energy)}")
    print(f"Energy value: {energy}")
else:
    print(f"File not found: {xyz_path}")

print("\n" + "=" * 80)
print("TESTING multiple solvent files")
print("=" * 80)

solvent_files = ['Acetone.csv', 'H2O.csv', 'DMF.csv']
for fname in solvent_files:
    solvent_path = os.path.join(current_dir, 'solvents', fname)
    if os.path.exists(solvent_path):
        print(f"\nLoading: {solvent_path}")
        elements, positions = extract_elements_and_positions(solvent_path)
        print(f"  Elements: {elements}")
        print(f"  Positions shape: {positions.shape}")
    else:
        print(f"File not found: {solvent_path}")

print("\n" + "=" * 80)
