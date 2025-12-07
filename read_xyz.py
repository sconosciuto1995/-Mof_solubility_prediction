from ase.io import read
import re

def extract_last_snapshot(file_path, return_energy=False):
    """
    Read the last snapshot from an XYZ file.
    By default returns (atoms_list, positions_array).
    If return_energy=True returns (atoms_list, positions_array, energy_or_None).
    """
    snapshots = read(file_path, index=':')
    last_snapshot = snapshots[-1]
    atoms = last_snapshot.get_chemical_symbols()
    coordinates = last_snapshot.get_positions()

    if return_energy:
        energy = None
        try:
            with open(file_path, 'r') as file:
                for line in reversed(file.readlines()):
                    match = re.search(r'E\s*(-?\d+\.\d+)', line)
                    if match:
                        energy = float(match.group(1))
                        break
        except Exception:
            energy = None
        return atoms, coordinates, energy

    return atoms, coordinates
# ...existing code...