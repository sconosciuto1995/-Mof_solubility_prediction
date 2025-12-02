from ase.io import read


def extract_last_snapshot(file_path):
    # Read all snapshots from the .xyz file
    snapshots = read(file_path, index=':')

    # Get the last snapshot
    last_snapshot = snapshots[-1]

    # Extract atoms and coordinates
    atoms = last_snapshot.get_chemical_symbols()
    coordinates = last_snapshot.get_positions()

    # Initialize energy variable
    energy = None

    # Extract energy from the last snapshot
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in reversed(lines):
            match = re.search(r'E\s*(-?\d+\.\d+)', line)
            if match:
                energy = float(match.group(1))
                break

    # Check if energy was found
    if energy is None:
        raise ValueError(f"Energy not found in file: {file_path}")

    return atoms, coordinates, energy
