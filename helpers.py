import numpy as np
from sklearn.metrics import pairwise_distances
import torch 
from torch_geometric.data import Data, Batch
import os
import csv
import pandas as pd

from read_xyz import extract_last_snapshot
def get_graph(atoms, pos, target=None):
    '''
    Get a Data graph from the numpy coordinates, the type of atom and the target.
    '''
        # edge index   
    # we create edges that are fully connected. IE. the following lines
    # will create a 2d vector with all-to-all connections
    # i.e. if we have 3 nodes (atoms)
    # the following commands will create the vector [0,0,0,1,1,1,2,2,2][0,1,2,0,1,2,0,1,2]    
    a = np.arange(len(atoms))
    edges = np.array(np.meshgrid(a,a)).T.reshape(-1,2).T
    edges = torch.tensor(edges, dtype=torch.int64)

    #here we will create some values for the nodes. These values will come from
    # the properties of the atom that each node represents
    atom_to_num = {
        'C': 6, 'O': 8, 'Zn': 30, 'Pt': 78,
        'H': 1, 'Br': 35, 'I': 53,
        'F': 9, 'Cl': 17, 'S': 16, 'N': 7, 'B': 5, 'Ag': 47, 'Bf': 5
    }  # atom to atomic number
    atom_to_en = {
        'C': 2.55, 'O': 3.44, 'Zn': 1.65, 'Pt': 2.28,
        'H': 2.20, 'Br': 2.96, 'I': 2.66,
        'F': 3.98, 'Cl': 3.16, 'S': 2.58, 'N': 3.04, 'B': 2.04, 'Ag': 1.93
    }  # atom to electronegativity (Pauling)
    atom_to_r = {
        'C': 70, 'O': 60, 'Zn': 135, 'Pt': 135,
        'H': 25, 'Br': 114, 'I': 133,
        'F': 50, 'Cl': 99, 'S': 105, 'N': 65, 'B': 85, 'Ag': 165
    }  # atom to (approx.) covalent radius in pm

    atomic_nums = np.asarray([atom_to_num[atom] for atom in atoms])[:, np.newaxis] # keep as numpy for later use
    electroneg = torch.tensor(np.asarray([atom_to_en[atom] for atom in atoms])[:, np.newaxis], dtype=torch.float)
    atomic_radius = torch.tensor(np.asarray([atom_to_r[atom] for atom in atoms])[:, np.newaxis], dtype=torch.float)

    # In the loop we extract the nodes' embeddings, edges connectivity 
    # and label for a graph, process the information and put it in a Data
    # object, then we add the object to a list

    # Node features
    # atomic number abd electronegativity 
    # Edge features
        # shape [N', D'] N': number of edges, D': number of edge features
        # cm matrix and bond matrix   

    # Here we calculate the coulomb matrix. This is a distance metric in a sense. 
    # it shows how connected two nodes are (strength of electric interaction)    
    pair_dist = pairwise_distances(pos)   
    cm = (atomic_nums*atomic_nums.T) / pair_dist
    np.fill_diagonal(cm, 0.5*atomic_nums**2.4)
    cm = cm.flatten()[:, np.newaxis]
    edge_attr = torch.tensor(cm, dtype=torch.float)
    edge_attr = torch.cat([torch.tensor(cm, dtype=torch.float), torch.tensor(pair_dist.flatten()[:, np.newaxis], dtype = torch.float)], dim = 1)
    # here we encode the molecule level energy
    if target:
        target = torch.tensor(target, dtype=torch.float)


    # and here we package all the node attributes (electronegativity, atomic radius and atomic number into one array)
    node_attrs = torch.cat([torch.tensor(atomic_nums, dtype=torch.float), electroneg,atomic_radius], dim=1)


    #the node attributes, edges, edge attributes and targets are packaged into one data object
    #that is very handy to use for representing the graph
    graph = Data(x=node_attrs,
            edge_index=edges,
            edge_attr=edge_attr, 
            y=target)
    
    return graph

def extract_elements_and_positions(csv_file_path):
    """
    Extrait les noms des éléments et leurs positions à partir d'un fichier CSV.
    
    Args:
        csv_file_path (str): Chemin vers le fichier CSV contenant les données atomiques
        
    Returns:
        tuple: (elements, positions)
            - elements (list): Liste des noms d'éléments (ex: ['F', 'F', 'F', 'F', 'B'])
            - positions (numpy.ndarray): Array numpy de shape (n_atoms, 3) contenant les coordonnées x, y, z
    
    Example:
        >>> elements, positions = extract_elements_and_positions('anions/BF4.csv')
        >>> print(elements)
        ['F', 'F', 'F', 'F', 'B']
        >>> print(positions.shape)
        (5, 3)
    """
    elements = []
    positions = []
    
    with open(csv_file_path, 'r') as file:
        # Ignorer la première ligne si elle contient un commentaire
        first_line = file.readline().strip()
        if first_line.startswith('//'):
            # Lire la ligne d'en-tête
            header_line = file.readline().strip()
        else:
            # La première ligne est déjà l'en-tête
            file.seek(0)  # Retour au début du fichier
            header_line = file.readline().strip()
        
        # Créer un lecteur CSV
        csv_reader = csv.reader(file)
        
        for row in csv_reader:
            if len(row) >= 4:  # Vérifier qu'on a au moins 4 colonnes (element, x, y, z)
                element = row[0].strip()
                x = float(row[1].strip())
                y = float(row[2].strip())
                z = float(row[3].strip())
                
                elements.append(element)
                positions.append([x, y, z])
    
    # Convertir les positions en array numpy
    positions = np.array(positions)
    
    return elements, positions

def parse_solubility_csv(csv_path, pillarplex_ids=None):
    """
    Parse the experimental solubility file and return triplets (anion, solvent, label).
    
    Args:
        csv_path (str): Path to the solubility CSV file
        pillarplex_ids (list or str): ID(s) of the Pillarplex to extract. 
                                      If None, extract all available.
                                      If str, extract single ID (ex: "2.1")
                                      If list, extract multiple IDs (ex: ["2.1", "3.1"])
        
    Returns:
        list: List of tuples (anion_id, solvent, label, pillarplex_id) 
              where label in {0:'no', 1:'slightly', 2:'yes'}
        
    Example:
        triplets = parse_solubility_csv('pillarplex_salts_solubility.csv', pillarplex_ids=['2.1', '3.1'])
        # [(1, 'Acetone', 0, '2.1'), (1, 'H2O', 0, '2.1'), ...]
    """
    # Normalize pillarplex_ids input
    if pillarplex_ids is None:
        pillarplex_ids = None  # Extract all
    elif isinstance(pillarplex_ids, str):
        pillarplex_ids = [pillarplex_ids]
    
    # Mapping label text -> numeric
    label_map = {'no': 0, 'slightly': 1, 'yes': 2}
    
    # Read CSV file
    df = pd.read_csv(csv_path)
    
    # Find all Pillarplex sections
    all_triplets = []
    
    for idx, row in df.iterrows():
        cell_value = str(row.iloc[0]).strip() if not pd.isna(row.iloc[0]) else ""
        
        # Check if this is a Pillarplex section header (format: "X.Y;Solvent:;...")
        if ';Solvent:' in cell_value:
            current_pillarplex_id = cell_value.split(';')[0].strip()
            
            # Filter by requested IDs if specified
            if pillarplex_ids is not None and current_pillarplex_id not in pillarplex_ids:
                continue
            
            # Extract solvent names (next line)
            solvents_row = idx + 1
            if solvents_row >= df.shape[0]:
                continue
            
            solvents = []
            for col_idx in range(1, df.shape[1]):
                val = str(df.iloc[solvents_row, col_idx]).strip()
                if val and val != 'nan':
                    solvents.append(val)
            
            # Extract solubility data (following lines)
            data_start_row = idx + 2
            
            for row_idx in range(data_start_row, df.shape[0]):
                anion_id_cell = str(df.iloc[row_idx, 0]).strip()
                
                # Stop if we reach a new section
                if ';Solvent:' in anion_id_cell or (anion_id_cell and 
                    not anion_id_cell.replace('.', '').replace(',', '').isdigit()):
                    break
                
                # Parse anion ID
                try:
                    anion_id = int(anion_id_cell)
                except ValueError:
                    continue
                
                # Extract labels for each solvent
                for col_idx, solvant in enumerate(solvents):
                    cell_value = str(df.iloc[row_idx, col_idx + 1]).strip().lower()
                    
                    # Ignore missing data
                    if cell_value == 'nan' or cell_value == '-':
                        continue
                    
                    # Convert label
                    if cell_value in label_map:
                        label = label_map[cell_value]
                        all_triplets.append((anion_id, solvant, label, current_pillarplex_id))
    
    if not all_triplets and pillarplex_ids is not None:
        raise ValueError(f"No data found for Pillarplex IDs: {pillarplex_ids}")
    
    return all_triplets


def create_dataset_from_solubility(
    solubility_csv_path,
    anions_dir,
    ligands_dict,
    solvents_dir,
    pillarplex_ids=None,
    current_dir=None
):
    """
    Create a complete dataset (graphs + labels) from the solubility CSV.
    Supports multiple Pillarplex IDs with their corresponding ligand files.
    
    Args:
        solubility_csv_path (str): Path to pillarplex_salts_solubility.csv
        anions_dir (str): Directory containing anion files (*.csv)
        ligands_dict (dict or str): Mapping of pillarplex_id -> ligand_file_path
                                    If str, use same ligand for all Pillarplex IDs
                                    If dict: ex: {"2.1": "path/to/Ag_Pillarplex-Br.xyz", 
                                                  "3.1": "path/to/Au_Pillarplex-Br.xyz"}
        solvents_dir (str): Directory containing solvent files (*.csv)
        pillarplex_ids (list or str): Pillarplex ID(s) to use. 
                                      If None, use all available in CSV
        current_dir (str): Current directory (default os.getcwd())
        
    Returns:
        tuple: (triplets_data, labels) where:
            - triplets_data : list of tuples (anion_graph, ligand_graph, solvent_graph)
            - labels : list of labels (0, 1, 2)
    """
    if current_dir is None:
        current_dir = os.getcwd()
    
    # Parse solubility data
    solubility_data = parse_solubility_csv(solubility_csv_path, pillarplex_ids)
    
    # Handle ligands_dict normalization
    if isinstance(ligands_dict, str):
        # Same ligand for all Pillarplex IDs
        all_pillarplex_ids = set([data[3] for data in solubility_data])
        ligands_dict = {pid: ligands_dict for pid in all_pillarplex_ids}
    
    # Cache for ligands, anions, solvents
    ligand_cache = {}
    anion_cache = {}
    solvent_cache = {}
    
    triplets_data = []
    labels = []
    
    for anion_id, solvant, label, pillarplex_id in solubility_data:
        # Load ligand (with cache)
        if pillarplex_id not in ligand_cache:
            if pillarplex_id not in ligands_dict:
                print(f"Warning: No ligand file specified for Pillarplex {pillarplex_id}, skipping.")
                continue
            
            ligand_file = ligands_dict[pillarplex_id]
            ligand_path = os.path.join(current_dir, ligand_file)
            
            if os.path.exists(ligand_path):
                atoms_ligand, pos_ligand = extract_last_snapshot(ligand_path)
                ligand_cache[pillarplex_id] = get_graph(atoms_ligand, pos_ligand)
            else:
                print(f"Warning: Ligand file {ligand_path} not found, skipping Pillarplex {pillarplex_id}.")
                continue
        
        # Load anion (with cache)
        anion_key = f"anion_{anion_id}"
        if anion_key not in anion_cache:
            anion_filename = f"{anion_id}.csv"
            anion_path = os.path.join(current_dir, anions_dir, anion_filename)
            
            if not os.path.exists(anion_path):
                # Try other common naming conventions
                anion_path = None
                for fname in os.listdir(os.path.join(current_dir, anions_dir)):
                    if fname.startswith(str(anion_id)) and fname.endswith('.csv'):
                        anion_path = os.path.join(current_dir, anions_dir, fname)
                        break
            
            if anion_path and os.path.exists(anion_path):
                atoms_anion, pos_anion = extract_elements_and_positions(anion_path)
                anion_cache[anion_key] = get_graph(atoms_anion, pos_anion)
            else:
                print(f"Warning: Anion {anion_id} not found, skipping.")
                continue
        
        # Load solvent (with cache)
        solvent_key = f"solvent_{solvant}"
        if solvent_key not in solvent_cache:
            solvent_filename = f"{solvant}.csv"
            solvent_path = os.path.join(current_dir, solvents_dir, solvent_filename)
            
            if os.path.exists(solvent_path):
                atoms_solvent, pos_solvent = extract_elements_and_positions(solvent_path)
                solvent_cache[solvent_key] = get_graph(atoms_solvent, pos_solvent)
            else:
                print(f"Warning: Solvent {solvant} not found, skipping.")
                continue
        
        # Add triplet and label
        triplet = (anion_cache[anion_key], ligand_cache[pillarplex_id], solvent_cache[solvent_key])
        triplets_data.append(triplet)
        labels.append(label)
    
    return triplets_data, labels