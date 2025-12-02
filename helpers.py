import numpy as np
from sklearn.metrics import pairwise_distances
import torch 
from torch_geometric.data import Data, Batch
import os
import csv

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
    atom_to_num = {'C': 6, 'O':8, 'Zn':30, 'Pt':78} # atom to atomic number
    atom_to_en = {'C': 2.55, 'O':3.44, 'Zn':1.65, 'Pt':2.28} # atom to electronegativity
    atom_to_r = {'C': 70, 'O':60, 'Zn':135, 'Pt':135} # atom to radius

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