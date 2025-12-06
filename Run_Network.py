from gnnClass import *
import os
import numpy as np
import warnings
from sklearn.utils import shuffle
import torch
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader as GeoLoader
import gpytorch
from helpers import *
seed = 42
import torch.nn.functional as F
import random
from read_xyz import *
import torch
import gpytorch
#%%
graph_anions = []
graph_ligand = []
graph_solvent = []
#create a dummy variable that will serve as the prediction target, i.e.
#let us assume that the e_value is the value of the energy of the entire molecule


current_dir = os.getcwd()
datafile_anions='/Users/manocharbonnier/Desktop/Computational Material Design/Projet 1 /mof_solubility/anions/ClO4.csv'
atoms_anions_list,atoms_anions_pos_list=extract_elements_and_positions(datafile_anions)


datafile_ligand='/Users/manocharbonnier/Desktop/Computational Material Design/Projet 1 /mof_solubility/ligand/xyz/Ag_Pillarplex-Br.xyz'
atoms_ligand_list,atom_ligand_pos_list=extract_last_snapshot(datafile_ligand)


datafile_solvent='/Users/manocharbonnier/Desktop/Computational Material Design/Projet 1 /mof_solubility/solvent/Acetone.csv'




atoms_solvent_list,atoms_solvent_pos_list=extract_elements_and_positions(datafile_solvent)

graph_anions_data = get_graph(atoms_anions_list, atoms_anions_pos_list)
graph_ligand_data = get_graph(atoms_ligand_list, atom_ligand_pos_list)
graph_solvent_data = get_graph(atoms_solvent_list, atoms_solvent_pos_list)

graph_triplets = [(graph_anions_data, graph_ligand_data, graph_solvent_data)]

# we call the get_graph function (see helpers.py) to generate the graph from these info

#%%
# we choose wether the computations will be on the gpu or the cpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# we put our data in a dataloader. This is because we might wish to feed them piecemeal 
#into the training process because we might not have the memory to load everything at once
train_loader = GeoLoader(graph, batch_size=1, shuffle=True)

# we define the model as we did with simple NN. The model definition is in gnn.py
gnn_model = BaseGNN(graph[0].x.shape[-1], graph[0].edge_attr.shape[-1],)
#then we put the model on the device the calculations will take place
gnn_model.to(device)
# here we define the optimizer, its properties and loss function
learning_rate = 0.001
optimizer = torch.optim.Adam(gnn_model.parameters(), lr=learning_rate)
criterion = F.mse_loss
layers_pillarplex = [NNConv(
            nodeembed_to, # first, pass the initial size of the nodes
            nodeembed_to, # and their output-size
            nn.Sequential(
                nn.Linear(edgeembed_to, nodeembed_to**2), nn.ReLU()
            ),aggr= 'mean'), NNConv(
            nodeembed_to, # first, pass the initial size of the nodes
            nodeembed_to, # and their output-size
            nn.Sequential(
                nn.Linear(edgeembed_to, nodeembed_to**2), nn.ReLU()
            ),aggr= 'mean']
gnn_model_pillarplex = BaseGNN(blabla, layers_pillarplex)
gnn_model_solvent = BaseGNN(blabla)
gnn_model_anion = BaseGNN(blabla)
NN = NN(blabal)
# %%

## the training loop##
for epoch in range(100):
    #layers like batchnorm behave differently during training or tests.
    #during training it is important to set the model behaviour to .train()
    gnn_model.train()

    #here we go over all the data of the dataloader (we dont split in training and test, this is a toy example)
    for data in train_loader:
        #remember all the data should be on the same device
        data = data.to(device)
        # zeroing the gradients
        optimizer.zero_grad()
        output_solvent = gnn_model_solvent(data.solvent)
        output_pillarplex = gnn_model_pillarplex(data.pillarplex)
        output_anion = gnn_model_anion(data.additive)
        concatenated_vector = torch.cat((output_solvent, output_pillarplex, output_anion))
        prediction = NN(concatenated_vector)
        loss = criterion(prediction, data.y)
        #calculating the loss
        # loss = criterion(gnn_model(data), data.y)
        #calculating the new gradients
        loss.backward()
        #updating the model parameters
        optimizer.step()
        #printing the loss 
        print(loss.item())

# %%
## prediction time##
# same layers that behave different during training behave different when we want to use the 
#network for prediction so we switch them to .eval() to generate predictions
gnn_model.eval()
# we could wrap the data into another loader but there is another way which you see below
gnn_model(Batch.from_data_list(graph).to(device))


# %%
