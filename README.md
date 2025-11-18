# -Mof_solubility_prediction
Our goal is to predict this.  We will essentially encode the anion, solvent and ligand as graphs. Then we will process each of them with a separate GNN. We will concatenate the results in a tensor. We will use this tensor as input to a neural network and we will try to predict if it will dissolve or not.
