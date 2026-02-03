import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.nn import NNConv, global_mean_pool, global_max_pool
import gpytorch

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)

class SingleGNN(nn.Module):
    """
    GNN qui traite un (batch de) graphe(s) et retourne une représentation poolée.
    Conçu pour être instancié trois fois (anion, ligand, solvent).
    """
    def __init__(self, nodefeat_num=3, edgefeat_num=2, nodeembed_to=64, edgeembed_to=32):
        super().__init__()
        self._node_embedding = nn.Sequential(
            nn.Linear(nodefeat_num, nodeembed_to),
            nn.ReLU()
        )
        self._node_embednorm = nn.BatchNorm1d(nodeembed_to)
        self._edge_embedding = nn.Sequential(
            nn.Linear(edgefeat_num, edgeembed_to),
            nn.ReLU()
        )
        self._edge_embednorm = nn.BatchNorm1d(edgeembed_to)

        self._first_conv = NNConv(
            nodeembed_to,
            nodeembed_to,
            nn.Sequential(nn.Linear(edgeembed_to, nodeembed_to**2), nn.Tanh()),
            aggr='mean'
        )
        self._first_conv_bn = nn.BatchNorm1d(nodeembed_to)

        self._second_conv = NNConv(
            nodeembed_to,
            nodeembed_to,
            nn.Sequential(nn.Linear(edgeembed_to, nodeembed_to**2), nn.ReLU()),
            aggr='mean'
        )
        self._second_conv_bn = nn.BatchNorm1d(nodeembed_to)

        # pooling methods (mean + max)
        self._pooling = [global_mean_pool, global_max_pool]
        self.output_size = 2 * nodeembed_to

    def forward(self, graph: Batch):
        # graph can be a torch_geometric.data.Data or Batch
        x = graph.x.float()
        edge_index = graph.edge_index
        edge_attr = graph.edge_attr.float()
        batch_vec = getattr(graph, "batch", torch.zeros(x.size(0), dtype=torch.long, device=x.device))

        # embeddings
        x = self._node_embedding(x)
        # BatchNorm expects shape [N, F] but BatchNorm1d takes (B, F) where B = N here
        x = self._node_embednorm(x)
        edge_attr = self._edge_embedding(edge_attr)
        edge_attr = self._edge_embednorm(edge_attr)

        # graph convolutions
        x = self._first_conv(x, edge_index, edge_attr)
        x = self._first_conv_bn(x)
        x = self._second_conv(x, edge_index, edge_attr)
        x = self._second_conv_bn(x)

        # global pooling (for batched graphs)
        pooled = torch.cat([p(x, batch_vec) for p in self._pooling], dim=1)  # shape [batch_size, 2*nodeembed_to]
        return pooled

class CombinedModel(nn.Module):
    """
    Modèle combinant trois SingleGNN (anion, ligand, solvent) puis un NN sur la concaténation.
    Retourne des logits pour classification (num_classes).
    """
    def __init__(self, nodefeat_num=3, edgefeat_num=2, nodeembed_to=64, edgeembed_to=32, num_classes=3):
        super().__init__()
        self.anion_gnn = SingleGNN(nodefeat_num, edgefeat_num, nodeembed_to, edgeembed_to)
        self.ligand_gnn = SingleGNN(nodefeat_num, edgefeat_num, nodeembed_to, edgeembed_to)
        self.solvent_gnn = SingleGNN(nodefeat_num, edgefeat_num, nodeembed_to, edgeembed_to)

        total_feat = self.anion_gnn.output_size + self.ligand_gnn.output_size + self.solvent_gnn.output_size

        self.predictor = nn.Sequential(
            nn.Linear(total_feat, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)  # logits
        )
        self.predictor.apply(init_weights)

    def forward(self, anion_graph: Batch, ligand_graph: Batch, solvent_graph: Batch):
        a_repr = self.anion_gnn(anion_graph)
        l_repr = self.ligand_gnn(ligand_graph)
        s_repr = self.solvent_gnn(solvent_graph)
        combined = torch.cat([a_repr, l_repr, s_repr], dim=1)
        logits = self.predictor(combined)
        return logits

# backward-compatible alias
BaseGNN = CombinedModel

class FeatureExtractor(nn.Module):
    """
    Extrait les représentations intermédiaires des trois GNNs sans passer par le NN final.
    Utile pour l'analyse des features apprises.
    """
    def __init__(self, base_gnn: CombinedModel):
        super().__init__()
        self.anion_gnn = base_gnn.anion_gnn
        self.ligand_gnn = base_gnn.ligand_gnn
        self.solvent_gnn = base_gnn.solvent_gnn

    def forward(self, anion_graph: Batch, ligand_graph: Batch, solvent_graph: Batch):
        a_repr = self.anion_gnn(anion_graph)
        l_repr = self.ligand_gnn(ligand_graph)
        s_repr = self.solvent_gnn(solvent_graph)
        combined = torch.cat([a_repr, l_repr, s_repr], dim=1)
        return combined
    
class ExactGPLayer(gpytorch.models.ExactGP):
    """
    Modèle GP exact utilisant les features extraites par FeatureExtractor.
    """
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPLayer, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
class ExactGNNGP(nn.Module):
    """
    Modèle combinant un extracteur de features GNN et une couche GP exacte.
    """

    def __init__(self, feature_extractor, train_x, train_y):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.gp_layer = ExactGPLayer(train_x, train_y, self.likelihood)
    
    def forward(self, anion_graph, ligand_graph, solvent_graph):
        features = self.feature_extractor(anion_graph, ligand_graph, solvent_graph)
        return self.gp_layer(features)
    
    def predict(self, anion_graph, ligand_graph, solvent_graph):
        self.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            gp_dist = self.forward(anion_graph, ligand_graph, solvent_graph)
            pred_dist = self.likelihood(gp_dist)
            
            mean = pred_dist.mean
            
            lower, upper = pred_dist.confidence_region()

            
        return mean, lower, upper