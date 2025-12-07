import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.nn import NNConv, global_mean_pool, global_max_pool

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