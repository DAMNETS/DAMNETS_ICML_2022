import torch
from bigg.model.tree_model import RecurTreeGen
from torch_geometric.nn import GCN, GAT
import networkx as nx


class DamnetsPosNeg(torch.nn.Module):
    def __init__(self, model_args):
        super(DamnetsPosNeg, self).__init__()
        bigg_args = model_args.bigg
        gnn_args = model_args.gnn
        self.gnn = GAT(in_channels=bigg_args.max_num_nodes,
                       hidden_channels=bigg_args.embed_dim,
                       num_layers=gnn_args.num_layers,
                       dropout=gnn_args.dropout,
                       heads=gnn_args.heads,
                       )
        self.pos_model = RecurTreeGen(bigg_args)
        self.neg_model = RecurTreeGen(bigg_args)

    def forward_train(self, batch, pos_ids, neg_ids, num_nodes):
        gnn_embeds = self.gnn(batch.x, batch.edge_index)
        pos_ll, _ = self.pos_model.forward_train(pos_ids, gnn_embeds, num_nodes)
        neg_ll, _ = self.neg_model.forward_train(neg_ids, gnn_embeds, num_nodes)
        return -1 * (pos_ll + neg_ll) / num_nodes

    def forward(self, num_nodes, edges, node_feat):
        gnn_embeds = self.gnn(node_feat, edges)
        _, pos_edges, _ = self.pos_model(num_nodes, gnn_embeds)
        pos_g = nx.empty_graph(num_nodes)
        # Build the delta matrix
        pos_g.add_edges_from(pos_edges)
        delta_pos = nx.to_numpy_array(pos_g)
        _, neg_edges, _ = self.neg_model(num_nodes, gnn_embeds)
        neg_g = nx.empty_graph(num_nodes)
        neg_g.add_edges_from(neg_edges)
        delta_neg = nx.to_numpy_array(neg_g)
        return delta_pos - delta_neg