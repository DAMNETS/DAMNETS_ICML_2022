import torch
from bigg.model.tree_model import RecurTreeGen
from torch_geometric.nn import GCN, GAT
import networkx as nx


class DamnetsSigned(torch.nn.Module):
    def __init__(self, model_args):
        super(DamnetsSigned, self).__init__()
        bigg_args = model_args.bigg
        gnn_args = model_args.gnn
        self.gnn = GAT(in_channels=bigg_args.max_num_nodes,
                       hidden_channels=bigg_args.embed_dim,
                       num_layers=gnn_args.num_layers,
                       dropout=gnn_args.dropout,
                       heads=gnn_args.heads,
                       )
        self.decoder = RecurTreeGen(bigg_args)

    def forward_train(self, node_feat, edges, graph_ids, num_nodes):
        gnn_embeds = self.gnn(node_feat, edges)
        ll, states = self.decoder.forward_train(graph_ids, gnn_embeds, num_nodes)
        return -1 * (ll) / num_nodes

    def forward(self, num_nodes, node_feat, edges, get_ll=False, delta_edges=None):
        gnn_embeds = self.gnn(node_feat, edges)
        ll, sampled_edges, row_states = self.decoder(num_nodes, gnn_embeds, edge_list=delta_edges)
        if get_ll:
            return sampled_edges, (ll.item() / num_nodes)
        return sampled_edges
