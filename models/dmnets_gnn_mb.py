import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GAT


class DMNETS_GNN_MB(nn.Module):
    def __init__(self, args):
        super(DMNETS_GNN_MB, self).__init__()
        self.model_name = 'DMNETS_GNN_MB'
        self.args = args
        self.model_args = args.model
        self.loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.n = self.args.dataset.max_n

        if hasattr(args.experiment.train, 'label_smoothing'):
            self.label_smoothing = args.experiment.train.label_smoothing
        else:
            self.label_smoothing = 0
        if hasattr(args.experiment.train, 'prev_edge_smoothing'):
            self.prev_edge_smoothing = args.experiment.train.prev_edge_smoothing
        else:
            self.prev_edge_smoothing = 0

        # Initialise the Encoder
        self.hidden_size = self.model_args.hidden_size
        enc_args = self.model_args.encoder

        self.encoder = GAT(self.n, self.hidden_size,
                           num_layers=enc_args.num_layers,
                           heads=enc_args.heads,
                           dropout=enc_args.dropout,)

        dec_args = self.model_args.decoder
        dec_args.hidden_size = self.hidden_size

        self.decoder = GAT(self.hidden_size, self.hidden_size,
                           num_layers=dec_args.num_layers,
                           heads=dec_args.heads,
                           dropout=dec_args.dropout,)

        self.num_mix_comp = args.model.num_mix_comp

        self.output_theta_1 = nn.Sequential(
            nn.Dropout(self.model_args.output_theta.dropout),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.num_mix_comp))

        self.output_theta_2 = nn.Sequential(
            nn.Dropout(self.model_args.output_theta.dropout),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.num_mix_comp))

        self.output_alpha = nn.Sequential(
            nn.Dropout(self.model_args.output_alpha.dropout),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.num_mix_comp))


    def forward(self, data):
        edges_x = data['edges_x'] if 'edges_x' in data else None
        node_feat = data['node_feat'] if 'node_feat' in data else None
        edges_y = data['edges_y'] if 'edges_y' in data else None
        subgraph_idx = data['subgraph_idx'] if 'subgraph_idx' in data else None
        diffs_idx = data['diffs_idx'] if 'diffs_idx' in data else None
        labels = data['labels'] if 'labels' in data else None
        prev_edges = data['prev_edges'] if 'prev_edges' in data else None
        node_feat_idx = data['node_feat_idx'] if 'node_feat_idx' in data else None
        batch_size = data['batch_size'] if 'batch_size' in data else None
        is_sampling = data['is_sampling'] if 'is_sampling' in data else None
        num_steps = data['num_steps'] if 'num_steps' in data else None
        # adj = data['adj'] if 'adj' in data else None


        if not is_sampling:
            if self.training:
                labels = labels * (1 - self.label_smoothing) + (0.5 * self.label_smoothing)
                prev_edges = prev_edges * (1 - self.prev_edge_smoothing) + (0.5 * self.prev_edge_smoothing)
            alpha_logits, theta_logits = self._inference(node_feat, edges_x,
                                                         node_feat_idx, edges_y,
                                                         diffs_idx, prev_edges)

            return compute_loss(alpha_logits, theta_logits, labels, self.loss_fn, subgraph_idx, self.n)

        else:
            return self.recursive_forward(node_feat, edges_x, prev_edges, batch_size, num_steps)

    def _inference(self, node_feat, edges_x,
                   node_feat_idx, edges_y,
                   diffs_idx, prev_edges):
        # TODO: diffs_idx only needs to be 1-d now, no longer use 2nd dim
        encoder_outputs = self.encoder(node_feat, edges_x)
        decoder_outputs = self.decoder(encoder_outputs[node_feat_idx], edges_y)
        diffs = encoder_outputs[diffs_idx[:, 0]] - decoder_outputs
        alpha_logits = self.output_alpha(diffs)
        theta_logits_1 = self.output_theta_1(diffs)
        theta_logits_2 = self.output_theta_2(diffs)
        theta_logits = theta_logits_1 * prev_edges[:, None] + theta_logits_2 * (1-prev_edges)[:, None]
        return alpha_logits, theta_logits

    def _sampling(self, node_feat, x_edges, prev_edges, B):
        n = node_feat.shape[1]
        encoder_outputs = self.encoder(node_feat, x_edges)
        A = torch.zeros((B, n, n)).to(node_feat.device)
        for i in range(1, n):
            adj = A[:, :i, :i]
            adj = adj + adj.transpose(1, 2)
            edges = [
                adj[bb].to_sparse().coalesce().indices() + bb * adj.shape[1]
                for bb in range(B)
            ]
            edges = torch.cat(edges, dim=1)
            node_feat_idx = np.concatenate([np.arange(i) + b * n for b in range(B)])
            decoder_outputs = self.decoder(encoder_outputs[node_feat_idx], edges)
            # Compute the diffs
            idx_col = [i + b * n for b in range(B) for _ in range(i)]
            diffs = encoder_outputs[idx_col] - decoder_outputs#[node_feat_idx]
            alpha_logits = self.output_alpha(diffs).view(B, -1, self.num_mix_comp)
            theta_logits_1 = self.output_theta_1(diffs)
            theta_logits_2 = self.output_theta_2(diffs)
            # Get the prev_edges for the ith row
            edge_ix = sum(np.arange(i+1, dtype=np.uint8))
            prev_edges_i = torch.flatten(prev_edges[:, edge_ix - i:edge_ix])
            theta_logits = theta_logits_1 * prev_edges_i[:, None] + theta_logits_2 * (1 - prev_edges_i[:, None])
            theta_logits = theta_logits.view(B, -1, self.num_mix_comp)

            A[:, i:i + 1, :i] = sample_row_mb(alpha_logits.mean(dim=1), theta_logits)
        A = torch.tril(A, diagonal=-1)
        A = A + A.transpose(1, 2)
        return A

    def recursive_forward(self, node_feat, x_edges, prev_edges, batch_size, num_steps, to_cpu=False):
        ' Returns a list with T sampled transitions of the dynamic network from x.'
        out_ts = []
        for t in range(num_steps):
            adj = self._sampling(node_feat, x_edges, prev_edges, batch_size)
            x_edges = [
                adj[bb].to_sparse().coalesce().indices().long() + bb * adj.shape[1]
                for bb in range(batch_size)
            ]
            x_edges = torch.cat(x_edges, dim=1)
            idx = np.array([[i, j] for i in range(1, adj.shape[1]) for j in range(i)])
            prev_edges = adj[:, idx[:, 0], idx[:, 1]]
            adj = adj.cpu() if to_cpu else adj
            out_ts.append(adj)
        return out_ts


def compute_loss(alpha_logits, theta_logits, labels, loss_fn,
                 subgraph_idx, num_nodes, normalise_rows=False):
    num_rows = num_nodes - 1
    K = alpha_logits.shape[1]
    num_subgraph = subgraph_idx[-1] + 1
    # Compute the loss per edge
    adj_loss = torch.stack([loss_fn(theta_logits[:, kk], labels) for kk in range(K)], dim=1)
    subgraph_k_idx = subgraph_idx.unsqueeze(1).expand(-1, K)
    reduce_adj_loss = torch.zeros(num_subgraph, K).to(labels.device)
    reduce_adj_loss = reduce_adj_loss.scatter_add(0, subgraph_k_idx, adj_loss)
    if normalise_rows:
        adj_const = torch.zeros_like(reduce_adj_loss)
        adj_const = adj_const.scatter_add(0, subgraph_k_idx, torch.ones_like(adj_loss))
        reduce_adj_loss /= adj_const

    const = torch.zeros(num_subgraph).to(labels.device)
    const = const.scatter_add(0, subgraph_idx, torch.ones_like(subgraph_idx).float())
    reduce_alpha = torch.zeros((num_subgraph, K)).to(labels.device)
    reduce_alpha = reduce_alpha.scatter_add(0, subgraph_k_idx, alpha_logits)
    reduce_alpha /= const.view(-1, 1)
    # TODO: normalise by const
    reduce_alpha = F.log_softmax(reduce_alpha, -1) # Log alpha probabilities for each edge

    log_prob = -reduce_adj_loss + reduce_alpha
    log_prob = torch.logsumexp(log_prob, dim=1)  # Loss per row for all batches
    log_prob = log_prob.view(-1, num_rows)
    loss = log_prob.sum(dim=1).mean()
    return -loss


def sample_row_mb(log_alpha, log_theta):
    B = log_alpha.shape[0]
    prob_alpha = F.softmax(log_alpha, -1)  # Sum to 1 in the K dimension.
    # Below tells you which component in the mixture to use for each batch
    alpha = torch.multinomial(prob_alpha, 1).squeeze(dim=1)
    prob = []
    # TODO: consider vectorizing this if possible? - we will loop it on TF episodes
    for b in range(B):
        prob += [torch.sigmoid(log_theta[b, :, alpha[b]])]
    prob = torch.stack(prob, dim=0)
    res = torch.bernoulli(prob).unsqueeze(dim=1)
    return res

