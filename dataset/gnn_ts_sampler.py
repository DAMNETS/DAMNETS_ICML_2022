import os
import pickle
from tqdm import tqdm

import torch
import numpy as np
import networkx as nx


class GNNTSampler(torch.utils.data.Dataset):
    def __init__(self, ts_list, args, tag='train'):
        self.args = args
        self.T = len(ts_list[0])
        self.N = len(ts_list)
        graphs_flatten = [G for ts in ts_list for G in ts]
        self.max_n = max([G.number_of_nodes() for G in graphs_flatten])
        if not hasattr(args.dataset, 'max_n'):
            args.dataset.max_n = self.max_n
        self.ts_list = ts_list
        data_cache = os.path.join(args.save_dir, 'data_cache')
        if not os.path.isdir(data_cache):
            os.makedirs(data_cache)

        self.file_names = []
        print(f'Processing {tag} data.')
        pbar = tqdm(total = self.N * (self.T - 1))
        ix = 0
        for b in range(self.N):
            for t in range(self.T-1):
                x = nx.to_numpy_array(ts_list[b][t])
                y = nx.to_numpy_array(ts_list[b][t+1])
                data = self.process_pair(x, y)
                path = os.path.join(data_cache, f'{tag}_{ix}.pkl')
                pickle.dump(data, open(path, 'wb'))
                self.file_names.append(path)
                ix += 1
                pbar.update(1)
        print('Dataset length: ', len(self.file_names))

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        return pickle.load(open(self.file_names[idx], 'rb'))

    def process_pair(self, A_1, A_2):
        n = A_1.shape[0]
        edges_x = torch.from_numpy(A_1).to_sparse()
        edges_x = edges_x.coalesce().indices().long()
        edges_y = []
        labels = []
        diffs_idx = []
        subgraph_idx = []
        node_feat_idx = []
        node_feat = np.diag(np.ones(n))
        subgraph_count = 1
        subgraph_size = []
        prev_edges = []

        for i in range(1, n):
            # Get the lower triangle, add the new node, connect it up to existing subgraph.
            adj_block = A_2[:i, :i]
            adj_block += adj_block.transpose()
            # Get the edges for the subgraph
            edge_idx = torch.from_numpy(adj_block).to_sparse()
            edges_y += [edge_idx.coalesce().indices().long()]
            subgraph_size += [i]  # Size of first subgraph is 1 node. Grows one at a time.
            idx_row_gnn, idx_col_gnn = np.meshgrid(
                (np.ones(1) * i).astype(np.int8), np.arange(i))
            idx_row_gnn = idx_row_gnn.reshape(-1, 1)
            idx_col_gnn = idx_col_gnn.reshape(-1, 1)
            diffs_idx += [
                np.concatenate([idx_row_gnn, idx_col_gnn], axis=1)
            ]
            labels += [
                A_2[idx_row_gnn, idx_col_gnn].flatten().astype(np.uint8)
            ]
            # TODO: make this sparse matrix?? Will probably be faster in most cases and save lot of VRAM
            prev_edges += [
                A_1[idx_row_gnn, idx_col_gnn].flatten().astype(np.uint8)
            ]
            subgraph_idx += [np.ones_like(labels[-1]).astype(np.int64) * i - 1]
            node_feat_idx += [np.arange(i)]  # (0, 1, ..., i-1)
            subgraph_count += 1
        cum_size = np.cumsum([0] + subgraph_size)
        for i in range(len(edges_y)):
            edges_y[i] = edges_y[i] + cum_size[i]
            diffs_idx[i][:, 1] += cum_size[i]
        data = {'edges_x': edges_x,
                'edges_y': torch.cat(edges_y, dim=1),
                'node_feat': node_feat,
                'subgraph_idx': np.concatenate(subgraph_idx),
                'diffs_idx': np.concatenate(diffs_idx),
                'labels': np.concatenate(labels),
                'prev_edges': np.concatenate(prev_edges),
                'node_feat_idx': np.concatenate(node_feat_idx),
                'total_subgraph_incr': sum(subgraph_size)}
        return data

    def collate_fn(self, batch):
        '''
        This function collates a batch of graph pairs (G_t, G_t+1).
        It stacks all the adjacency matrices into one large, block diagonal matrix and increments
        all the edge indices accordingly (for the GNN), as well as incrementing the required indexing objects.
        '''
        # If you're debugging this and looking at a 'skip' in indices,
        # this is often intentional as the first subgraph has 1 node with
        # no edges, so the index skips there.
        n = batch[0]['node_feat'].shape[0]
        # Need to increment node base for edges
        idx_base = np.array([0] + [bb['total_subgraph_incr'] for bb in batch])
        idx_base = np.cumsum(idx_base)
        data = {}
        data['edges_x'] = torch.cat(
            [bb['edges_x'] + b * n for b, bb in enumerate(batch)], dim=1
        ).long()
        data['edges_y'] = torch.cat(
            [bb['edges_y'] + idx_base[b] for b, bb in enumerate(batch)], dim=1).long()
        data['node_feat'] = torch.from_numpy(
            np.concatenate([bb['node_feat'] for bb in batch], axis=0)
        ).float()
        data['subgraph_idx'] = torch.from_numpy(
            np.concatenate([bb['subgraph_idx'] + b * (n-1) for b, bb in enumerate(batch)])
        ).long()
        for b, bb in enumerate(batch):
            batch[b]['diffs_idx'][:, 0] += b * n
            # batch[b]['diffs_idx'] += idx_base[b]
        data['diffs_idx'] = torch.from_numpy(
            np.concatenate([bb['diffs_idx'] for b, bb in enumerate(batch)])
        ).long()
        data['labels'] = torch.from_numpy(
            np.concatenate([bb['labels'] for bb in batch])
        ).float()
        data['prev_edges'] = torch.from_numpy(
            np.concatenate([bb['prev_edges'] for bb in batch])
        ).float()
        data['node_feat_idx'] = torch.from_numpy(
            np.concatenate([bb['node_feat_idx'] + b * n for b, bb in enumerate(batch)])
        ).long()
        return data
