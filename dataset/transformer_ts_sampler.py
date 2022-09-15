import torch
import numpy as np
import networkx as nx
from tqdm import tqdm
import pickle
import os


class TFTSampler(torch.utils.data.Dataset):
    def __init__(self, ts_list, args, tag='train'):
        self.args = args
        self.T = len(ts_list[0])
        self.N = len(ts_list)
        graphs_flatten = [G for ts in ts_list for G in ts]
        self.max_n = max([G.number_of_nodes() for G in graphs_flatten])

        self.data = []

        print(f'Processing {tag} data.')
        pbar = tqdm(total = self.N * (self.T - 1))
        ix = 0
        if self.args.model.ablation:
            for adj, delta in ts_list:
                x = np.tril(nx.to_numpy_array(adj), k=-1)
                y = np.abs(np.tril(nx.to_numpy_array(delta), k=-1))
                labels = []
                for i in range(1, y.shape[0]):
                    labels += [y[i, :i]]
                labels = np.concatenate(labels)
                # Remove the last row of y adjacency (don't need it for forward pass)
                y = y[:-1]
                # Set the first row to be all ones (SOS token for forward pass)
                y[0] = 1
                data = {'x': x, 'y': y, 'y_lab': labels}
                self.data.append(data)
                ix += 1
                pbar.update(1)
        else:
            for b in range(self.N):
                for t in range(self.T - 1):
                    x = np.tril(nx.to_numpy_array(ts_list[b][t]), k=-1)
                    y = np.tril(nx.to_numpy_array(ts_list[b][t + 1]), k=-1)
                    labels = []
                    for i in range(1, y.shape[0]):
                        labels += [y[i, :i]]
                    labels = np.concatenate(labels)
                    # Remove the last row of y adjacency (don't need it for forward pass)
                    y = y[:-1]
                    # Set the first row to be all ones (SOS token for forward pass)
                    y[0] = 1
                    data = {'x': x, 'y': y, 'y_lab': labels}
                    self.data.append(data)
                    ix += 1
                    pbar.update(1)
        print('Dataset length: ', len(self.data))

    def collate_fn(self, batch):
        return {
            'x': torch.stack([to_float_tensor(sample['x']) for sample in batch]),
            'y': torch.stack([to_float_tensor(sample['y']) for sample in batch]),
            'y_lab': torch.stack([to_float_tensor(sample['y_lab']) for sample in batch])
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def to_float_tensor(t):
    return torch.tensor(t, dtype=torch.float32)