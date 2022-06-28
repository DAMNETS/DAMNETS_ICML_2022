import torch
import numpy as np
import networkx as nx
import os
import pickle
from tqdm import tqdm


class GNNTestSampler(torch.utils.data.Dataset):
    def __init__(self, ts_list, args, tag='test'):
        self.T = len(ts_list[0])
        self.N = len(ts_list)
        self.batch_size = min(args.experiment.test.batch_size, self.N)
        graphs_flatten = [G for ts in ts_list for G in ts]
        self.max_n = max([G.number_of_nodes() for G in graphs_flatten])
        if not hasattr(args.dataset, 'max_n'):
            args.dataset.max_n = self.max_n

        self.file_names = []
        data_cache = os.path.join(args.save_dir, 'data_cache')
        if not os.path.isdir(data_cache):
            os.makedirs(data_cache)
        print(f'Processing {tag} data')
        pbar = tqdm(total=self.N)
        for bb in range(self.N):
            ts_batch = []
            for t in range(self.T - 1):
                G_0 = ts_list[bb][t]
                A = nx.to_numpy_array(G_0)
                data = {'adj': A}
                ts_batch.append(data)
            path = os.path.join(data_cache, f'{tag}_{bb}.pkl')
            pickle.dump(ts_batch, open(path, 'wb'))
            self.file_names.append(path)
            pbar.update(1)
        print('Dataset length: ', len(self.file_names))

    def collate_sampling(self, batch, t):
        data = {}
        data['adj'] = torch.from_numpy(
            np.stack([bb['adj'] for bb in batch])).float()
        data['is_sampling'] = True
        data['batch_size'] = len(batch)
        return data

    def collate_fn(self, batch):
        # Batch is lists of NTS
        return [self.collate_sampling([bb[t] for bb in batch], t=t) for t in range(len(batch[0]))]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        return pickle.load(open(self.file_names[idx], 'rb'))
