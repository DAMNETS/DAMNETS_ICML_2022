import torch
import networkx as nx
import os
import pickle
from tqdm import tqdm
from torch.utils.data.dataloader import default_collate


class TFTestSampler(torch.utils.data.Dataset):
    def __init__(self, ts_list, args, is_multistep=False, tag='test'):
        self.T = len(ts_list[0])
        self.N = len(ts_list)
        graphs_flatten = [G for ts in ts_list for G in ts]
        self.max_n = max([G.number_of_nodes() for G in graphs_flatten])
        if not hasattr(args.dataset, 'max_n'):
            args.dataset.max_n = self.max_n

        self.file_names = []
        data_cache = os.path.join(args.save_dir, 'data_cache')
        if not os.path.isdir(data_cache):
            os.makedirs(data_cache)
        print(f'Processing {tag} data')
        ts_ix = 0
        for b in tqdm(range(self.N)):
            ts_batch = []
            n_timesteps = 1 if not is_multistep else self.T - 1
            for kk in range(n_timesteps):
                x = nx.to_numpy_array(ts_list[b][kk])
                # Add a row of all ones to the first row of y (so we don't have to do it on the hot path)
                data = {'adj': torch.tensor(x, dtype=torch.float32)}
                ts_batch.append(data)
            path = os.path.join(data_cache, f'{tag}_{b}.pkl')
            pickle.dump(data, open(path, 'wb'))
            self.file_names.append(path)

        print('Dataset length: ', len(self.file_names))

    def _collate_slice(self, batch):
        data = default_collate(batch)
        data['is_sampling'] = True
        return data

    def collate_fn(self, batch):
        return [self._collate_slice([bb[kk] for bb in batch]) for kk in range(len(batch[0]))]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        return pickle.load(open(self.file_names[idx], 'rb'))
