import os
import argparse
import numpy as np
import torch
import random
import yaml
import pickle as pkl
from easydict import EasyDict as edict
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import multiprocessing as mp

from utils.graph_generators import *
from utils.arg_helper import mkdir
from utils.graph_utils import save_graph_list, load_graph_ts
import gc
from torch_geometric.utils.convert import from_networkx


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Script to generate the synthetic data for the DAMNETS experiments")
    parser.add_argument(
        '-n',
        '--num_workers',
        type=int,
        default=1,
        help='Number of workers used to generate the data'
    )
    parser.add_argument(
        '-p',
        '--data_path',
        type=str,
        default='data',
        help='The data directory.'
    )
    parser.add_argument(
        '-d',
        '--dataset_name',
        type=str,
        default='',
        help='The name of the pickled dataset in the directory, stored as lists of networkx graphs. '
             'Do not include .pkl.'
    )
    parser.add_argument(
        '-t',
        '--train_p',
        type=float,
        help='The proportion of total data to use for training. The remainder will be used for test.'
    )
    parser.add_argument(
        '-v',
        '--val_p',
        type=float,
        help='The proportion of the training data to use for validation.'
    )
    parser.add_argument(
        '-r',
        '--randomize',
        type=bool,
        help='If True, will randomly select the time series to use for training/test/validation.',
        default=True
    )
    args = parser.parse_args()
    return args


def format_data(graph_ts, n_workers, start_idx=0):
    '''
    Prepare the data. This involves computing the delta matrices for the
    network time series, inserting them into the TreeLib and then putting them into torch geometric
    format for computation.
    Args:
        graph_ts: The time series to pre-process.
        start_idx: If some time-series have already been processed, we want to start indexing
        from the index of the last one in the previous batch (as they are accessed
        via this index in the TreeLib underlying the bigg model). This is used for training and validation split.
    Returns: A pytorch dataloader that loads the previous network combined with the index in the TreeLib of the
    associated delta matrix we want to learn.
    '''
    print('Computing Deltas')
    diffs = process_map(compute_adj_delta, graph_ts, max_workers=n_workers)
    graph_ts = [ts[:-1] for ts in graph_ts]
    # Flatten (will go into treelib in this order).
    diffs = [nx.Graph(diff) for diff_ts in diffs for diff in diff_ts]
    graph_ts = [g for ts in graph_ts for g in ts]
    num_nodes = graph_ts[0].number_of_nodes()
    print('Converting to networkx format')
    data = process_map(from_networkx, graph_ts, max_workers=n_workers, chunksize=20)
    # data = [from_networkx(g) for g in tqdm(graph_ts)]  # convert to torch_geometric format w/ edgelists.
    one_hot = torch.eye(num_nodes)
    print('Setting attributes.')
    for i in tqdm(range(len(data))):
        data[i].x = one_hot
        data[i].graph_id = i + start_idx
    return (data, diffs), len(diffs)

def main():
    c_args = parse_arguments()
    save_dir = c_args.data_path
    mkdir(save_dir)

    graphs_path = os.path.join(save_dir, f'{c_args.dataset_name}.pkl')
    graphs = load_graph_ts(graphs_path)
    if c_args.randomize:
        print('Randomizing')
        random.shuffle(graphs)
    train_ix = int(len(graphs) * c_args.train_p)
    if train_ix == 0:  # debugging
        train_graphs = test_graphs = graphs
    else:
        train_graphs = graphs[:train_ix]
        test_graphs = graphs[train_ix:]
    val_len = int(len(train_graphs) * c_args.val_p)
    val_graphs = train_graphs[:val_len] if val_len > 0 else train_graphs[val_len:]
    # Remove validation from training set
    train_graphs = train_graphs[val_len:]
    ## Set number of nodes for bigg model (keep names same for compatability)
    # self.model_args.bigg.max_num_nodes = nx.number_of_nodes(self.train_graphs[0][0])
    print('Number of Training TS: ', len(train_graphs))
    print('Number of Val TS: ', len(val_graphs))
    print('Number of Test TS: ', len(test_graphs))
    print('TS length (T): ', len(test_graphs[0]))
    print('Saving Graphs')
    save_graph_list(
        train_graphs, os.path.join(save_dir, f'{c_args.dataset_name}_train_graphs_raw.pkl'))
    save_graph_list(
        val_graphs, os.path.join(save_dir, f'{c_args.dataset_name}_val_graphs_raw.pkl'))
    save_graph_list(
        test_graphs, os.path.join(save_dir, f'{c_args.dataset_name}_test_graphs.pkl'))
    ## test doesn't need any pre-processing

    del test_graphs
    del graphs
    gc.collect()
    ## put into required format, save.
    train_processed, idx = format_data(train_graphs, c_args.num_workers)
    save_graph_list(
        train_processed, os.path.join(save_dir, f'{c_args.dataset_name}_train_graphs.pkl'))
    del train_processed
    del train_graphs
    gc.collect()

    val_processed, _ = format_data(val_graphs, c_args.num_workers, idx)
    save_graph_list(
        val_processed, os.path.join(save_dir, f'{c_args.dataset_name}_val_graphs.pkl'))

    # args.experiment.graph_save_dir = self.args.save_dir


if __name__ == '__main__':
    main()