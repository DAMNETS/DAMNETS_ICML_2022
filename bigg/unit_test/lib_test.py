# coding=utf-8
# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# pylint: skip-file
from tqdm import tqdm
import torch
import torch.optim as optim

import numpy as np
import random
import networkx as nx
import pickle as pkl
from bigg.common.configs import cmd_args, set_device
from bigg.model.tree_clib.tree_lib import setup_treelib, TreeLib
from bigg.model.tree_model import RecurTreeGen

from utils.graph_generators import compute_adj_delta, n_community_decay_ts, n_community_const_ts
from torch_geometric.utils.convert import from_networkx
from torch_geometric.loader import DataLoader
from torch_geometric.utils.convert import from_networkx
from eval import *
from easydict import EasyDict as edict
from torch.optim.swa_utils import AveragedModel, SWALR
from torch_geometric.nn import GAT
from utils.ba_ts_generator import barabasi_albert_graph_ts


if __name__ == '__main__':
    random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    cmd_args.gpu = 0
    set_device(cmd_args.gpu)
    setup_treelib(cmd_args)
    T = 50
    c_sizes = [75, 75, 75]
    # cmd_args.embed_dim = 1024
    args = edict({'model': {'name': 'treegnn'}, 'dataset': {'c_sizes': c_sizes}, 'save_dir': '.'})

    problem = '3c'
    #### CONSTANT TEST
    # ts_list = [n_community_const_ts(c_sizes, T) for i in range(100)]
    # val_list = [n_community_const_ts(c_sizes, T) for i in range(20)]
    # test_list = [n_community_const_ts(c_sizes, T) for i in range(100)]

    ### SINGLE TIME SERIES TEST
    # ts_list = [n_community_decay_ts(c_sizes, T, p_int=0.9, p_ext=0.05)] * 10
    # val_list = ts_list
    # test_list = ts_list

    ## USUAL
    ts_list = [n_community_decay_ts(c_sizes, T, p_int=0.9, p_ext=0.05, decay_prop=0.05) for i in range(50)]
    val_list = [n_community_decay_ts(c_sizes, T, p_int=0.9, p_ext=0.05, decay_prop=0.05) for i in range(20)]
    test_list = [n_community_decay_ts(c_sizes, T, p_int=0.9, p_ext=0.05, decay_prop=0.05) for i in range(20)]

    # problem = 'ba'
    ## BA
    # ts_list = [barabasi_albert_graph_ts(50, 5) for i in range(50)]
    # val_list = [barabasi_albert_graph_ts(50, 5) for i in range(20)]
    # test_list = [barabasi_albert_graph_ts(50, 5) for i in range(20)]


    # G = n_community_decay_ts(c_sizes, T, p_ext=0.01)
    # ts_list = [G]
    # test_list = [G]
    # test_list = ts_list
    n_samples = 1
    n_epochs = 10
    ## SWA PARAMS
    use_swa = False
    swa_start = 20
    anneal_epochs = 2

    diffs = [compute_adj_delta(ts) for ts in ts_list]
    ts_list = [ts[:-1] for ts in ts_list]

    # Flatten them out for insertion into the treelib.
    diffs = [nx.Graph(diff) for diff_ts in diffs for diff in diff_ts]
    ts_list = [g for ts in ts_list for g in ts]
    # train_graphs = [nx.barabasi_albert_graph(10, 2) for i in range(3)]
    for d in diffs:
        TreeLib.InsertGraph(d)
    num_nodes = ts_list[0].number_of_nodes()
    data = [from_networkx(g) for g in ts_list]
    ## Set node features
    one_hot = torch.eye(num_nodes)
    for i in range(len(data)):
        data[i].x = one_hot
        data[i].graph_id = i
    loader = DataLoader(data, batch_size=32, shuffle=True)
    ## setup validation ##
    val_diffs = [compute_adj_delta(ts) for ts in val_list]
    val_list = [ts[:-1] for ts in val_list]
    val_diffs = [nx.Graph(diff) for diff_ts in val_diffs for diff in diff_ts]
    val_list = [g for ts in val_list for g in ts]
    for d in val_diffs:
        TreeLib.InsertGraph(d)
    data_val = [from_networkx(g) for g in val_list]
    ## Set node features
    for i in range(len(data_val)):
        data_val[i].x = one_hot
        data_val[i].graph_id = i + len(data)
    val_loader = DataLoader(data_val, batch_size=32, shuffle=True)
    #####
    cmd_args.max_num_nodes = num_nodes
    cmd_args.learning_rate = 0.001

    model = RecurTreeGen(cmd_args).to(cmd_args.device)
    optimizer = optim.Adam(model.parameters(), lr=cmd_args.learning_rate, weight_decay=1e-5)
    sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                 patience=100,
                                                 verbose=True,
                                                 min_lr=1e-6,
                                                 cooldown=50)
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=0.0005, anneal_epochs=anneal_epochs)

    for i in range(n_epochs):
        for batch in loader:
            model.train()
            graph_ids = batch.graph_id
            batch.to(cmd_args.device)
            optimizer.zero_grad()
            ll, _ = model.forward_train(graph_ids, batch, num_nodes)
            loss = -ll / num_nodes
            print('iter', i, 'loss', loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cmd_args.grad_clip)
            optimizer.step()
            # sched.step(loss)
        with torch.no_grad():
            for batch in val_loader:
                model.eval()
                graph_ids = batch.graph_id
                batch.to(cmd_args.device)
                ll, _ = model.forward_train(graph_ids, batch, num_nodes)
                loss = -ll / num_nodes
                print('iter', i, 'val_loss', loss.item())

        if i > swa_start and use_swa:
            swa_model.update_parameters(model)
            swa_scheduler.step()

    # Sample the trained model
    # test_list = [n_community_decay_ts(c_sizes, T) for i in range(100)]
    # test_list = [[nx.Graph(g) for g in G] for _ in range(2)]
    sampled_ts_list = []
    model.eval()
    for i in range(n_samples):
        with torch.no_grad():
            for ts in tqdm(test_list):
                samples_ts = [ts[0]]
                for g in ts[:-1]:
                    ## Convert to adjacency for GNN
                    edges = from_networkx(g).edge_index.to(0)
                    node_feat = torch.eye(num_nodes).to(0)
                    if use_swa:
                        _, pred_edges, _ = swa_model(num_nodes, edges, node_feat)
                    else:
                        _, pred_edges, _ = model(num_nodes, edges, node_feat)
                    pred_g = nx.empty_graph(num_nodes)
                    # Build the delta matrix
                    pred_g.add_edges_from(pred_edges)
                    delta = nx.to_numpy_array(pred_g)
                    adj = nx.to_numpy_array(g)
                    adj = (adj + delta) % 2
                    samples_ts.append(nx.Graph(adj))
                sampled_ts_list.append(samples_ts)

    plot_network_statistics(test_list, sampled_ts_list, args, 1)
    if problem == '3c':
        three_comm_plots(sampled_ts_list, test_list, args, 1)
    elif problem == 'ba':
        ba_plots(sampled_ts_list, test_list, args, 1)
