import os
import torch
import networkx as nx
from torch.utils.data import DataLoader
import torch.optim as optim
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter
import pickle
from tqdm import tqdm
import time
import multiprocessing as mp

from models import *
from bigg.model.tree_model import RecurTreeGen
from dataset import *
from utils import graph_generators
from utils.train_utils import snapshot_model, load_model
from utils.graph_generators import compute_adj_delta
from utils.gnn_validator import GNNValidator
import utils.graph_utils as graph_utils
from bigg.model.tree_clib.tree_lib import setup_treelib, TreeLib
from bigg.model.tree_model import RecurTreeGen
from models.damnets_pos_neg import DamnetsPosNeg
from torch_geometric.utils.convert import from_networkx
from torch_geometric.loader import DataLoader
import json

class Runner:
    def __init__(self, args, is_test=False):
        self.args = args
        self.model_args = args.model
        self.exp_args = args.experiment
        if not is_test:
            print('Mode: Training | Loading Graphs')
            # train_graphs = graph_utils.load_graph_ts(args.data_file)
            ## Set number of nodes for bigg model (keep names same for compatability)
            # self.model_args.bigg.max_num_nodes = nx.number_of_nodes(self.train_graphs[0][0])
            # print('Number of Training TS: ', len(self.train_graphs))
            # print('Number of Val TS: ', len(self.val_graphs))
            # print('TS length (T): ', len(test_graphs[0]))
            # print('Saving Graphs')
            # graph_utils.save_graph_list(
            #     self.train_graphs, os.path.join(self.args.save_dir, 'train_graphs.pkl'))
            # graph_utils.save_graph_list(
            #     self.val_graphs, os.path.join(self.args.save_dir, 'val_graphs.pkl'))
            # graph_utils.save_graph_list(
            #     test_graphs, os.path.join(self.args.save_dir, 'test_graphs.pkl'))
            # args.experiment.graph_save_dir = self.args.save_dir

            self.train_loss = None
            self.validator = None
            self.writer = SummaryWriter(args.save_dir) if self.exp_args.train.use_writer else None


    def _make_loader(self, graphs, pos_deltas, neg_deltas):
        assert len(pos_deltas) == len(neg_deltas)
        for i in tqdm(range(len(pos_deltas))):
            TreeLib.InsertGraph(pos_deltas[i])
            TreeLib.InsertGraph(neg_deltas[i])
        return DataLoader(graphs,
                          batch_size=self.exp_args.train.batch_size,
                          shuffle=self.exp_args.train.shuffle_data,
                          pin_memory=True)
    def train(self):
        # Create data loader
        dataset = os.path.join(self.args.data_path, self.args.dataset_name)
        train_graphs, train_pos_deltas, train_neg_deltas = graph_utils.load_graph_ts(dataset + '_train_graphs.pkl')
        val_graphs, val_pos_deltas, val_neg_deltas = graph_utils.load_graph_ts(dataset + '_val_graphs.pkl')
        num_nodes = nx.number_of_nodes(train_pos_deltas[0])
        device = self.exp_args.device
        ## set bigg args for compatability
        self.model_args.bigg.gpu = -1 if device == 'cpu' else device
        self.model_args.bigg.device = device
        self.model_args.bigg.seed = self.args.seed
        self.model_args.bigg.max_num_nodes = num_nodes
        setup_treelib(self.model_args.bigg)

        train_loader = self._make_loader(train_graphs, train_pos_deltas, train_neg_deltas)
        val_loader = self._make_loader(val_graphs, val_pos_deltas, val_neg_deltas)
        print('All data prepared. Loading model.')
        model = DamnetsPosNeg(self.model_args).to(device)
        # model_pos = RecurTreeGen(self.model_args).to(device)
        # model_neg = RecurTreeGen(self.model_args).to(device)
        if self.writer is not None:
            self.writer.add_text('Model', str(model))

        print(model)
        # train_loader = self.make_loader(self.train_graphs)
        # val_loader = self.make_loader(self.val_graphs, start_idx=len(self.train_graphs))
        self.validator = GNNValidator(val_loader, model, self.args, self.writer)
        opt = optim.Adam(model.parameters(),
                         lr=self.exp_args.train.lr,
                         weight_decay=self.exp_args.train.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt,
                                                     patience=self.exp_args.validation.patience,
                                                     min_lr=float(self.exp_args.validation.min_lr),
                                                     cooldown=self.exp_args.validation.cooldown,
                                                     factor=self.exp_args.validation.decay_factor,
                                                     verbose=True)
        snapshot_epochs = self.exp_args.train.snapshot_epochs
        results = defaultdict(list)
        iter_count = 0
        train_start = time.time()
        self.train_start = time.time()
        epoch_time = 0
        for epoch in range(self.exp_args.train.epochs):
            start = time.time()
            model.train()
            for batch in train_loader:
                pos_ids = batch.pos_id
                neg_ids = batch.neg_id
                batch.to(device)
                loss = model.forward_train(batch, pos_ids, neg_ids, num_nodes)
                # ll, _ = model.forward_train(batch, pos_ids, neg_ids, num_nodes)
                # loss = -ll / (num_nodes * len(graph_ids))
                loss.backward()
                loss = loss.item()
                if self.writer is not None:
                    self.writer.add_scalar('Loss/Train', loss, iter_count)
                results['train_loss'] += [loss]
                results['train_step'] += [iter_count]
                if (iter_count + 1) % self.args.experiment.train.accum_grad == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.exp_args.train.clip_grad)
                    opt.step()
                    opt.zero_grad()
                if iter_count % self.args.experiment.train.display_iters == 0:
                    print(f'Iteration: {iter_count} | Train Loss: {loss: .3f} | '
                          f'Epoch: {epoch} | '
                          f'Best Validation: {self.validator.best_val_loss: .3f} @ epoch {self.validator.best_epoch} | '
                          f'Last Val Loss: {self.validator.val_losses[-1]: .3f} | '
                          f'Total Train Time: {(time.time() - train_start) / 60: .3f}m | '
                          f'Last Epoch Time: {epoch_time: .3f}s | ')
                iter_count += 1
                      # f'LR: {scheduler.get_last_lr()[0]: .6f}')
            if epoch % self.args.experiment.validation.val_epochs == 0:
                stop = self.validator.validate(epoch)
                scheduler.step(self.validator.val_losses[-1])
                if stop:
                    print('EARLY STOPPING TRIGGERED')
                    break
            if (epoch + 1) % snapshot_epochs == 0:
                snapshot_model(model, epoch, self.args)
            epoch_time = time.time() - start
        pickle.dump(results, open(os.path.join(self.args.save_dir, 'train_stats.pkl'), 'wb'))
        self.save_training_info()

    def save_training_info(self):
        with open('experiment_files/last_train.txt', 'w') as f:
            f.write(self.args.save_dir)
        if self.writer is not None:
            # self.writer.add_text('Loss/Corrections', str(self.validator.correction_epochs))
            end_time = time.time() - self.train_start
            self.writer.add_text('Training Time', f'{end_time / 60: .3f} Minutes')
            self.writer.add_text('Args', pretty_json(self.args))
            self.writer.close()

    def test(self):
        dataset = os.path.join(self.args.data_path, self.args.dataset_name)
        test_list = graph_utils.load_graph_ts(dataset + '_test_graphs.pkl')
        device = self.exp_args.device
        num_nodes = nx.number_of_nodes(test_list[0][0])
        self.model_args.bigg.max_num_nodes = num_nodes
        self.model_args.bigg.gpu = -1 if device == 'cpu' else device
        self.model_args.bigg.device = device
        self.model_args.bigg.seed = self.args.seed
        setup_treelib(self.model_args.bigg)

        model = DamnetsPosNeg(self.model_args).to(device)
        # model = RecurTreeGen(self.model_args).to(device)
        best_markov_file = os.path.join(self.args.model_save_dir,
                                       f'{self.args.experiment.best_val_epoch}.pt')
        print(f'Best Model File : {best_markov_file}')
        load_model(model, best_markov_file)
        sampled_ts_list = []
        model.eval()
        pbar = tqdm(total=len(test_list) * (len(test_list[0]) - 1))
        with torch.no_grad():
            for ts in test_list:
                samples_ts = [ts[0]]
                for g in ts[:-1]:
                    ## Convert to adjacency for GNN
                    edges = from_networkx(g).edge_index.to(device)
                    node_feat = torch.eye(num_nodes).to(device)
                    # _, pred_edges, _ = model(num_nodes, edges, node_feat)
                    # pred_g = nx.empty_graph(num_nodes)
                    # # Build the delta matrix
                    # pred_g.add_edges_from(pred_edges)
                    # delta = nx.to_numpy_array(pred_g)
                    delta = model(num_nodes, edges, node_feat)
                    adj = nx.to_numpy_array(g)
                    # adj = (adj + delta) % 2
                    adj = (adj + delta).clip(0, 1)
                    samples_ts.append(nx.Graph(adj))
                    pbar.update()
                sampled_ts_list.append(samples_ts)

        print('--Saving Sampled TS--')
        save_name = os.path.join(self.args.save_dir, 'sampled_ts.pkl')
        graph_utils.save_graph_list(sampled_ts_list, save_name)

        with open('experiment_files/last_test.txt', 'w') as f:
            f.write(self.args.save_dir)


def pretty_json(hp):
    json_hp = json.dumps(hp, indent=2)
    return "".join("\t" + line for line in json_hp.splitlines(True))
