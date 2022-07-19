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
import numpy as np
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
from models.damnets_signed import DamnetsSigned
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

            self.train_loss = None
            self.validator = None
            self.writer = SummaryWriter(args.save_dir) if self.exp_args.train.use_writer else None


    def _make_loader(self, zipped):
        graphs = []
        for g, delta in zipped:
            g_id = TreeLib.InsertGraph(delta)
            g.graph_id = g_id
            graphs.append(g)
        return DataLoader(graphs,
                          batch_size=self.exp_args.train.batch_size,
                          shuffle=self.exp_args.train.shuffle_data,
                          pin_memory=True)
    def train(self):
        # Create data loader
        dataset = os.path.join(self.args.data_path, self.args.dataset_name)
        train_zipped = graph_utils.load_graph_ts(dataset + '_train_graphs.pkl')
        val_zipped = graph_utils.load_graph_ts(dataset + '_val_graphs.pkl')
        num_nodes = nx.number_of_nodes(train_zipped[0][1])
        device = self.exp_args.device
        ## set bigg args for compatability
        self.model_args.bigg.gpu = -1 if device == 'cpu' else device
        self.model_args.bigg.device = device
        self.model_args.bigg.seed = self.args.seed
        self.model_args.bigg.max_num_nodes = num_nodes
        setup_treelib(self.model_args.bigg)

        train_loader = self._make_loader(train_zipped)
        val_loader = self._make_loader(val_zipped)
        print('All data prepared. Loading model.')
        model = DamnetsSigned(self.model_args).to(device)
        if self.writer is not None:
            self.writer.add_text('Model', str(model))

        print(model)
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
                graph_ids = batch.graph_id
                batch.to(device)
                loss = model.forward_train(batch.x, batch.edge_index, graph_ids, num_nodes) / len(graph_ids)
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
                if scheduler.cooldown_counter == self.exp_args.validation.cooldown:
                    # So that it doesn't early stop right after a lr decay
                    self.validator.update_buffer(epoch)
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

        # TODO: add support for loading a generic snapshot.
        model = DamnetsSigned(self.model_args).to(device)
        load_model(model, os.path.join(self.args.model_save_dir, 'val.pt'))
        sampled_ts_list = []
        model.eval()
        pbar = tqdm(total=len(test_list) * (len(test_list[0]) - 1))
        add_ratio = []
        remv_ratio = []
        ar_ratio = []
        with torch.no_grad():
            for ts in test_list:
                samples_ts = [ts[0]]
                for g_ in ts[:-1]:
                    g = nx.Graph(g_)
                    ## Convert to adjacency for GNN
                    edges = from_networkx(g).edge_index.to(device)
                    node_feat = torch.eye(num_nodes).to(device)
                    delta_entries = model(num_nodes, node_feat, edges, g)
                    # Add the sampled new edges
                    new_edges = [(i, j) for i, j, w in delta_entries if w == 1]
                    print(f'Num additions: {len(new_edges)}')
                    add_ratio.append(sum([not g.has_edge(*edge) for edge in new_edges]) / len(new_edges))
                    print(f'Num valid additions: {sum([not g.has_edge(*edge) for edge in new_edges])}')
                    g.add_edges_from(new_edges)
                    # Remove the samples deletions
                    deletions = [(i, j) for i, j, w in delta_entries if w == -1]
                    print(f'Num removals: {len(deletions)}')
                    remv_ratio.append(sum([g.has_edge(*edge) for edge in deletions]) / (len(deletions) + 1))
                    print(f'Num valid removals: {sum([g.has_edge(*edge) for edge in deletions])}')
                    g.remove_edges_from(deletions)
                    ar_ratio.append(len(new_edges) / (len(deletions) + 1))
                    # NOTE: add/remove edges_from silently fails if edge already exists or doesn't exist.
                    # This is intended behaviour for us.
                    samples_ts.append(g)
                    pbar.update()
                sampled_ts_list.append(samples_ts)
        print('Average add ratio: ' , np.mean(add_ratio))
        print('Average rmv ratio: ', np.mean(remv_ratio))
        print('Average ar ratio: ', np.mean(ar_ratio))
        print('--Saving Sampled TS--')
        save_name = os.path.join(self.args.save_dir, 'sampled_ts.pkl')
        graph_utils.save_graph_list(sampled_ts_list, save_name)

        with open('experiment_files/last_test.txt', 'w') as f:
            f.write(self.args.save_dir)


def pretty_json(hp):
    json_hp = json.dumps(hp, indent=2)
    return "".join("\t" + line for line in json_hp.splitlines(True))
