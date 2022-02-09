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

from models import *
from dataset import *
from utils import graph_generators
from utils.train_utils import snapshot_model, load_model
from utils.gnn_validator import GNNValidator
import utils.graph_utils as graph_utils
from eval import plot_network_statistics, ba_plots, three_comm_plots, plot_top_N
import json

class Runner:
    def __init__(self, args, is_test=False):
        self.args = args
        self.dataset_args = args.dataset
        self.model_args = args.model
        self.exp_args = args.experiment
        if not is_test:
            print('Mode: Training | Loading Graphs')
            try:
                graphs = graph_utils.load_graph_ts(args.dataset.path)
                train_ix = int(len(graphs) * self.exp_args.train.train_prop)
                if train_ix == 0:
                    self.train_graphs = test_graphs = graphs
                else:
                    self.train_graphs = graphs[:train_ix]
                    test_graphs = graphs[train_ix:]
                self.dataset_args.T = len(graphs[0])
            except AttributeError:
                self.train_graphs, test_graphs = graph_generators.generate_graphs(args.dataset)
            val_len = int(len(self.train_graphs) * self.exp_args.validation.val_p)
            self.val_graphs = self.train_graphs[:val_len] if val_len > 0 else self.train_graphs[val_len:]
            self.train_graphs = self.train_graphs[val_len:]
            # Prepare the validation graphs
            print('Number of Training TS: ', len(self.train_graphs))
            print('Number of Val TS: ', len(self.val_graphs))
            print('Number of Test TS: ', len(test_graphs))
            print('TS length (T): ', self.dataset_args.T)
            print('Saving Graphs')
            graph_utils.save_graph_list(
                self.train_graphs, os.path.join(self.args.save_dir, 'train_graphs.pkl'))
            graph_utils.save_graph_list(
                self.val_graphs, os.path.join(self.args.save_dir, 'val_graphs.pkl'))
            graph_utils.save_graph_list(
                test_graphs, os.path.join(self.args.save_dir, 'test_graphs.pkl'))
            args.experiment.test.graph_dir = self.args.save_dir

            self.train_loss = None
            self.validator = None
            self.writer = SummaryWriter(args.save_dir) if self.exp_args.train.use_writer else None

    def train(self):
        # Create data loader
        gpus = self.exp_args.gpus
        train_dataset = eval(self.dataset_args.loader_name)(self.train_graphs, self.args, tag='train')

        model = eval(self.model_args.name)(self.args)
        if self.writer is not None:
            self.writer.add_text('Model', str(model))
        model = torch.nn.DataParallel(model, device_ids=gpus).to(gpus[0])

        print(model)
        train_loader = DataLoader(train_dataset,
                                  batch_size=self.exp_args.train.batch_size,
                                  collate_fn=train_dataset.collate_fn,
                                  shuffle=self.exp_args.train.shuffle_data,
                                  pin_memory=True)
        self.validator = GNNValidator(self.val_graphs, model, self.args, self.writer)
        opt = optim.Adam(model.parameters(),
                         lr=self.exp_args.train.lr,
                         weight_decay=self.exp_args.train.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, self.exp_args.train.cycle_len)
        snapshot_epochs = self.exp_args.train.snapshot_epochs
        results = defaultdict(list)
        iter_count = 0
        train_start = time.time()
        self.train_start = time.time()
        for epoch in range(self.exp_args.train.epochs):
            start = time.time()
            train_iterator = train_loader.__iter__()
            if self.writer is not None:
                self.writer.add_scalar('Learning Rate', scheduler.get_last_lr()[0], epoch)
            for inner_iter in range(len(train_loader) // len(gpus)):
                model.train()
                batch_data = []
                for _ in gpus:
                    data = train_iterator.next()
                    batch_data.append(data)
                    iter_count += 1
                opt.zero_grad()
                loss = model(*batch_data).mean()  # Loss already averaged within batch in fwd, we average over gpus here
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.exp_args.train.clip_grad)
                opt.step()
                loss = float(loss.data.cpu().numpy())
                if self.writer is not None:
                    self.writer.add_scalar('Loss/Train', loss, iter_count)
                results['train_loss'] += [loss]
                results['train_step'] += [iter_count]
                if iter_count % self.args.experiment.train.display_iters == 0 \
                        and epoch > self.args.experiment.validation.val_epochs:
                    print(f'Iteration: {iter_count} | Train Loss: {loss: .3f} | '
                          f'Epoch: {epoch} | '
                          f'Best Validation: {self.validator.best_val_loss: .3f} @ epoch {self.validator.best_epoch} | '
                          f'Last Val Loss: {self.validator.val_losses[-1]: .3f} | '
                          f'Total Train Time: {(time.time() - train_start) / 60: .3f}m | '
                          f'Last Epoch Time: {epoch_time: .3f}s | '
                          f'LR: {scheduler.get_last_lr()[0]: .6f}')
            scheduler.step()
            if epoch % self.args.experiment.validation.val_epochs == 0 and epoch > 0:
                stop = self.validator.validate(epoch)
                if stop:
                    print('EARLY STOPPING TRIGGERED')
                    break
            if (epoch + 1) % snapshot_epochs == 0:
                snapshot_model(model, epoch, self.args)
            epoch_time = time.time() - start
        pickle.dump(results, open(os.path.join(self.args.save_dir, 'train_stats.pkl'), 'wb'))
        self.save_training_info()

    def save_training_info(self):
        # Plot the losses
        with open('experiment_files/last_train.txt', 'w') as f:
            f.write(self.args.save_dir)
        if self.writer is not None:
            self.writer.add_text('Loss/Corrections', str(self.validator.correction_epochs))
            end_time = time.time() - self.train_start
            self.writer.add_text('Training Time', f'{end_time / 60: .3f} Minutes')
            self.writer.add_text('Args', pretty_json(self.args))
            self.writer.close()

    def sample_nts(self, graphs, model):
        sampler = GNNTestSampler(graphs, self.args, tag=f'multistep_test')
        loader = DataLoader(sampler,
                             collate_fn=sampler.collate_fn,
                             batch_size=self.args.experiment.test.batch_size)

        model.eval()
        print('Beginning Sampling')
        with torch.no_grad():
            ts_batches = []
            for batch_idx, ts_data in enumerate(tqdm(loader)):
                # Each data is a batch of time series
                ts_batch = []
                for t, data in enumerate(ts_data):
                    ts_batch += model.forward(data)
                ts_batch = [ts_data[0]['adj']] + ts_batch
                assert len(ts_batch) == len(graphs[0])
                ts_batches.append(ts_batch)
            print('Sampling Complete')
        ts_list = []
        for ts_batch in ts_batches:
            for b in range(ts_batch[0].shape[0]):
                ts = []
                for t in range(self.args.dataset.T):
                    ts.append(nx.Graph(ts_batch[t][b].cpu().numpy()))
                assert len(ts) == len(graphs[0])
                ts_list.append(ts)
        assert len(ts_list) == len(graphs)
        return ts_list

    def test(self):
        gpus = self.exp_args.gpus
        # device = self.exp_args.gpus[0]
        test_graphs = graph_utils.load_graph_ts(
            os.path.join(self.args.experiment.test.graph_dir, 'test_graphs.pkl'))
        model = eval(self.model_args.name)(self.args)#.to(device)
        model = torch.nn.DataParallel(model, device_ids=None, output_device=gpus[0]).to(gpus[0])
        ## Sample the Markov Results
        best_markov_file = os.path.join(self.args.model_save_dir,
                                       f'{self.args.experiment.test.best_val_epoch}.pt')
        print(f'Best Model File : {best_markov_file}')
        load_model(model, best_markov_file)
        markov_ts = self.sample_nts(test_graphs, model)
        print('Plotting Summary Statistics')
        self.eval(markov_ts, test_graphs, tag=self.args.model.name)
        save_name = os.path.join(self.args.save_dir, 'sampled_ts.pkl')
        graph_utils.save_graph_list(markov_ts, save_name)

        print('Plotting test')
        plot_top_N(test_graphs, self.args, 'test')

        with open('experiment_files/last_test.txt', 'w') as f:
            f.write(self.args.save_dir)

    def eval(self, sampled_ts, test_ts, tag):
        name = self.args.dataset.name
        if name == 'ba':
            ba_plots(sampled_ts, test_ts, self.args, tag)
        elif name == '3_comm_decay':
            three_comm_plots(sampled_ts, test_ts, self.args, tag)
        plot_network_statistics(test_ts, sampled_ts, self.args, tag)
        plot_top_N(sampled_ts, self.args, tag)


def pretty_json(hp):
    json_hp = json.dumps(hp, indent=2)
    return "".join("\t" + line for line in json_hp.splitlines(True))
