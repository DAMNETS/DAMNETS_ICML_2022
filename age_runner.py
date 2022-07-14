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

from models import *
from dataset import *
from utils import graph_generators
from utils.train_utils import snapshot_model, load_model
from utils.age_validator import AGEValidator
import utils.graph_utils as graph_utils
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

    def train(self):
        # Create data loader
        dataset = os.path.join(self.args.data_path, self.args.dataset_name)
        train_graphs = graph_utils.load_graph_ts(dataset + '_train_graphs_raw.pkl')
        val_graphs = graph_utils.load_graph_ts(dataset + '_val_graphs_raw.pkl')
        device = self.exp_args.device
        train_dataset = TFTSampler(train_graphs, self.args, tag='train')
        val_dataset = TFTSampler(val_graphs, self.args, tag='val')
        self.args.model.input_size = nx.number_of_nodes(train_graphs[0][0])
        model = AGE(self.args).to(device)
        if self.writer is not None:
            self.writer.add_text('Model', str(model))
        print(model)
        train_loader = DataLoader(train_dataset,
                                  batch_size=self.exp_args.train.batch_size,
                                  collate_fn=train_dataset.collate_fn,
                                  shuffle=self.exp_args.train.shuffle_data,
                                  pin_memory=True)
        val_loader = DataLoader(val_dataset,
                                  batch_size=self.exp_args.train.batch_size,
                                  collate_fn=train_dataset.collate_fn,
                                  shuffle=self.exp_args.train.shuffle_data,
                                  pin_memory=True)
        self.validator = AGEValidator(val_loader, model, self.args, self.writer)
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
            for batch in train_loader:
                model.train()
                # opt.zero_grad()
                loss = model(batch).mean()  # Loss already averaged within batch in fwd, we average over gpus here
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.exp_args.train.clip_grad)
                # opt.step()
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
        # Plot the losses
        with open('experiment_files/last_age_train.txt', 'w') as f:
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

    # def test_(self):
    #     gpus = self.exp_args.gpus
    #     # device = self.exp_args.gpus[0]
    #     test_graphs = graph_utils.load_graph_ts(
    #         os.path.join(self.args.experiment.test.graph_dir, 'test_graphs.pkl'))
    #     model = eval(self.model_args.name)(self.args)#.to(device)
    #     model = torch.nn.DataParallel(model, device_ids=None, output_device=gpus[0]).to(gpus[0])
    #     ## Sample the Markov Results
    #     best_markov_file = os.path.join(self.args.model_save_dir,
    #                                    f'{val}.pt')
    #     # print(f'Best Model File : {best_markov_file}')
    #     load_model(model, best_markov_file)
    #     markov_ts = self.sample_nts(test_graphs, model)
    #
    #     with open('experiment_files/last_age_test.txt', 'w') as f:
    #         f.write(self.args.save_dir)


    def test(self):
        dataset = os.path.join(self.args.data_path, self.args.dataset_name)
        test_list = graph_utils.load_graph_ts(dataset + '_test_graphs.pkl')
        device = self.exp_args.device
        num_nodes = nx.number_of_nodes(test_list[0][0])

        self.args.model.input_size = num_nodes
        model = AGE(self.args).to(device)
        best_markov_file = os.path.join(self.args.model_save_dir,
                                        'val.pt')
        # print(f'Best Model File : {best_markov_file}')
        load_model(model, best_markov_file)
        sampled_ts_list = []
        model.eval()
        pbar = tqdm(total=len(test_list) * (len(test_list[0]) - 1))
        with torch.no_grad():
            for ts in test_list:
                samples_ts = [ts[0]]
                for g in ts[:-1]:
                    adj = np.tril(nx.to_numpy_array(g), k=-1)
                    data = {'adj': torch.tensor(adj, dtype=torch.float32).unsqueeze(0),
                            'is_sampling': True}
                    adj_pred = model(data).squeeze(0)
                    adj_pred = (adj_pred + adj_pred.T).cpu().numpy()
                    samples_ts.append(nx.Graph(adj_pred))
                    pbar.update()
                sampled_ts_list.append(samples_ts)

        print('--Saving Sampled TS--')
        save_name = os.path.join(self.args.save_dir, 'age_samples.pkl')
        graph_utils.save_graph_list(sampled_ts_list, save_name)

        with open('experiment_files/last_age_test.txt', 'w') as f:
            f.write(self.args.save_dir)


def pretty_json(hp):
    json_hp = json.dumps(hp, indent=2)
    return "".join("\t" + line for line in json_hp.splitlines(True))
