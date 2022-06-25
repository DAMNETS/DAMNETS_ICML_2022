import os
import argparse
import numpy as np
import torch
import random
import yaml
import pickle as pkl
from easydict import EasyDict as edict

from utils.graph_generators import *
from utils.arg_helper import mkdir
from utils.graph_utils import save_graph_list


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Script to generate the synthetic data for the DAMNETS experiments")
    parser.add_argument(
        '-c',
        '--config_file',
        type=str,
        help="Name of dataset config file.",
    )
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
        help='Save path for the data'
    )
    parser.add_argument(
        '-d',
        '--dataset_name',
        type=str,
        default='',
        help='A name for the saved dataset. If none given, will use the name of the generating function.'
    )
    args = parser.parse_args()
    return args


def main():
    c_args = parse_arguments()
    save_path = c_args.data_path
    mkdir(save_path)
    path = os.path.join(save_path, c_args.config_file)
    config = edict(yaml.full_load(open(path, 'r')))
    config.n_workers = c_args.num_workers
    random.seed(config.seed)
    np.random.seed(config.seed)


    ts_list = generate_graphs(config)
    dataset_name = config.name if c_args.dataset_name == '' else c_args.dataset_name
    path = os.path.join(save_path, f'{dataset_name}.pkl')
    save_graph_list(ts_list, path)


if __name__ == '__main__':
    main()