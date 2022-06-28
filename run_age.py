import argparse
import os
import numpy as np
import torch
import random

from age_runner import Runner
from utils.arg_helper import get_config


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="AGE - Attention Based Graph Evolution.")
    parser.add_argument('-t', '--test', help="Test model", action='store_true')
    parser.add_argument(
        '-c',
        '--config_file',
        type=str,
        help="Path of model config file, if empty and -t flag is given will test from the last model trained.",
        nargs='?',
        const=''
    )
    parser.add_argument(
        '-d',
        '--dataset',
        type=str,
        help='Name of the dataset to use. This should be generated in the data/ folder. Do not add .pkl to the name.'
             'Only needed for train.',
        default=''
    )
    args = parser.parse_args()
    return args


def main():
    c_args = parse_arguments()
    if c_args.test:
        if c_args.config_file == '':
            with open('experiment_files/last_train.txt', 'r') as f:
                config_file = os.path.join(f.readline(), 'config.yaml')
        else:
            config_file = c_args.config_file
        args = get_config(config_file, is_test=True)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        runner = Runner(args, is_test=True)
        runner.test()
    else:
        args = get_config(os.path.join('experiment_configs', c_args.config_file), tag=c_args.dataset)
        # args.data_file = os.path.join(args.data_path, f'{c_args.dataset}.pkl')
        args.dataset_name = c_args.dataset
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        runner = Runner(args)
        try:
            runner.train()
        except KeyboardInterrupt:
            print('Stopping Training')
            runner.save_training_info()


if __name__ == '__main__':
    main()
