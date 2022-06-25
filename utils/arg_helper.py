import yaml
import os
import time
from easydict import EasyDict as edict


def get_config(config_file, exp_dir=None, is_test=False, tag=''):
    """ Construct and snapshot hyper parameters """
    config = edict(yaml.full_load(open(config_file, 'r')))

    # create hyper parameters
    config.run_id = str(os.getpid())
    if is_test:
        tag = config.dataset_name
    tags = [
        config.model.name, tag,
        time.strftime('%Y-%b-%d-%H-%M-%S'), config.run_id
    ]
    if is_test:
        tags.insert(1, 'test')
    config.exp_name = '_'.join(tags)

    # if hasattr(config, 'exp_name'):
    #     if config.exp_name == 'default':
    #         config.exp_name = '_'.join(tags)
    # else:
    #     config.exp_name = '_'.join(tags)

    if exp_dir is not None:
        config.exp_dir = exp_dir

    config.save_dir = os.path.join(config.exp_dir, config.exp_name)
    save_name = os.path.join(config.save_dir, 'config.yaml')

    # snapshot hyperparameters
    mkdir(config.exp_dir)
    mkdir(config.save_dir)

    if not is_test:
        config.model_save_dir = os.path.join(config.save_dir, 'models')
        mkdir(config.model_save_dir)

    # Save experimental parameters to the experiment file
    yaml.dump(edict2dict(config), open(save_name, 'w'), default_flow_style=False)

    return config


def edict2dict(edict_obj):
    dict_obj = {}

    for key, vals in edict_obj.items():
        if isinstance(vals, edict):
            dict_obj[key] = edict2dict(vals)
        else:
            dict_obj[key] = vals
    return dict_obj


def mkdir(folder):
    if not os.path.isdir(folder):
        os.makedirs(folder)
