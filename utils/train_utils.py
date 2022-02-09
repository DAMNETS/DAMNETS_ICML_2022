import os
import torch
import yaml
from utils.arg_helper import edict2dict


def snapshot_model(model, epoch, args, is_best=False):
    torch.save(model.state_dict(), os.path.join(args.model_save_dir, f'{epoch}.pt'))
    if is_best:
        # update config file's test path
        save_name = os.path.join(args.save_dir, 'config.yaml')
        args.experiment.test.best_val_epoch = epoch
        yaml.dump(edict2dict(args),
                  open(save_name, 'w'), default_flow_style=False)


def load_model(model, file_name):
    model.load_state_dict(torch.load(file_name))