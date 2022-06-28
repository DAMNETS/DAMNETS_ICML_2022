import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GAT

from bigg.model.tree_model import RecurTreeGen


class DMNETS_GNN_TREE(nn.Module):
    def __init__(self, args):
        super(DMNETS_GNN_TREE, self).__init__()
        self.tree_decoder = RecurTreeGen(args.decoder_args)

