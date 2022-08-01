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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from torch.nn.parameter import Parameter
from bigg.common.pytorch_util import glorot_uniform, MLP, BinaryTreeLSTMCell
from tqdm import tqdm
from bigg.model.util import AdjNode, ColAutomata, AdjRow
from bigg.model.tree_clib.tree_lib import TreeLib
from bigg.torch_ops import multi_index_select, PosEncoding


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def hc_multi_select(ids_from, ids_to, h_froms, c_froms):
    h_vecs = multi_index_select(ids_from,
                                ids_to,
                                *h_froms)
    c_vecs = multi_index_select(ids_from,
                                ids_to,
                                *c_froms)
    return h_vecs, c_vecs


def tree_state_select(h_bot, c_bot, h_buf, c_buf, fn_all_ids):
    bot_froms, bot_tos, prev_froms, prev_tos = fn_all_ids()
    if h_buf is None or prev_tos is None:
        h_vecs = multi_index_select([bot_froms], [bot_tos], h_bot)
        c_vecs = multi_index_select([bot_froms], [bot_tos], c_bot)
    elif h_bot is None or bot_tos is None:
        h_vecs = multi_index_select([prev_froms], [prev_tos], h_buf)
        c_vecs = multi_index_select([prev_froms], [prev_tos], c_buf)
    else:
        h_vecs, c_vecs = hc_multi_select([bot_froms, prev_froms],
                                         [bot_tos, prev_tos],
                                         [h_bot, h_buf], [c_bot, c_buf])
    return h_vecs, c_vecs


def batch_tree_lstm2(h_bot, c_bot, h_buf, c_buf, fn_all_ids, cell):
    h_list = []
    c_list = []
    for i in range(2): # Get left and right.
        h_vecs, c_vecs = tree_state_select(h_bot, c_bot, h_buf, c_buf, lambda : fn_all_ids(i))
        h_list.append(h_vecs)
        c_list.append(c_vecs)
    return cell((h_list[0], c_list[0]), (h_list[1], c_list[1]))


def batch_tree_lstm3(h_bot, c_bot, h_buf, c_buf, h_past, c_past, fn_all_ids, cell):
    # Fenwick batching.
    if h_past is None:
        return batch_tree_lstm2(h_bot, c_bot, h_buf, c_buf, lambda i: fn_all_ids(i)[:-2], cell)
    elif h_bot is None:
        return batch_tree_lstm2(h_buf, c_buf, h_past, c_past, lambda i: fn_all_ids(i)[2:], cell)
    elif h_buf is None:
        return batch_tree_lstm2(h_bot, c_bot, h_past, c_past, lambda i: fn_all_ids(i)[0, 1, 4, 5], cell)
    else:
        h_list = []
        c_list = []
        for i in range(2):
            bot_froms, bot_tos, prev_froms, prev_tos, past_froms, past_tos = fn_all_ids(i)
            h_vecs, c_vecs = hc_multi_select([bot_froms, prev_froms, past_froms],
                                             [bot_tos, prev_tos, past_tos],
                                             [h_bot, h_buf, h_past],
                                             [c_bot, c_buf, c_past])
            h_list.append(h_vecs)
            c_list.append(c_vecs)
        return cell((h_list[0], c_list[0]), (h_list[1], c_list[1]))


# class FenwickTree(nn.Module):
#     def __init__(self, args):
#         super(FenwickTree, self).__init__()
#         embed_dim = args.embed_dim
#         self.init_h0 = Parameter(torch.Tensor(1, embed_dim))
#         self.init_c0 = Parameter(torch.Tensor(1, embed_dim))
#         glorot_uniform(self)
#
#         self.merge_cell = BinaryTreeLSTMCell(embed_dim)
#         self.summary_cell = BinaryTreeLSTMCell(embed_dim)
#         if args.pos_enc:
#             self.pos_enc = PosEncoding(embed_dim, args.device, args.pos_base)
#         else:
#             self.pos_enc = lambda x: 0
#
#     def reset(self, list_states=[]):
#         self.list_states = []
#         for l in list_states:
#             t = []
#             for e in l:
#                 t.append(e)
#             self.list_states.append(t)
#
#     def append_state(self, state, level):
#         if level >= len(self.list_states):
#             num_aug = level - len(self.list_states) + 1
#             for i in range(num_aug):
#                 self.list_states.append([])
#         self.list_states[level].append(state)
#
#     def forward(self, new_state=None):
#         if new_state is None:
#             if len(self.list_states) == 0:
#                 return (self.init_h0, self.init_c0)
#         else:
#             self.append_state(new_state, 0)
#         pos = 0
#         # Compute the raw summaries needed.
#         while pos < len(self.list_states):
#             if len(self.list_states[pos]) >= 2:
#                 lch_state, rch_state = self.list_states[pos]  # assert the length is 2
#                 new_state = self.merge_cell(lch_state, rch_state)
#                 self.list_states[pos] = []
#                 self.append_state(new_state, pos + 1)
#             pos += 1
#         state = None
#         # Summarise all the raw states computed.
#         for pos in range(len(self.list_states)):
#             if len(self.list_states[pos]) == 0:
#                 continue
#             cur_state = self.list_states[pos][0]
#             if state is None:
#                 state = cur_state
#             else:
#                 state = self.summary_cell(state, cur_state)
#         return state
#
#     def forward_train(self, h_bot, c_bot, h_buf0, c_buf0, prev_rowsum_h, prrev_rowsum_c):
#         # embed row tree
#         tree_agg_ids = TreeLib.PrepareRowEmbed()
#         # This prepares the ids from which choose the bottom up row aggregations to be
#         # merged into the fenwick tree. h_bot is the same as in the bottom up aggregation,
#         # and appears here to give the value for empty rows, as well as for the case where
#         # a root itself has a node TODO: check this last point.
#         # tree_agg_ids contains the locations to perform summary at
#         # each possible level of the Fenwick tree.
#         row_embeds = [(self.init_h0, self.init_c0)]
#         if h_bot is not None:
#             row_embeds.append((h_bot, c_bot))
#         if prev_rowsum_h is not None:
#             row_embeds.append((prev_rowsum_h, prrev_rowsum_c))
#         if h_buf0 is not None:
#             row_embeds.append((h_buf0, c_buf0))
#
#         th_bot = h_bot
#         tc_bot = c_bot
#         for i, all_ids in enumerate(tree_agg_ids):
#             fn_ids = lambda x: all_ids[x]
#             if i:
#                 th_bot = tc_bot = None
#
#             new_states = batch_tree_lstm3(th_bot, tc_bot,
#                                           row_embeds[-1][0], row_embeds[-1][1],
#                                           prev_rowsum_h, prrev_rowsum_c,
#                                           fn_ids, self.merge_cell)
#             row_embeds.append(new_states)
#         h_list, c_list = zip(*row_embeds)
#         joint_h = torch.cat(h_list, dim=0)
#         joint_c = torch.cat(c_list, dim=0)
#
#         # get history representation
#         init_select, all_ids, last_tos, next_ids, pos_info = TreeLib.PrepareRowSummary()
#         cur_state = (joint_h[init_select], joint_c[init_select])
#         ret_state = (joint_h[next_ids], joint_c[next_ids])
#         hist_rnn_states = []
#         hist_froms = []
#         hist_tos = []
#         for i, (done_from, done_to, proceed_from, proceed_input) in enumerate(all_ids):
#             hist_froms.append(done_from)
#             hist_tos.append(done_to)
#             hist_rnn_states.append(cur_state)
#
#             next_input = joint_h[proceed_input], joint_c[proceed_input]
#             sub_state = cur_state[0][proceed_from], cur_state[1][proceed_from]
#             cur_state = self.summary_cell(sub_state, next_input)
#         hist_rnn_states.append(cur_state)
#         hist_froms.append(None)
#         hist_tos.append(last_tos)
#         hist_h_list, hist_c_list = zip(*hist_rnn_states)
#         pos_embed = self.pos_enc(pos_info)
#         row_h = multi_index_select(hist_froms, hist_tos, *hist_h_list) + pos_embed
#         row_c = multi_index_select(hist_froms, hist_tos, *hist_c_list) + pos_embed
#         return (row_h, row_c), ret_state


class BitsRepNet(nn.Module):
    def __init__(self, args):
        super(BitsRepNet, self).__init__()
        self.bits_compress = args.bits_compress
        self.out_dim = args.embed_dim
        assert self.out_dim >= self.bits_compress
        self.device = args.device

    def forward(self, on_bits, n_cols):
        h = torch.ones(1, self.out_dim).to(self.device) * -10
        h[0, :n_cols] = 0
        pos_bit_pos = [x for (x, sign) in on_bits if sign == 1]
        neg_bit_pos = [x for (x, sign) in on_bits if sign == -1]
        h[0, pos_bit_pos] = 1.0
        h[0, neg_bit_pos] = -1.0

        return h, h


class RecurTreeGen(nn.Module):
    def __init__(self, args):
        super(RecurTreeGen, self).__init__()
        self.directed = args.directed
        self.self_loop = args.self_loop
        self.bits_compress = args.bits_compress
        self.greedy_frac = args.greedy_frac
        self.share_param = args.share_param
        self.embed_dim = args.embed_dim
        self.dropout = args.dropout
        if not self.bits_compress:
            self.leaf_h0_pos = Parameter(torch.Tensor(1, args.embed_dim))
            self.leaf_c0_pos = Parameter(torch.Tensor(1, args.embed_dim))
            self.leaf_h0_neg = Parameter(torch.Tensor(1, args.embed_dim))
            self.leaf_c0_neg = Parameter(torch.Tensor(1, args.embed_dim))
            self.empty_h0 = Parameter(torch.Tensor(1, args.embed_dim))
            self.empty_c0 = Parameter(torch.Tensor(1, args.embed_dim))

        self.init_h0 = Parameter(torch.Tensor(1, args.embed_dim))
        self.init_c0 = Parameter(torch.Tensor(1, args.embed_dim))

        self.topdown_left_embed = Parameter(torch.Tensor(4, args.embed_dim))
        self.topdown_right_embed = Parameter(torch.Tensor(4, args.embed_dim))
        glorot_uniform(self)

        # h0 = torch.ones(1, args.embed_dim)
        # h0 /= torch.norm(h0)
        # self.leaf_h0_pos = torch.nn.Parameter(h0, requires_grad=False)
        # self.leaf_h0_neg = torch.nn.Parameter(h0 * -1, requires_grad=False)

        if self.bits_compress > 0:
            self.bit_rep_net = BitsRepNet(args)

        if self.share_param:
            self.m_l2r_cell = BinaryTreeLSTMCell(args.embed_dim)
            self.lr2p_cell = BinaryTreeLSTMCell(args.embed_dim)
            self.pred_has_ch = MLP(args.embed_dim, [2 * args.embed_dim, 1], dropout=self.dropout)
            self.m_pred_has_left = MLP(args.embed_dim, [2 * args.embed_dim, 1], dropout=self.dropout)
            self.m_pred_has_right = MLP(args.embed_dim, [2 * args.embed_dim, 1], dropout=self.dropout)
            self.m_cell_topdown = nn.LSTMCell(args.embed_dim, args.embed_dim)
            self.m_cell_topright = nn.LSTMCell(args.embed_dim, args.embed_dim)
            self.fuser = MLP(2 * args.embed_dim, [4 * args.embed_dim, args.embed_dim], dropout=self.dropout)
            self.m_pred_sign = MLP(args.embed_dim, [2 * args.embed_dim, 1], dropout=self.dropout)
        else:
            fn_pred = lambda: MLP(args.embed_dim, [2 * args.embed_dim, 1])
            fn_tree_cell = lambda: BinaryTreeLSTMCell(args.embed_dim)
            fn_lstm_cell = lambda: nn.LSTMCell(args.embed_dim, args.embed_dim)
            num_params = int(np.ceil(np.log2(args.max_num_nodes))) + 1
            self.pred_has_ch = fn_pred()

            pred_modules = [[] for _ in range(2)]
            tree_cell_modules = []
            lstm_cell_modules = [[] for _ in range(2)]
            for _ in range(num_params):
                for i in range(2):
                    pred_modules[i].append(fn_pred())
                    lstm_cell_modules[i].append(fn_lstm_cell())
                tree_cell_modules.append(fn_tree_cell())

            self.has_left_modules, self.has_right_modules = [nn.ModuleList(l) for l in pred_modules]
            self.l2r_modules= nn.ModuleList(tree_cell_modules)
            self.cell_topdown_modules, self.cell_topright_modules = [nn.ModuleList(l) for l in lstm_cell_modules]
            self.lr2p_cell = fn_tree_cell()
        # self.row_tree = FenwickTree(args)

        if args.tree_pos_enc:
            self.tree_pos_enc = PosEncoding(args.embed_dim, args.device, args.pos_base, bias=np.pi / 4)
        else:
            self.tree_pos_enc = lambda x: 0

        self.label_smoothing = 0.0
        self.use_st_attn = args.use_st_attn
        if args.use_st_attn: # Use source-target attention against GNN embeddings
            layer = nn.TransformerDecoderLayer(args.embed_dim,
                                                     nhead=args.num_heads,
                                                     dim_feedforward=args.dim_feedforward,
                                                     dropout=args.dropout,
                                                     batch_first=True)
            self.decoder = nn.TransformerDecoder(layer, args.num_tf_layers)
        else:  # Just use self-attention on rows of delta matrix.
            layer = nn.TransformerEncoderLayer(args.embed_dim,
                                                     nhead=args.num_heads,
                                                     dim_feedforward=args.dim_feedforward,
                                                     dropout=args.dropout,
                                                     batch_first=True)
            self.decoder = nn.TransformerEncoder(layer, args.num_tf_layers)
        self.row_pos_enc = PosEncoding(args.embed_dim, args.device, args.pos_base, bias=np.pi / 4)
        self.num_edges = 0

    def cell_topdown(self, x, y, lv):
        cell = self.m_cell_topdown if self.share_param else self.cell_topdown_modules[lv]
        return cell(x, y)

    def cell_topright(self, x, y, lv):
        cell = self.m_cell_topright if self.share_param else self.cell_topright_modules[lv]
        return cell(x, y)

    def l2r_cell(self, x, y, lv):
        cell = self.m_l2r_cell if self.share_param else self.l2r_modules[lv]
        return cell(x, y)

    def pred_has_left(self, x, lv):
        mlp = self.m_pred_has_left if self.share_param else self.has_left_modules[lv]
        return mlp(x)

    def pred_has_right(self, x, lv):
        mlp = self.m_pred_has_right if self.share_param else self.has_right_modules[lv]
        return mlp(x)

    def pred_sign(self, x):
        return self.m_pred_sign(x)

    def get_empty_state(self):
        if self.bits_compress:
            return self.bit_rep_net([], 1)
        else:
            return (self.empty_h0, self.empty_c0)

    def get_prob_fix(self, prob):
        p = prob * (1 - self.greedy_frac)
        if prob >= 0.5:
            p += self.greedy_frac
        return p

    def append_state(self, state, list, lv):
        if len(list) < lv:
            while len(list) < lv:
                list.append([])
        list[lv].append(state)

    def sample_leaf(self, state, col_sm, tree_node, ll, side=None):
        logits = self.pred_sign(state[0])
        p = torch.sigmoid(logits)
        if col_sm.supervised:  # TODO: check in supervised case that nothing gets through here we don't want to.
            if col_sm.next_edge is not None and tree_node.edge[1] == col_sm.next_edge[0]:
                sign = col_sm.next_edge[1]
            else:
                sign = 0
            # sign_ = torch.LongTensor([sign if 0 <= sign else 1]).to(logits.device)
        else:
            edge_decision = torch.bernoulli(p).item()
            edge_sign = col_sm.get_sign(*tree_node.edge)
            sign = edge_sign * edge_decision
        tree_node.bits_rep = [(0, sign)]
        if sign != 0:
            col_sm.add_edge(tree_node.col_range[0], sign)  # just incr position if supervised
        if tree_node.is_root:
            print('LEAF IS ROOT')
        # print(f'Depth: {tree_node.depth} | Side: {side if side is not None else 0}'
        #       f' Prob : {torch.softmax(logits, dim=1).detach().cpu().numpy()} | Sign: {sign}')
        # ce = F.cross_entropy(logits, sign_)
        has_edge = (sign != 0)
        ll = ll + (torch.log(p) if has_edge else torch.log(1 - p))
        if self.bits_compress:
            return ll, self.bit_rep_net(tree_node.bits_rep, tree_node.n_cols), has_edge
        else:
            if sign == 1:
                state = (self.leaf_h0_pos, self.leaf_c0_pos)
            elif sign == -1:
                state = (self.leaf_h0_neg, self.leaf_c0_neg)
            else:
                assert sign == 0
                state = (self.empty_h0, self.empty_c0)
            return ll, state, has_edge, None #sign_.item() + 1

    def gen_row(self, ll, state, tree_node, col_sm, lb, ub, side=None):
        assert lb <= ub
        if tree_node.is_root:
            # self.top_states.append(state[0].detach().cpu().numpy())
            # if tree_node.is_leaf:
            #     ll, state, has_edge, _ = self.sample_leaf(state, col_sm, tree_node, ll)
            #     return ll, state, has_edge
            # else:
            prob_has_edge = torch.sigmoid(self.pred_has_ch(state[0]))
            if col_sm.supervised:
                has_edge = len(col_sm.indices) > 0
            else:
                has_edge = np.random.rand() < self.get_prob_fix(prob_has_edge.item())
                if ub == 0:
                    has_edge = False
                if tree_node.n_cols <= 0:
                    has_edge = False
                if lb:
                    has_edge = True
            ll = ll + (torch.log(prob_has_edge) if has_edge else torch.log(1 - prob_has_edge))
            tree_node.has_edge = has_edge
        else:
            assert ub > 0
            tree_node.has_edge = True

        if not tree_node.has_edge:  # an empty tree
            return ll, self.get_empty_state(), 0

        if tree_node.is_leaf:
            edge_sign = col_sm.get_sign(*tree_node.edge)
            tree_node.bits_rep = [(0, edge_sign)]
            assert edge_sign != 0
            if edge_sign != 0:
                col_sm.add_edge(tree_node.col_range[0], edge_sign)  # just incr position if supervised
            if tree_node.is_root:
                print('LEAF IS ROOT')
            if self.bits_compress:
                return ll, self.bit_rep_net(tree_node.bits_rep, tree_node.n_cols), has_edge
            else:
                if edge_sign == 1:
                    state = (self.leaf_h0_pos, self.leaf_c0_pos)
                elif edge_sign == -1:
                    state = (self.leaf_h0_neg, self.leaf_c0_neg)
                else:
                    assert edge_sign == 0
                    state = (self.empty_h0, self.empty_c0)
                return ll, state, 1  # sign_.item() + 1
        else:
            tree_node.split()
            mid = (tree_node.col_range[0] + tree_node.col_range[1]) // 2
            left_prob = torch.sigmoid(self.pred_has_left(state[0], tree_node.depth))
            if col_sm.supervised:
                has_left = col_sm.next_edge[0] < mid
            else:
                has_left = np.random.rand() < self.get_prob_fix(left_prob.item())
                if ub == 0:
                    has_left = False
                if lb > tree_node.rch.n_cols:
                    has_left = True
            ll = ll + (torch.log(left_prob) if has_left else torch.log(1 - left_prob))
            left_pos = self.tree_pos_enc([tree_node.lch.n_cols])
            state = self.cell_topdown(self.topdown_left_embed[[int(has_left)]] + left_pos, state, tree_node.depth)
            if has_left:
                lub = min(tree_node.lch.n_cols, ub)
                llb = max(0, lb - tree_node.rch.n_cols)
                ll, left_state, num_left = self.gen_row(ll, state, tree_node.lch, col_sm, llb, lub, 'left')
            else:
                left_state = self.get_empty_state()
                num_left = 0

            right_pos = self.tree_pos_enc([tree_node.rch.n_cols])
            topdown_state = self.l2r_cell(state, (left_state[0] + right_pos, left_state[1] + right_pos), tree_node.depth)
            rlb = max(0, lb - num_left)
            rub = min(tree_node.rch.n_cols, ub - num_left)
            if not has_left:  # Know it has edge, not in left => it's in right.
                has_right = True
            else:
                right_prob = torch.sigmoid(self.pred_has_right(topdown_state[0], tree_node.depth))
                if col_sm.supervised:
                    has_right = col_sm.has_edge(mid, tree_node.col_range[1])
                else:
                    has_right = np.random.rand() < self.get_prob_fix(right_prob.item())
                    if rub == 0:
                        has_right = False
                    if rlb:
                        has_right = True
                ll = ll + (torch.log(right_prob) if has_right else torch.log(1 - right_prob))
            topdown_state = self.cell_topright(self.topdown_right_embed[[int(has_right)]], topdown_state, tree_node.depth)

            if has_right:  # has edge in right child
                ll, right_state, num_right = self.gen_row(ll, topdown_state, tree_node.rch, col_sm, rlb, rub, 'right')
            else:
                right_state = self.get_empty_state()
                num_right = 0
            if tree_node.col_range[1] - tree_node.col_range[0] <= self.bits_compress:
                summary_state = self.bit_rep_net(tree_node.bits_rep, tree_node.n_cols)
            else:
                summary_state = self.lr2p_cell(left_state, right_state)
            return ll, summary_state, num_left + num_right

    def forward(self, node_end, gnn_embeds, g, edge_list=None,
                node_start=0, list_states=[], lb_list=None, ub_list=None, col_range=None, num_nodes=None, display=False):
        pos = 0
        total_ll = 0.0
        edges = []
        # controller_state = (self.init_h0, self.init_c0)
        if num_nodes is None:
            num_nodes = node_end
        pbar = range(0, node_end)
        if display:
            pbar = tqdm(pbar)
        controller_states = []
        h = self.init_h0 + self.row_pos_enc([num_nodes])
        c = self.init_c0
        gnn_embeds = gnn_embeds.unsqueeze(0) # for transformer batching.
        for i in pbar:
            if edge_list is None:
                col_sm = ColAutomata(supervised=False, g=g)
            else:
                indices = []
                while pos < len(edge_list) and i == edge_list[pos][0]:
                    indices.append((edge_list[pos][1], edge_list[pos][2]))
                    pos += 1
                indices.sort(key = lambda x: x[0])
                col_sm = ColAutomata(supervised=True, indices=indices, g=g)

            cur_row = AdjRow(i, self.directed, self.self_loop, col_range=col_range)
            lb = 0 if lb_list is None else lb_list[i]
            ub = cur_row.root.n_cols if ub_list is None else ub_list[i]
            mask = generate_square_subsequent_mask(i + 1).to(gnn_embeds.device)
            if self.use_st_attn:
                new_h = self.decoder(h.unsqueeze(0), gnn_embeds, tgt_mask=mask)
            else:
                new_h = self.decoder(h.unsqueeze(0), mask=mask)
                new_h = torch.cat([new_h[:, [-1]], gnn_embeds[:, [i]]], dim=2)
                new_h = self.fuser(new_h)
            #new_h = new_h.squeeze(0)
            new_h = new_h[0, [-1]]  # Get the last value
            controller_state = (new_h, c)
            controller_states.append(controller_state)

            ll, state_bot, _ = self.gen_row(0, controller_state, cur_row.root, col_sm, lb, ub)
            assert lb <= len(col_sm.indices) <= ub
            next_h = state_bot[0] + self.row_pos_enc([num_nodes - (i + 1)])
            h = torch.concat([h, next_h])
            c = state_bot[1]

            edges += [(i, x, w) for x, w in col_sm.indices]
            total_ll = total_ll + ll

        # c0 = [s[0] for s in controller_states]
        # c0 = torch.concat(c0).detach().cpu().numpy()
        # h0 = h.detach().cpu().numpy()
        return total_ll, edges, _#self.row_tree.list_states

    def _smooth_labels(self, labels):
        if self.label_smoothing == 0:
            return labels
        else:
            return (1.0 - self.label_smoothing) * labels + (0.5 * self.label_smoothing)

    def binary_ll(self, pred_logits, np_label, need_label=False, reduction='sum'):
        pred_logits = pred_logits.view(-1, 1)
        label = torch.tensor(np_label, dtype=torch.float32).to(pred_logits.device).view(-1, 1)
        # label = self._smooth_labels(label)
        loss = F.binary_cross_entropy_with_logits(pred_logits, label, reduction=reduction)
        if need_label:
            return -loss, label
        return -loss

    def categorical_ll(self, logits, np_label, reduction='sum'):
        pred_logits = logits.view(-1, 3)
        label = torch.tensor(np_label, dtype=torch.long).to(pred_logits.device)
        loss = F.cross_entropy(logits, label, reduction=reduction)
        return -loss

    def forward_row_trees(self, graph_ids, list_node_starts=None, num_nodes=-1, list_col_ranges=None):
        TreeLib.PrepareMiniBatch(graph_ids, list_node_starts, num_nodes, list_col_ranges)
        # embed trees
        all_ids = TreeLib.PrepareTreeEmbed()

        if not self.bits_compress:
            h_bot = torch.cat([self.empty_h0, self.leaf_h0_pos, self.leaf_h0_neg], dim=0)
            c_bot = torch.cat([self.empty_c0, self.leaf_c0_pos, self.leaf_c0_neg], dim=0)
            fn_hc_bot = lambda d: (h_bot, c_bot)
        else:
            binary_embeds, base_feat = TreeLib.PrepareBinary()
            fn_hc_bot = lambda d: (binary_embeds[d], binary_embeds[d]) if d < len(binary_embeds) else base_feat
        max_level = len(all_ids) - 1
        h_buf_list = [None] * (len(all_ids) + 1)
        c_buf_list = [None] * (len(all_ids) + 1)
        # All ids contains, for each level in the tree, the left and right states to select.
        # These are split into two-types, which are referred to in code as 'bot' and 'prev'.
        # Bot indicates the node is a leaf, in which case the value will be selected from h_bot from {0, 1, 2}
        # representing the parameters for {0, 1, -1}. Prev represents an existing embedding in h_buf_list, which
        # have been combined already by the TreeLSTM cells, so these prev indices start appearing above the lowest
        # level of the tree.
        for d in range(len(all_ids) - 1, -1, -1): # Work from bottom of tree up.
            fn_ids = lambda i: all_ids[d][i]
            if d == max_level:
                h_buf = c_buf = None
            else:
                h_buf = h_buf_list[d + 1]
                c_buf = c_buf_list[d + 1]
            h_bot, c_bot = fn_hc_bot(d + 1)
            new_h, new_c = batch_tree_lstm2(h_bot, c_bot, h_buf, c_buf, fn_ids, self.lr2p_cell)
            h_buf_list[d] = new_h
            c_buf_list[d] = new_c
        return fn_hc_bot, h_buf_list, c_buf_list

    def forward_row_summaries(self, graph_ids, list_node_starts=None, num_nodes=-1, prev_rowsum_states=[None, None], list_col_ranges=None):
        fn_hc_bot, h_buf_list, c_buf_list = self.forward_row_trees(graph_ids, list_node_starts, num_nodes, list_col_ranges)
        row_states, next_states = self.row_tree.forward_train(*(fn_hc_bot(0)), h_buf_list[0], c_buf_list[0], *prev_rowsum_states)
        return row_states, next_states

    def forward_row(self, h_bot, c_bot, h_buf0, c_buf0, gnn_embeds, num_nodes):
        ids = TreeLib.PrepareRowIndices()
        ids_fn = lambda : ids
        # Add in the SOS token for transformer.
        h_bot_ = torch.concat([self.init_h0, h_bot])
        c_bot_ = torch.concat([self.init_c0, c_bot])
        # Fill in the SOS and non-zero rows.
        new_s = tree_state_select(h_bot_, c_bot_, h_buf0, c_buf0, ids_fn)
        # Add row positional encodings.
        row_pos = [num_nodes - (i % num_nodes) for i in range(new_s[0].shape[0])]
        h = new_s[0] + self.row_pos_enc(row_pos)
        mask = generate_square_subsequent_mask(num_nodes).to(gnn_embeds.device)
        # Put into batch form for decoder.
        h = h.reshape(-1, num_nodes, self.embed_dim)
        if self.use_st_attn:
            gnn_embeds = gnn_embeds.reshape(-1, num_nodes, self.embed_dim)
            # decoder does both source-target and self attention.
            h = self.decoder(h, gnn_embeds, tgt_mask = mask)
            h = h.reshape(-1, self.embed_dim)
        else:
            h = self.decoder(h, mask=mask)
            h = h.reshape(-1, self.embed_dim)
            h = self.fuser(torch.concat([h, gnn_embeds], dim=1))

        # s0 = new_s[0].detach().cpu().numpy()
        # h0 = new_h.detach().cpu().numpy()
        return (h, new_s[1])

    def _predict_leaves(self, lr, lv, states, get_idx=False):
        leaf_ll = 0
        has_leaf = TreeLib.GetLeafMask(lr, lv)
        if has_leaf is not None and np.sum(has_leaf) > 0:
            leaf_states = states[has_leaf]
            leaf_logits = self.pred_sign(leaf_states)
            leaf_labels = TreeLib.GetLeafLabels(lr, lv)
            # leaf_labels[leaf_labels < 0] = 2  # Change to 2 for cross entropy
            leaf_labels = np.abs(leaf_labels)
            # print(f'-- Depth: {lv + 1} | lr : {lr}  --')
            # print(torch.softmax(leaf_logits, dim=1).detach().cpu().numpy())
            # print()
            leaf_ll = self.binary_ll(leaf_logits, leaf_labels, reduction='sum')
            if get_idx:
                leaf_idx = np.ones(len(has_leaf))
                leaf_idx[has_leaf] = leaf_labels + 1
                return leaf_ll, leaf_idx
        return leaf_ll, np.ones(len(has_leaf))

    def forward_train(self, graph_ids, gnn_embeds, n,
                      list_node_starts=None,
                      num_nodes=-1, prev_rowsum_states=[None, None], list_col_ranges=None):
        fn_hc_bot, h_buf_list, c_buf_list = self.forward_row_trees(graph_ids, list_node_starts, num_nodes, list_col_ranges)
        # row states are collapsed across the batch, so (N * n) * H - NOTE: Next states not used in this function -
        # tree_init = gnn_embeds.reshape(-1, n, self.embed_dim).sum()
        row_states = self.forward_row(*(fn_hc_bot(0)), h_buf_list[0], c_buf_list[0], gnn_embeds, n)
        ll = 0
        has_ch, _ = TreeLib.GetChLabel(0, dtype=np.bool) # Get correct labels
        # We want to predict whether to descent further for all rows that are not leaves.
        # For the leaf nodes we want to do a softmax prediction.
        logit_has_edge = self.pred_has_ch(row_states[0])
        # is_leaf = torch.tensor(~TreeLib.GetLeafMask(0, 0), dtype=torch.float32).to(logit_has_edge.device)
        has_ch_ll = self.binary_ll(logit_has_edge, has_ch, reduction='none').squeeze() # * is_leaf
        ll = ll + torch.sum(has_ch_ll)
        # leaf prediction.
        # root_leaf_ll, _ = self._predict_leaves(0, 0, row_states[0])
        # ll = ll + root_leaf_ll
        # Remove all the empty rows.
        cur_states = (row_states[0][has_ch], row_states[1][has_ch])

        lv = 0
        while True:  # Descent the tree (across batches), starting at the root. Compute top-down state and make preds.
            # Leaf nodes have already been handled, so they are removed here..
            is_nonleaf = TreeLib.QueryNonLeaf(lv)
            if is_nonleaf is None or np.sum(is_nonleaf) == 0:
                break
            cur_states = (cur_states[0][is_nonleaf], cur_states[1][is_nonleaf])

            # Make continuation predictions for those that don't have leaves to the left.
            left_logits = self.pred_has_left(cur_states[0], lv)
            has_left, num_left = TreeLib.GetChLabel(-1, lv)
            left_ll, float_has_left = self.binary_ll(left_logits, has_left, need_label=True) #, reduction='none')
            # TODO: update the has_left to handle leaf signs.
            # has_left_leaf_mask = torch.tensor(~TreeLib.GetLeafMask(-1, lv), dtype=torch.float32).to(left_logits.device)
            # Zero out those which have leaves to the left
            ll = ll + torch.sum(left_ll) #.squeeze() * has_left_leaf_mask)
            # Left leaf prediction.
            # left_leaf_ll, leaf_idx = self._predict_leaves(-1, lv, cur_states[0], get_idx=True)
            # ll = ll + left_leaf_ll

            left_update = self.topdown_left_embed[has_left] + self.tree_pos_enc(num_left)
            cur_states = self.cell_topdown(left_update, cur_states, lv)

            # Get the bottom up states for the left children of all nodes on this level
            left_ids = TreeLib.GetLeftRootStates(lv)
            h_bot, c_bot = fn_hc_bot(lv + 1)
            if lv + 1 < len(h_buf_list):
                h_next_buf, c_next_buf = h_buf_list[lv + 1], c_buf_list[lv + 1]
            else:
                h_next_buf = c_next_buf = None
            left_subtree_states = tree_state_select(h_bot, c_bot,
                                                    h_next_buf, c_next_buf,
                                                    lambda: left_ids)

            # Merge the bottom up states from the left children with the current topdown state
            has_right, num_right = TreeLib.GetChLabel(1, lv)
            right_pos = self.tree_pos_enc(num_right)
            left_subtree_states = [x + right_pos for x in left_subtree_states]
            topdown_state = self.l2r_cell(cur_states, left_subtree_states, lv)

            # Use the merged topdown state to predict the right child.
            right_logits = self.pred_has_right(topdown_state[0], lv)
            right_update = self.topdown_right_embed[has_right]
            topdown_state = self.cell_topright(right_update, topdown_state, lv)
            # The reason for multiplying by left is as follows: if still descending at this point, the tree must have
            # an edge of some sort. It could have multiple. So if it has left and right children, we need to make
            # predictions for both of them. If it doesn't have a left child, it must have a right child, therefore
            # we will always sample a right child and don't need a prediction there. (this only affects training,
            # at test time we just make predictions for both).
            # has_right_leaf_mask = torch.tensor(~TreeLib.GetLeafMask(1, lv), dtype=torch.float32).to(right_logits.device)
            right_ll = self.binary_ll(right_logits, has_right, reduction='none') * float_has_left
            # I want to zero out all the logits that are guaranteed by has_left, and I also want to zero out
            # all the logits that have a right leaf.
            ll = ll + torch.sum(right_ll) #.squeeze() * has_right_leaf_mask)

            # Right leaf label prediction.
            # right_leaf_ll, _ = self._predict_leaves(1, lv, topdown_state[0])
            # ll = ll + right_leaf_ll

            lr_ids = TreeLib.GetLeftRightSelect(lv, np.sum(has_left), np.sum(has_right))
            new_states = []
            for i in range(2): # h and c
                new_s = multi_index_select([lr_ids[0], lr_ids[2]], [lr_ids[1], lr_ids[3]],
                                            cur_states[i], topdown_state[i])
                new_states.append(new_s)
            cur_states = tuple(new_states)
            # print(f'll: {ll} | lv: {lv}')
            lv += 1

        return ll, _
