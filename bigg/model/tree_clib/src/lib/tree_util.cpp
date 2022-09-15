#include <iostream>
#include <algorithm>
#include <cassert>

#include "config.h"  // NOLINT
#include "tree_util.h"  // NOLINT
#include "struct_util.h"  // NOLINT


AdjNode::AdjNode(AdjNode* parent, int row, int col_begin, int col_end,
                 int depth)
{
    this->init(parent, row, col_begin, col_end, depth);
}

AdjNode::~AdjNode()
{
    if (this->lch != nullptr)
        delete this->lch;
    if (this->rch != nullptr)
        delete this->rch;
}

void AdjNode::init(AdjNode* parent, int row, int col_begin, int col_end,
                   int depth)
{
    this->lch = nullptr;
    this->rch = nullptr;
    this->parent = parent;
    this->row = row;
    this->col_begin = col_begin;
    this->col_end = col_end;
    this->depth = depth;
    this->mid = (col_begin + col_end) / 2;
    this->n_cols = col_end - col_begin;
    this->is_lowlevel = this->n_cols <= cfg::bits_compress;
    this->is_leaf = (this->n_cols <= 1);
    this->is_root = (this->parent == nullptr);
    if (is_lowlevel) {
         this->bits_rep_pos = BitSet(cfg::bits_compress);
         this->bits_rep_neg = BitSet(cfg::bits_compress);
    }
    this->has_edge = false;
    this->job_idx = -1;
    this->weight = 0;
}

void AdjNode::update_bits(int weight = 0)
{
    if (!is_lowlevel)
        return;
    if (is_leaf)
    {
        if (has_edge)
        {
            assert(weight != 0);
            if (weight > 0) {
                bits_rep_pos.set(0);
            }
            else {
                bits_rep_neg.set(0);
            }
        }

    } else {
        bits_rep_pos = lch->bits_rep_pos.left_shift(rch->n_cols);
        bits_rep_pos = bits_rep_pos.or_op(rch->bits_rep_pos);
        bits_rep_neg = lch->bits_rep_neg.left_shift(rch->n_cols);
        bits_rep_neg = bits_rep_neg.or_op(rch->bits_rep_neg);
    }
}

void AdjNode::split()
{
    if (this->lch != nullptr && this->rch != nullptr)
        return;
    if (this->is_leaf)
        return;
    this->lch = node_holder.get_pt(this, row, col_begin, mid, depth + 1);
    this->rch = node_holder.get_pt(this, row, mid, col_end, depth + 1);
}

AdjRow::AdjRow(int row, int col_start, int col_end)
{
    init(row, col_start, col_end);
}

AdjRow::~AdjRow()
{
    if (this->root != nullptr)
        delete this->root;
}

void AdjRow::init(int row, int col_start, int col_end)
{
    this->row = row;
    assert(!cfg::directed);
    int max_col = row;
    if (cfg::self_loop)
        max_col += 1;
    if (col_start < 0 || col_end < 0)
    {
        col_start = 0;
        col_end = max_col;
    }
    this->root = node_holder.get_pt(nullptr, row, col_start, col_end, 0);
}


void AdjRow::insert_edges(std::vector<std::pair<int, int> >& edge_list, std::vector<int>& prev_row)
{
    auto* col_sm = new ColAutomata(edge_list, prev_row);
    this->add_edges(this->root, col_sm);
    delete col_sm;
}

void AdjRow::add_edges(AdjNode* node, ColAutomata* col_sm)
{
    if (node->is_root)
    {
        node->has_edge = col_sm->num_indices > 0;
        job_collect.has_ch.push_back(node->has_edge);
        // Handle the edge case where the root is a leaf.
        int is_root_leaf = node->is_leaf;
        if (!is_root_leaf || node->row == 0) {
            job_collect.is_root_del_leaf.push_back(false);
            job_collect.is_root_add_leaf.push_back(false);
        }
        else { // Not the first row, but root is a leaf node.
            int weight;
            if (node->has_edge)
                weight = col_sm->add_edge(node->col_begin);
            else
                weight = 0;
            node->update_bits(weight);
            node->weight = weight;
            bool had_edge = col_sm->had_edge(node->col_begin);
            if (had_edge) {
                job_collect.root_del_weights.push_back(weight);
                job_collect.is_root_del_leaf.push_back(true);
                job_collect.is_root_add_leaf.push_back(false);
            } else {
                job_collect.root_add_weights.push_back(weight);
                job_collect.is_root_del_leaf.push_back(false);
                job_collect.is_root_add_leaf.push_back(true);
            }

        }
    } else {
        node->has_edge = true;
    }
    if (!node->has_edge) // Remove empty nodes.
        return;
    job_collect.append_bool(job_collect.is_internal, node->depth,
                            !(node->is_leaf));
    if (node->is_leaf) {
        if (!node->is_root) {
            int weight = col_sm->add_edge(node->col_begin);
            assert(weight != 0);
            node->update_bits(weight);
            node->weight = weight;
//        if (node->is_root) { // Still want to make sign predictions for roots that happen to be leaves.
//           assert(node->has_edge);
//           job_collect.root_weights.push_back(weight);
//        }
        }
    } else {
        node->split();
        // Is there a child somewhere to the left.
        bool has_left = (col_sm->next_edge() < node->mid);
        if (has_left)
            this->add_edges(node->lch, col_sm);
        job_collect.append_bool(job_collect.has_left, node->depth, has_left);
        job_collect.append_bool(job_collect.num_left, node->depth,
                                node->lch->n_cols);
        // lch and rch are always made. So we can run this regardless.
        // Is the lch a leaf. If it's not but no edges were added to it, add_edges is never called on it
        // so the tree doesn't descend that far anyway. So these are only leaves reached via ML training.
        int left_leaf_weight = node->lch->weight; // Can be -1, 0, 1
        bool has_left_leaf = node->lch->is_leaf;
        if(!has_left_leaf) {
            job_collect.append_bool(job_collect.has_left_add_leaf, node->depth, false);
            job_collect.append_bool(job_collect.has_left_del_leaf, node->depth, false);
        }
        if (has_left_leaf) {
            if (col_sm->had_edge(node->lch->col_begin)) { // Had an edge, can only be -1 or 0
                assert(left_leaf_weight <= 0);
                job_collect.append_bool(job_collect.left_del_weights, node->depth, left_leaf_weight);
                job_collect.append_bool(job_collect.has_left_add_leaf, node->depth, false);
                job_collect.append_bool(job_collect.has_left_del_leaf, node->depth, true);
            }
            else { // Didn't have an edge, so can only be 1 or 0
                assert(left_leaf_weight >= 0);
                job_collect.append_bool(job_collect.left_add_weights, node->depth, left_leaf_weight);
                job_collect.append_bool(job_collect.has_left_add_leaf, node->depth, true);
                job_collect.append_bool(job_collect.has_left_del_leaf, node->depth, false);
            }
        }

        bool has_right = has_left ?
            col_sm->has_edge(node->mid, node->col_end) : true; // Know it has edge, not in left => in right.
        if (has_right)
            this->add_edges(node->rch, col_sm);
        job_collect.append_bool(job_collect.has_right, node->depth, has_right);
        job_collect.append_bool(job_collect.num_right, node->depth,
                                node->rch->n_cols);
        int right_leaf_weight = node->rch->weight;
        bool has_right_leaf = node->rch->is_leaf;
        // We don't need to do any prediction if it doesn't have left (as it has an edge).
        if(!has_right_leaf || (has_right_leaf && !has_left)) {
            job_collect.append_bool(job_collect.has_right_add_leaf, node->depth, false);
            job_collect.append_bool(job_collect.has_right_del_leaf, node->depth, false);
        }
//        job_collect.append_bool(job_collect.has_right_leaf, node->depth, has_right_leaf);
        else {
            assert(has_right_leaf && has_left);
            if (col_sm->had_edge(node->rch->col_begin)) { // Can only be deletion.
                assert(right_leaf_weight <= 0);
                job_collect.append_bool(job_collect.right_del_weights, node->depth, right_leaf_weight);
                job_collect.append_bool(job_collect.has_right_add_leaf, node->depth, false);
                job_collect.append_bool(job_collect.has_right_del_leaf, node->depth, true);
            }
            else { // Can only be addition.
                assert(right_leaf_weight >= 0);
                job_collect.append_bool(job_collect.right_add_weights, node->depth, right_leaf_weight);
                job_collect.append_bool(job_collect.has_right_add_leaf, node->depth, true);
                job_collect.append_bool(job_collect.has_right_del_leaf, node->depth, false);
            }
        }
        node->update_bits();
        node->job_idx = job_collect.add_job(node);

        int cur_idx = (int)job_collect.has_left[node->depth].size() - 1;
        auto* ch = node->lch;
        if (ch->has_edge && !ch->is_leaf && !ch->is_lowlevel)
        {
            int pos = job_collect.job_position[ch->job_idx];
            job_collect.append_bool(job_collect.next_left_froms, node->depth,
                                    pos);
            job_collect.append_bool(job_collect.next_left_tos, node->depth,
                                    cur_idx);
        } else {
            int bid = -1;
            if (ch->has_edge && !ch->is_leaf) {
                bid = 2 + job_collect.job_position[ch->job_idx];
                std::cout << "2+ triggered in add edges" << std::endl;
            }

            else if (ch->is_leaf && ch->has_edge) {
                assert(ch->weight != 0);
                bid = (ch->weight > 0) ? 1 : 2;
            }
            else {
                assert(ch->weight == 0);
                bid = 0;
            }
            assert(bid != -1);
            job_collect.append_bool(job_collect.bot_left_froms, node->depth,
                                    bid);
            job_collect.append_bool(job_collect.bot_left_tos, node->depth,
                                    cur_idx);
        }
    }
}


PtHolder<AdjNode> node_holder;
PtHolder<AdjRow> row_holder;
