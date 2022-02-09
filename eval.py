import networkx as nx
from utils import graph_utils
import matplotlib.pyplot as plt
import numpy as np
from utils.arg_helper import get_config
import os
import collections
import pandas as pd
import multiprocessing
from functools import partial


def get_degree_seq(g_list):
    return sorted([d for G in g_list for n, d in G.degree()], reverse=True)


def get_final_degrees_df(ts, name, incr=-1):
    last_graphs = [t[incr] for t in ts]
    return pd.DataFrame({name: get_degree_seq(last_graphs)})


def ba_plots(sampled_ts, test_ts, args, tag):
    plt.figure(0)
    df = pd.concat([get_final_degrees_df(sampled_ts, args.model.name, incr=-1),
                    get_final_degrees_df(test_ts, 'Test', incr=-1)], axis=1)
    ax = df.plot(kind='density', xlim=(0, 25), title='Degree Distribution of $G_T$')
    ax.set_xlabel('Degree')
    save_path = os.path.join(args.save_dir, f'{tag}_ba_degree.pdf')
    plt.savefig(save_path, format='pdf', dpi=1200, bbox_inches='tight')
    plt.close()

def three_comm_plots(sampled_ts, test_ts, args, tag):
    comms = np.cumsum(args.dataset.c_sizes)
    fig, axs = plt.subplots(1, 3, figsize=(10, 3), sharey=True)
    plt.tight_layout()
    plot_community_eval(sampled_ts, test_ts, range(comms[1], comms[2]), axs[0], 'Third (Decaying)', args.model.name)
    plot_community_eval(sampled_ts, test_ts, range(comms[0]), axs[1], 'First', args.model.name)
    plot_community_eval(sampled_ts, test_ts, range(comms[0], comms[1]), axs[2], 'Second', args.model.name)
    axs[0].set_ylabel('Density')
    # writer.add_figure('Community Densities', fig)
    save_path = os.path.join(args.save_dir, f'{tag}_comm_densities.pdf')
    plt.savefig(save_path, format='pdf', dpi=1200, bbox_inches='tight')

def plot_density(ts, subgraph_nodes, ax, label, col=None):
    T = len(ts[0])
    sample_mean = np.zeros(T)
    sample_std = np.zeros(T)

    for t in range(T):
        densities = [nx.density(nx.subgraph(ts[b][t], subgraph_nodes)) for b in range(len(ts))]
        sample_mean[t] = np.mean(densities)
        sample_std[t] = np.std(densities)
    if col is None:
        ax.plot(sample_mean, label=label)
        ax.fill_between([t for t in range(T)], sample_mean - sample_std, sample_mean + sample_std, alpha=0.1)
    else:
        ax.plot(sample_mean, label=label, color=col)
        ax.fill_between([t for t in range(T)], sample_mean - sample_std, sample_mean + sample_std, color=col, alpha=0.1)

def plot_community_eval(sampled_ts, true_ts, subgraph_nodes, ax, label, model_name, ylim=None):
    T = len(sampled_ts[0])
    density_true = np.zeros(T)
    for t in range(T):
        density_true[t] = np.mean([nx.density(nx.subgraph(true_ts[b][t], subgraph_nodes))
                                   for b in range(len(true_ts))])
    ax.plot(density_true, color='black', label='Test')
    ax.set_xlabel('Time')
    plot_density(sampled_ts, subgraph_nodes, ax, model_name)
    # plot_density(tg_ts, subgraph_nodes, ax, 'TagGen')
    # plot_density(dymond_ts, subgraph_nodes, ax, 'Dymond')
    # ax.set_ylabel('Density')
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_title('{}'.format(label))
    ax.legend()

def plot_network_statistics(test_ts, sampled_ts, args, tag):
    fig, axs = plt.subplots(1, 3, figsize=(10, 3))
    plot_statistic_densities(nx.number_of_edges, test_ts, sampled_ts, axs[0], 'Number of Edges', args.model.name)
    plot_statistic_densities(nx.average_clustering, test_ts, sampled_ts, axs[1], 'Clustering Coefficient', args.model.name)
    plot_statistic_densities(nx.transitivity, test_ts, sampled_ts, axs[2], 'Transitivity', args.model.name)
    save_path = os.path.join(args.save_dir, f'{tag}_network_statistics.pdf')
    plt.savefig(save_path, format='pdf', dpi=1200, bbox_inches='tight')

def s_catch(g, s_fun=None):
    try:
        res = s_fun(g)
    except ZeroDivisionError:
        res = 0
    return res


def plot_statistic_density(ts_list, s_fun, ax, label, col=None, n_workers=8, fill=True):
    T = len(ts_list[0])
    mean = np.zeros(T)
    std = np.zeros(T)

    fun = partial(s_catch, s_fun=s_fun)
    with multiprocessing.Pool(n_workers) as p:
        for t in range(T):
            stats = p.map(fun, [ts[t] for ts in ts_list])
            mean[t] = np.mean(stats)
            std[t] = np.std(stats)

    if col is None:
        ax.plot(mean, label=label)
        if fill:
            ax.fill_between([t for t in range(T)], mean - std, mean + std, alpha=0.3)
    else:
        ax.plot(mean, label=label, color=col)
        if fill:
            ax.fill_between([t for t in range(T)], mean - std, mean + std, color=col, alpha=0.3)


def plot_statistic_densities(s_fun, test_ts, sampled_ts, ax, s_name, model_name):
    plot_statistic_density(test_ts, s_fun, ax, 'Test', 'black')
    plot_statistic_density(sampled_ts, s_fun, ax, model_name)
    ax.set_xlabel('Time')
    ax.set_title(s_name)
    ax.legend(prop={'size': 6})


def plot_top_N(ts, args, tag, N=5):
    T = len(ts[0])
    N = min(N, len(ts))
    fig, axs = plt.subplots(N, T, figsize=(20, 20))
    cols = ['red', 'blue']
    for n in range(N):
        for t in range(T):
            nx.draw(ts[n][t], ax=axs[n, t], node_size=10, node_color=cols[(n + t)%2])
    save_path = os.path.join(args.save_dir, f'{tag}_network_vis.pdf')
    plt.savefig(save_path, format='pdf', dpi=1200, bbox_inches='tight')
