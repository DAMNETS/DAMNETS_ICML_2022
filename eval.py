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
import argparse
from utils.graph_utils import load_graph_ts
from easydict import EasyDict as edict
import yaml
from utils.eval_helper import degree_stats, clustering_stats, spectral_stats
from utils.dist_helper import mmd_rbf
from tqdm import tqdm
import scipy.sparse as sp


def get_degree_seq(g_list):
    return sorted([d for G in g_list for n, d in G.degree()], reverse=True)


def get_final_degrees_df(ts, name, incr=-1):
    last_graphs = [t[incr] for t in ts]
    return pd.DataFrame({name: get_degree_seq(last_graphs)})


def make_ba_plots(sampled_ts, test_ts, model_name, save_dir):
    plt.figure(0)
    df = pd.concat([get_final_degrees_df(sampled_ts, model_name, incr=-1),
                    get_final_degrees_df(test_ts, 'Test', incr=-1)], axis=1)
    ax = df.plot(kind='density', xlim=(0, 25), title='Degree Distribution of $G_T$')
    ax.set_xlabel('Degree')
    save_path = os.path.join(save_dir, 'ba_degree.pdf')
    plt.savefig(save_path, format='pdf', dpi=1200, bbox_inches='tight')
    plt.close()


def make_three_comm_plots(sampled_ts, test_ts, model_name, save_dir):
    c_sizes = [int(test_ts[0][0].number_of_nodes() / 3)] * 3
    comms = np.cumsum(c_sizes)
    fig, axs = plt.subplots(1, 3, figsize=(10, 3), sharey=True)
    plt.tight_layout()
    plot_community_eval(sampled_ts, test_ts, range(comms[1], comms[2]), axs[0], 'Third (Decaying)', model_name)
    plot_community_eval(sampled_ts, test_ts, range(comms[0]), axs[1], 'First', model_name)
    plot_community_eval(sampled_ts, test_ts, range(comms[0], comms[1]), axs[2], 'Second', model_name)
    axs[0].set_ylabel('Density')
    save_path = os.path.join(save_dir, '3_comm_densities.pdf')
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


def s_catch(g, s_fun=None):
    try:
        res = s_fun(g)
    except ZeroDivisionError:
        res = 0
    return res


def compute_mean_std_stat(ts_list, s_fun, n_workers=8):
    ''' Compute the marginal mean and std of the summary statistic s_fun across time '''
    T = len(ts_list[0])
    mean = np.zeros(T)
    std = np.zeros(T)
    raw = []
    fun = partial(s_catch, s_fun=s_fun)
    with multiprocessing.Pool(n_workers) as p:
        for t in range(T):
            stats = p.map(fun, [ts[t] for ts in ts_list])
            mean[t] = np.mean(stats)
            std[t] = np.std(stats)
            raw.append(stats)
    return mean, std, raw


def plot_statistic_density(mean, std, ax, label, col=None, fill=True):
    if col is None:
        ax.plot(mean, label=label)
        if fill:
            ax.fill_between([t for t in range(len(mean))], mean - std, mean + std, alpha=0.3)
    else:
        ax.plot(mean, label=label, color=col)
        if fill:
            ax.fill_between([t for t in range(len(mean))], mean - std, mean + std, color=col, alpha=0.3)
    return mean, std


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


def plot_network_statistics(stats, save_dir=''):
    model_names = stats.columns.get_level_values(0).unique()
    stat_names = stats.columns.get_level_values(1).unique()
    fig, axs = plt.subplots(1, len(stat_names), figsize=(15, 3))
    for idx, statistic in enumerate(stat_names):
        stat_data = stats.xs(statistic, level=1, axis=1)
        for model in model_names:
            model_data = stat_data[model]
            plot_statistic_density(pd.to_numeric(model_data['mean']), pd.to_numeric(model_data['std']), axs[idx], model)
        axs[idx].set_title(statistic)
        axs[idx].legend(prop={'size': 6})
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'network_statistics.pdf')
    plt.savefig(save_path, format='pdf', dpi=1200, bbox_inches='tight')

def statistics_compute_cpl(G):
    """Compute characteristic path length."""
    A = nx.to_numpy_array(G)
    P = sp.csgraph.shortest_path(sp.csr_matrix(A))
    return P[((1 - np.isinf(P)) * (1 - np.eye(P.shape[0]))).astype(np.bool)].mean()


def compute_network_statistics(ts_list, model_name):
    s_funs = {'Density': nx.density,
              'Clustering' : nx.average_clustering,
              'Transitivity': nx.transitivity,
              'Assortativity': nx.degree_assortativity_coefficient,
              'CP': statistics_compute_cpl,}
              # 'Claw': statistics_claw_count}
    dfs = []
    print('Computing statistics for model', model_name)
    for stat_name, stat in s_funs.items():
        print(f'Computing {stat_name}')
        mean, std, raw = compute_mean_std_stat(ts_list, stat)
        index = pd.MultiIndex.from_tuples([(model_name, stat_name, 'mean'),
                                           (model_name, stat_name, 'std'),
                                           (model_name, stat_name, 'raw')])
        dfs.append(pd.DataFrame([mean, std, raw], index=index, dtype=np.float).T)
    return pd.concat(dfs, axis=1)


def compute_spectral_bipartivity(ts_list):
    print('Computing Spectral Bipartivity')
    stat_name = 'SB'
    stat = nx.bipartite.spectral_bipartivity
    mean, std, raw = compute_mean_std_stat(ts_list, stat)
    index = pd.MultiIndex.from_tuples([(model_name, stat_name, 'mean'),
                                       (model_name, stat_name, 'std'),
                                       (model_name, stat_name, 'raw')])
    return pd.DataFrame([mean, std, raw], index=index, dtype=np.float).T


def compute_local_mmds(sampled_ts, test_ts, model_name):
    deg_mmd = []
    clust_mmd = []
    spec_mmd = []
    print('Computing Local MMDs')
    for t in tqdm(range(1, len(sampled_ts[0]))):
        ## Get the graphs at this time slice
        sampled = [ts[t] for ts in sampled_ts]
        test = [ts[t] for ts in test_ts]
        ## Compute MMD between the slices
        deg_mmd.append(degree_stats(test, sampled))
        clust_mmd.append(clustering_stats(test, sampled))
        spec_mmd.append(spectral_stats(test, sampled))
    index = pd.MultiIndex.from_tuples(
        [(model_name, 'Degree_MMD'), (model_name, 'Clustering_MMD'), (model_name, 'Spectral_MMD')])
    mmds = np.array([deg_mmd, clust_mmd, spec_mmd]).T

    df = pd.DataFrame(mmds, columns=index)
    return df


def to_numpy_batch(list):
    ''' Convert a list of length N into a numpy array of size N * 1 '''
    return np.expand_dims(np.array(list), axis=1)


def mmd_from_multi(test_df, model_df, statistic_name):
    ''' Compute the MMD from a df with a single row (time slice), multi-index column as Model/Statistic/(mean/std/raw) '''
    model_raw = to_numpy_batch(model_df[statistic_name, 'raw'])
    test_raw = to_numpy_batch(test_df[statistic_name, 'raw'])
    try:
        mmd = mmd_rbf(model_raw, test_raw)
    except:
        mmd = np.nan
    return mmd

def compute_global_mmds(stats):
    dfs = []
    for model in stats.drop('Test', axis=1).columns.get_level_values(0).unique():
        trans_mmd = []
        assort_mmd = []
        cp_mmd = []
        for t in tqdm(range(1, len(stats.index))):
            test_stats_t = stats.loc[t]['Test']
            model_stats_t = stats.loc[t][model]
            trans_mmd.append(mmd_from_multi(test_stats_t, model_stats_t, 'Transitivity'))
            assort_mmd.append(mmd_from_multi(test_stats_t, model_stats_t, 'Assortativity'))
            cp_mmd.append(mmd_from_multi(test_stats_t, model_stats_t, 'CP'))
        index = pd.MultiIndex.from_tuples([(model, 'Trans_MMD'), (model, 'Assort_MMD'), (model, 'CP_MMD')])
        mmds = np.array([trans_mmd, assort_mmd, cp_mmd]).T
        dfs.append(pd.DataFrame(mmds, columns=index))
    return pd.concat(dfs, axis=1)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="A script to compute and plot the metrics for evaluation of generated network time series.")
    parser.add_argument(
        '-s',
        '--sampled_path',
        type=str,
        help="The path to the generated network time series. If left blank, will load from the previous run.",
        default=''
    )
    parser.add_argument(
        '-t',
        '--test_path',
        type=str,
        help="The path to the test network time series. If left blank, will load from the previous run.",
        default=''
    )

    parser.add_argument(
        '-d',
        '--dataset_name',
        type=str,
        help='The name of the dataset, for producing dataset specific plots. If none given,'
             'will only produce generic plots.'
    )
    parser.add_argument(
        '-m',
        '--model_name',
        type=str,
        help='The name of the model that generated the samples (not needed for DAMNETS, only used for labelling',
        default=''
    )
    parser.add_argument(
        '-o',
        '--output_dir',
        type=str,
        help='The directory to save outputs to.'
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    c_args = parse_arguments()
    if c_args.sampled_path == '':
        print('No path to graphs given, loading previous test run.')
        with open('experiment_files/last_test.txt', 'r') as f:
            test_dir = f.readline()
        output_dir = test_dir
        ## Load the sampled time series.
        sampled_path = os.path.join(test_dir, 'sampled_ts.pkl')
        config_file = os.path.join(test_dir, 'config.yaml')
        config = edict(yaml.full_load(open(config_file, 'r')))
        model_name = config.model.name
        ## Load the test time series.
        test_path = os.path.join(config.data_path, config.dataset_name) + '_test_graphs.pkl'
        dataset_name = config.dataset_name
    else:
        sampled_path = c_args.sampled_path
        test_path = c_args.test_path
        model_name = c_args.model_name
        dataset_name = c_args.dataset_name
        output_dir = c_args.output_dir
    sampled_ts = load_graph_ts(sampled_path)
    test_ts = load_graph_ts(test_path)
    ## Compute the network statistics
    sampled_stats = compute_network_statistics(sampled_ts, model_name)
    test_stats = compute_network_statistics(test_ts, 'Test')
    stats = pd.concat([sampled_stats, test_stats], axis=1)
    print(stats)
    ## Plot the network statistics in time.
    plot_network_statistics(stats, save_dir=output_dir)
    ## Compute the MMDs.
    local_mmds = compute_local_mmds(sampled_ts, test_ts, model_name)
    global_mmds = compute_global_mmds(stats)
    mmds = pd.concat([local_mmds, global_mmds], axis=1)
    print('------ Raw MMDS ------')
    print(mmds)
    print('------ Mean MMDS ------')
    print(mmds.mean())
    ## Produce dataset-specific plots.
    if dataset_name == '3_comm_decay':
        make_three_comm_plots(sampled_ts, test_ts, model_name, save_dir=output_dir)
    elif dataset_name == 'ba':
        make_ba_plots(sampled_ts, test_ts, model_name, save_dir=output_dir)
    elif dataset_name == 'bipartite_contraction':
        sb = compute_spectral_bipartivity(sampled_ts)
        print(sb)
        stats = pd.concat([stats, sb], axis=1)
    ## Save dataframes.
    stats.to_pickle(path=os.path.join(output_dir, 'stats.pkl'))
    mmds.to_pickle(path=os.path.join(output_dir, 'mmds.pkl'))




