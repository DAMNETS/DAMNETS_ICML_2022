import networkx as nx
import pickle
import numpy as np
import os
import utils.graph_utils as graph_utils
from utils.arg_helper import get_config
from types import SimpleNamespace
from baselines.TagGen.graph_fairnet import Config, main, data_process


def create_edgelists(el_fstr, train_graphs):
    '''
    The TagGen code requires a certain filestructure to work, namely a data directory that contains a timestamped
    edgelist for each network. So this function just creates this file structure.
    Args:
        test_dir: The directory of the test run (of the graph encoder model) that you want to run the baselines on.
        The sampled graphs from tagGen will be placed there
    Returns: None

    '''
    try:
        os.mkdir(el_fstr)
    except FileExistsError:
        pass
    # Loop through each time series in training set
    for k, ts in enumerate(train_graphs):
        # Make a subdirectory for each timeseries to store edgelist
        ts_path = os.path.join(el_fstr, f'{k}')
        try:
            os.mkdir(ts_path)
        except FileExistsError:
            pass
        # Write each graph into an edgelist file (one per time series)
        with open(os.path.join(ts_path, 'edgelist.txt'), 'w') as f:
            for t, g in enumerate(ts):
                for i, j in list(g.edges()):
                    f.write(f'{i} {j} {t}\n')
        graph_utils.save_graph_list(ts, os.path.join(ts_path, f'nx_{k}.pkl'))

def train_test_tag_gen(T, el_fstr):
    tg_args = SimpleNamespace()
    tg_args.slices = T
    tg_args.window = 1
    tg_args.gpu = 0
    tg_args.biased = True

    # set mode and data dir below
    config = Config()
    for entry in os.scandir(el_fstr):
        print('Training at: ', entry.path)
        # tg_args.data_path = f'./data/{}/sequences.txt'.format(tg_args.data)
        tg_args.data_path = os.path.join(entry.path, 'sequences.txt')
        # data path of original sentences
        tg_args.embedding = os.path.join(entry.path, f'{entry.name}_emb')
        # tg_args.embedding = './data/{}/{}_emb'.format(tg_args.data, tg_args.data)
        config.embedding = os.path.join(entry.path, f'{entry.name}_emb')
        # config.embedding = './data/{}/{}_emb'.format(tg_args.data, tg_args.data)
        # config.node_embedding = './data/{}/{}_node_level_emb'.format(tg_args.data, tg_args.data)
        config.node_embedding = os.path.join(entry.path, f'{entry.name}_node_level_emb')
        # tg_args.model_path = './model_{}/'.format(tg_args.data)
        config.use_output_path = os.path.join(entry.path, f'{entry.name}_output_sequences.txt')
        # config.use_output_path = './data/{}/{}_output_sequences.txt'.format(tg_args.data, tg_args.data)
        # output_directory = "./data/{}".format(tg_args.data)
        output_directory = entry.path
        data_directory = os.path.join(entry.path, 'edgelist.txt')
        # data_directory = './data/{}/edgelist.txt'.format(tg_args.data)
        tg_args.emb_size = config.d_model
        interval = T
        tg_args.mode = True
        tg_args.data = entry.name
        data_process(tg_args, interval, tg_args.biased, time_windows=tg_args.window, data_directory=data_directory,
                     output_directory=output_directory,
                     directed=False)
        main(tg_args, config, output_directory)
        tg_args.mode = False
        main(tg_args, config, output_directory)


def dictionary_search(dictionary, search_value):
    for key, value in dictionary.items():
        if value == search_value:
            return key

def load_tag_gen_results(output_dir, data_directory_2):
    graph_attr = pickle.load(open(output_dir + '/graph.pickle', "rb"))
    original_network = graph_attr['graph']
    original_node_index = graph_attr['original_index']
    node_index = graph_attr['index']
    n = original_network.shape[0]
    min_time_stamp = np.inf
    max_time_stamp = 0
    with open(output_dir + '/edgelist_new.txt', 'r') as f:
        for line in f:
            line = list(map(int, line.split()))
            if line[2] < min_time_stamp:
                min_time_stamp = line[2]
            if line[2] > max_time_stamp:
                max_time_stamp = line[2]
    windows = max_time_stamp - min_time_stamp + 1
    original_network = np.zeros((windows, n, n), dtype=np.int8)
    with open(output_dir + '/edgelist_new.txt', 'r') as f:
        for line in f:
            line = list(map(int, line.split()))
            a_1 = int(dictionary_search(node_index, line[0]).split('_')[0])
            a_2 = int(dictionary_search(node_index, line[1]).split('_')[0])
            index_i = original_node_index[a_1]
            index_j = original_node_index[a_2]
            for k in range(line[2], max_time_stamp + 1):
                original_network[k, index_i, index_j] = 1
                original_network[k, index_j, index_i] = 1
    for i in range(n):
        for k in range(windows):
            original_network[k, i, i] = 1
    graph = np.zeros((windows, n, n), dtype=np.float16)
    edge_count = [int(np.sum(original_network[k])) for k in range(windows)]
    with open(data_directory_2, 'r+') as f:
        for line in f:
            line = line.rstrip("\n")
            nodes = list(map(int, line.split(',')))
            for i in range(len(nodes) - 1):
                if i <= len(nodes) - 1:
                    a_1 = list(map(int, dictionary_search(node_index, nodes[i]).split('_')))
                    a_2 = list(map(int, dictionary_search(node_index, nodes[i+1]).split('_')))
                    time_stamp = max(a_1[1], a_2[1])
                    index_i = original_node_index[a_1[0]]
                    index_j = original_node_index[a_2[0]]
                    r = np.random.uniform(low=0.85, high=1)
                    for k in range(time_stamp, windows):
                        graph[k, index_i, index_j] += r
                        graph[k, index_j, index_i] += r
    for i in range(n):
        for k in range(windows):
            graph[k, i, i] = graph[k, i, i] + np.random.uniform(low=0.85, high=1)
    for k in range(windows):
        DD = np.sort(graph[k].flatten())[::-1]
        threshold = DD[edge_count[k]]
        graph[k] = np.array(
            [[0 if graph[k, i, j] <= threshold else 1 for i in range(graph.shape[1])]
             for j in range(graph.shape[2])], dtype=np.int8)
    return graph

def run_tag_gen(test_dir=None, graphs_file=None):
    if test_dir is None:
        with open('experiment_files/last_test.txt', 'r') as f:
            test_dir = f.readline()
            args = get_config(os.path.join(test_dir, 'config.yaml'))
            train_dir = args.experiment.test.test_model_dir
    if graphs_file is None:
        graphs_file = 'test_graphs.pkl'
    train_graphs = graph_utils.load_graph_ts(os.path.join(test_dir, graphs_file))[-5:]
    T = len(train_graphs[0])
    el_fstr = os.path.join(test_dir, 'train_edgelists')
    create_edgelists(el_fstr, train_graphs)
    train_test_tag_gen(T, el_fstr)

    ts_list = []
    for entry in os.scandir(el_fstr):
        print(entry.name)
        data_directory_2 = os.path.join(entry.path, f'{entry.name}_output_sequences.txt')
        ts_array = load_tag_gen_results(entry.path, data_directory_2)
        ts_list.append([nx.Graph(ts_array[t]) for t in range(T)])
    graph_utils.save_graph_list(ts_list, os.path.join(test_dir, 'tag_gen_samples.pkl'))

if __name__ == '__main__':
    run_tag_gen()