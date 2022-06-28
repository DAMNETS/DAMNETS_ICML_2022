import pickle
import argparse
import igraph
import os
import networkx as nx
import utils.graph_utils as graph_utils
from utils.arg_helper import get_config
from baselines.DYMOND.DYMOND import get_dataset, learn_parameters, dymond_generate
from multiprocessing import Pool


def create_directories(fstr, train_graphs):
    try:
        os.mkdir(fstr)
    except FileExistsError:
        pass
    for k, ts in enumerate(train_graphs):
        # Make a subdirectory for each timeseries to store edgelist
        ts_path = os.path.join(fstr, f'{k}')
        try:
            os.mkdir(ts_path)
        except FileExistsError:
            pass
        graph_utils.save_graph_list(ts, os.path.join(ts_path, f'nx_{k}.pkl'))


def create_dymond_datasets(fstr):
    for entry in os.scandir(fstr):
        ts = graph_utils.load_graph_ts(os.path.join(entry.path, f'nx_{entry.name}.pkl'))
        edgelist = []
        edge_timesteps = []
        for t, G in enumerate(ts):
            edgelist += list(G.edges)
            edge_timesteps += [t + 1 for _ in G.edges]
        g = igraph.Graph(n=G.number_of_nodes(),
                         directed=False,
                         edges=edgelist,
                         edge_attrs={'timestep': edge_timesteps}
                         )
        for v in g.vs:
            v['nid'] = f'nid-{v.index}'  # Annotate with original index
            neighbors = list(set([u for u in g.neighbors(v)]))
            if len(neighbors) > 0:
                v_edges = g.es.select(_between=([v.index], neighbors))
                v['active'] = min(v_edges['timestep'])

        # Save to file
        graph_filename = f'{entry.name}_ig.pklz'
        g.write_picklez(os.path.join(entry.path, graph_filename))

        T = len(ts)
        timesteps = [i + 1 for i in range(T)]
        dataset_info = {'gname': graph_filename,
                        'L': 1,
                        'N': g.vcount(),
                        'T': len(timesteps),
                        'timesteps': timesteps
                        }
        dataset_info_file = os.path.join(entry.path, 'dataset_info.pkl')
        with open(dataset_info_file, 'wb') as output:
            pickle.dump(dataset_info, output)


def train_test_dymond(path):
    # for entry in os.scandir(fstr):
    dataset_dir, dataset_info, g = get_dataset(dataset_dir=path)
    learn_parameters(dataset_dir, dataset_info, g)
    dymond_generate(dataset_dir, dataset_info['T'] + 1)


def run_dymond(graphs_file, output_dir):
    # if test_dir is None:
    #     with open('experiment_files/last_test.txt', 'r') as f:
    #         test_dir = f.readline()
    #     args = get_config(os.path.join(test_dir, 'config.yaml'))
    #     test_dir = args.experiment.test.test_model_dir
    #
    # if graphs_file is None:
    #     graphs_file = 'test_graphs.pkl'
    # train_graphs = graph_utils.load_graph_ts(os.path.join(test_dir, graphs_file))
    train_graphs = graph_utils.load_graph_ts(graphs_file)
    T = len(train_graphs[0])
    fstr = os.path.join(output_dir, 'train_edgelists')

    create_directories(fstr, train_graphs)
    create_dymond_datasets(fstr)
    dirs = [entry.path for entry in os.scandir(fstr)]
    if len(dirs) == 1:
        train_test_dymond(dirs[0])
    else:
        with Pool() as p:
            p.map(train_test_dymond, dirs)
    ts_list = []

    for entry in os.scandir(fstr):
        sampled_ts = []
        sampled_fstr = os.path.join(entry.path, 'learned_parameters/generated_graph/generated_graph.pklz')
        ig_ts = igraph.Graph().Read_Picklez(sampled_fstr)
        for t in range(1, T + 1):
            g = nx.from_edgelist(list(e.tuple for e in ig_ts.es.select(lambda e: e['timestep'] == t)))
            sampled_ts.append(g)
        ts_list.append(sampled_ts)

    graph_utils.save_graph_list(ts_list, os.path.join(output_dir, 'dymond_samples.pkl'))

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="A script to run the DYMOND baseline model")
    parser.add_argument(
        '-p',
        '--graphs_path',
        type=str,
        help='The path to the graphs on which to train DYMOND.'
    )
    parser.add_argument(
        '-o',
        '--output_dir',
        type=str,
        help='The directory in which to place the sampled graphs from DYMOND.'
    )
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_arguments()
    run_dymond(args.graphs_path, args.output_dir)



