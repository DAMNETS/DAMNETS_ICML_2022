import pickle


def save_graph_list(G_list, fname):
    with open(fname, "wb") as f:
        pickle.dump(G_list, f)


def load_graph_list(fname):
    with open(fname, "rb") as f:
        graph_list = pickle.load(f)
    return graph_list


def load_graph_ts(fname):
    with open(fname, "rb") as f:
        ts_list = pickle.load(f)
    return ts_list #[ts for ts in ts_list]

