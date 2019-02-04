import numpy as np
import json
import configure as conf
import codecs
from tqdm import tqdm

def read_data(input_path, word_idx, if_increase_dict):
    graphs = []
    seqs = []
    with codecs.open(input_path, 'r', "utf-8") as fr:
        lines = fr.readlines()
        for _ in tqdm(range(len(lines))):
            line = lines[_].strip()
            jo = json.loads(line)
            seq = jo['seq']
            seqs.append(seq)
            if if_increase_dict:
                features = jo['g_ids_features']
                for id in features:
                    w = features[id]
                    if w not in word_idx:
                        word_idx[w] = len(word_idx) + 1

            graph = {}
            graph['g_ids_features'] = jo['g_ids_features']
            graph['g_adj'] = jo['g_adj']
            graphs.append(graph)

    return seqs, graphs

def vectorize_seq(word_idx, texts):
    tv = []
    for text in texts:
        stv = []
        for w in text.split():
            if w not in word_idx:
                stv.append(word_idx[conf.rare_tag])
            else:
                stv.append(word_idx[w])
        tv.append(stv)
    return tv

def collect_categories(paths):
    supertag_freq = {}
    for path in paths:
        with codecs.open(path, 'r', 'utf-8') as fr:
            lines = fr.readlines(path)
            for line in lines:
                info = line.strip().split(" ")
                for t in info:
                    t_info = t.split("|")
                    supertag = t_info[2]
                    if supertag not in supertag_freq:
                        supertag_freq[supertag] = 1
                    else:
                        supertag_freq[supertag] += 1

    sorted_list = sorted(supertag_freq.items(), key=lambda d: d[1])
    res = [conf.rare_tag]
    for _ in range(len(sorted_list)):
        freq = sorted_list[_][1]
        if freq >= 10:
            res.append(sorted_list[_][0])
    return res

def batch_graph(graphs):
    g_ids_features = {}
    g_fw_adj = {}
    g_bw_adj = {}
    g_nodes = []

    for g in graphs:
        id_adj = g['g_adj']
        features = g['g_ids_features']

        nodes = []

        # we first add all nodes into batch_graph and create a mapping from graph id to batch_graph id, this mapping will be
        # used in the creation of fw_adj and bw_adj

        id_gid_map = {}
        offset = len(g_ids_features.keys())
        for id in features.keys():
            feat = features[id]
            id = int(id)
            g_ids_features[offset + id] = feat
            id_gid_map[id] = offset + id
            nodes.append(offset + id)
        g_nodes.append(nodes)

        for id in id_adj:
            adj = id_adj[id]
            id = int(id)
            g_id = id_gid_map[id]
            if g_id not in g_fw_adj:
                g_fw_adj[g_id] = []
            for t in adj:
                t = int(t)
                g_t = id_gid_map[t]
                g_fw_adj[g_id].append(g_t)
                if g_t not in g_bw_adj:
                    g_bw_adj[g_t] = []
                g_bw_adj[g_t].append(g_id)

    node_size = len(g_ids_features.keys())
    for id in range(node_size):
        if id not in g_fw_adj:
            g_fw_adj[id] = []
        if id not in g_bw_adj:
            g_bw_adj[id] = []

    graph = {}
    graph['g_ids_features'] = g_ids_features
    graph['g_nodes'] = g_nodes
    graph['g_fw_adj'] = g_fw_adj
    graph['g_bw_adj'] = g_bw_adj

    return graph



def vectorize_batch_graph(graph, word_idx):
    # vectorize the graph feature and normalize the adj info
    id_features = graph['g_ids_features']
    gv = {}
    nv = []
    n_len_v = []
    word_max_len = 0
    for id in id_features:
        feature = id_features[id]
        word_max_len = max(word_max_len, len(feature.split()))
    # word_max_len = min(word_max_len, conf.word_size_max)

    for id in graph['g_ids_features']:
        feature = graph['g_ids_features'][id]
        fv = []
        for token in feature.split():
            if len(token) == 0:
                continue
            if token in word_idx:
                fv.append(word_idx[token])
            else:
                fv.append(word_idx[conf.unknown_word])

        if len(fv) > word_max_len:
            n_len_v.append(word_max_len)
        else:
            n_len_v.append(len(fv))

        for _ in range(word_max_len - len(fv)):
            fv.append(0)
        fv = fv[:word_max_len]
        nv.append(fv)

    # add an all-zero vector for the PAD node
    nv.append([0 for temp in range(word_max_len)])
    n_len_v.append(0)

    gv['g_ids_features'] = np.array(nv)
    gv['g_ids_feature_lens'] = np.array(n_len_v)

    # ============== vectorize adj info ======================
    g_fw_adj = graph['g_fw_adj']
    g_fw_adj_v = []

    degree_max_size = 0
    for id in g_fw_adj:
        degree_max_size = max(degree_max_size, len(g_fw_adj[id]))

    g_bw_adj = graph['g_bw_adj']
    for id in g_bw_adj:
        degree_max_size = max(degree_max_size, len(g_bw_adj[id]))

    degree_max_size = min(degree_max_size, conf.sample_size_per_layer)

    for id in g_fw_adj:
        adj = g_fw_adj[id]
        for _ in range(degree_max_size - len(adj)):
            adj.append(len(g_fw_adj.keys()))
        adj = adj[:degree_max_size]
        g_fw_adj_v.append(adj)

    # PAD node directs to the PAD node
    g_fw_adj_v.append([len(g_fw_adj.keys()) for _ in range(degree_max_size)])

    g_bw_adj_v = []
    for id in g_bw_adj:
        adj = g_bw_adj[id]
        for _ in range(degree_max_size - len(adj)):
            adj.append(len(g_bw_adj.keys()))
        adj = adj[:degree_max_size]
        g_bw_adj_v.append(adj)

    # PAD node directs to the PAD node
    g_bw_adj_v.append([len(g_bw_adj.keys()) for _ in range(degree_max_size)])

    # ============== vectorize nodes info ====================
    g_nodes = graph['g_nodes']
    graph_max_size = 0
    for nodes in g_nodes:
        graph_max_size = max(graph_max_size, len(nodes))

    g_node_v = []
    for nodes in g_nodes:
        mask = [1 for _ in range(len(nodes))]
        for _ in range(graph_max_size - len(nodes)):
            nodes.append(len(g_fw_adj.keys()))
            mask.append(0)
        nodes = nodes[:graph_max_size]
        g_node_v.append(nodes)

    gv['g_nodes'] = np.array(g_node_v)
    gv['g_bw_adj'] = np.array(g_bw_adj_v)
    gv['g_fw_adj'] = np.array(g_fw_adj_v)

    return gv
