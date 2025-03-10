import networkx as nx
import numpy as np
from causallearn.graph import GeneralGraph
from causallearn.graph.AdjacencyConfusion import AdjacencyConfusion
from cdt import metrics

from topic.util.util import directional_f1


def convert_nx_timed_graph_to_adj(timed_graph: nx.DiGraph) -> np.array:
    n_nodes = len(np.unique([i for (i, _) in timed_graph.nodes]))
    tau_max = len(np.unique([lag for (_, lag) in timed_graph.nodes]))

    wcg = np.zeros((n_nodes * (tau_max+1), n_nodes))
    adj = np.zeros((n_nodes, n_nodes))
    for ((i, lag_i), (j, lag_j)) in timed_graph.edges:
        assert lag_j == 0 #convention
        index = n_nodes * lag_i + i
        wcg[index][j] = 1
        if i!=j:
            adj[i][j] = 1
    return wcg, adj

def convert_wcg_to_nx_digraph(n_nodes, tau_max, wcg):
    timed_graph = nx.DiGraph()
    nodeset = set([(i, lag) for i in range(n_nodes) for lag in range(tau_max)])
    assert wcg.shape==(n_nodes * (tau_max + 1), n_nodes)
    timed_graph.add_nodes_from(nodeset)

    for (i, lag_i) in timed_graph.nodes:
        for (j, lag_j) in timed_graph.nodes:
            if lag_j != 0:
                continue#convention
            index = n_nodes * lag_i + i
            if wcg[index][j] == 1:
                timed_graph.add_edge((i, lag_i), (j, lag_j))
    return timed_graph

def convert_nx_graph_to_adj(nx_graph: nx.DiGraph) -> np.array:
    np_adj = np.zeros((len(nx_graph.nodes), len(nx_graph.nodes)))
    for e1, e2 in nx_graph.edges:
        if (e2, e1) not in nx_graph.edges:
            np_adj[e1][e2] = 1
    return np_adj

def compare_nx_digraph_to_dag(nx_graph: nx.DiGraph, nx_true_graph: nx.DiGraph, enable_SID_call: bool):
    np_dag = convert_nx_graph_to_adj(nx_graph)
    np_true_dag = convert_nx_graph_to_adj(nx_true_graph)
    return compare_np_digraph_to_dag(np_dag, np_true_dag, enable_SID_call)


def compare_ggnx_pag_to_dag(gg_pag: GeneralGraph, nx_true_graph: nx.DiGraph, enable_SID_call):
    np_true_dag = convert_nx_graph_to_adj(nx_true_graph)
    return compare_gg_pag_to_dag(gg_pag, np_true_dag, enable_SID_call)


def compare_gg_pag_to_dag(gg_pag: GeneralGraph, np_true_dag: np.array, enable_SID_call: bool):
    # Option 1: convert pag to its corresponding dag
    from causallearn.utils.PDAG2DAG import pdag2dag

    gg_dag = gg_pag#pdag2dag(gg_pag)
    np_dag_adj = np.zeros((len(gg_dag.nodes), len(gg_dag.nodes)))
    #nx_dag_graph = nx.DiGraph()
    #nx_dag_graph.add_nodes_from(set(range(len(gg_dag.graph))))
    for i in range(len(gg_dag.graph)):
        for j in range(len(gg_dag.graph)):
            if gg_dag.graph[j][i] == 1 and gg_dag.graph[i][j] == -1:  # convention: means a causal edge (for FCI, GES, PC etc)
                np_dag_adj[i][j] = 1
               # nx_dag_graph.add_edge(i, j)

    metrics_dag = compare_np_digraph_to_dag(np_true_dag, np_dag_adj, enable_SID_call, allow_cycles=True)

    # Option 2: eval pags directly
    return metrics_dag
    from causallearn.utils.DAG2PAG import dag2pag

    gg_true_dag = None #todo implement this
    gg_true_pag = dag2pag(gg_true_dag, [])
    gg_true_cpdag = dag2cpdag(gg_true_dag)

    np_pag_adj = gg_pag.graph
    adj = AdjacencyConfusion(gg_true_cpdag, gg_pag)

    pagTP = adj.get_adj_tp()
    pagFP = adj.get_adj_fp()
    pagFN = adj.get_adj_fn()
    pagTN = adj.get_adj_tn()

    pagPrec = adj.get_adj_precision()
    pagRec = adj.get_adj_recall()

    # Structural Hamming Distance
    from causallearn.graph.SHD import SHD
    pagSHD = SHD(gg_true_cpdag, gg_pag).get_shd()

    den = pagTP + 1 / 2 * (pagFP + pagFN)

    pagF1 = pagTP / den if den > 0 else 1

    metrics = metrics_dag
    metrics_pag = {
        "pag_shd": pagSHD,
        "pag_tp": pagTP,
        "pag_fp": pagFP,
        "pag_fn": pagFN,
        "pag_tn": pagTN,
        "pag_f1": pagF1,
        "pag_prec": pagPrec,
        "pag_rec": pagRec,
    }
    metrics.update(metrics_pag)


def compare_np_digraph_to_dag(res_dag: np.array, true_dag: np.array,
                              enable_SID_call=False, allow_cycles=False, has_time_lags=False):
    """ Evaluation of two adjacency matrices representing fully oriented DAGs """
    N = res_dag.shape[1]
    N_or_t = res_dag.shape[0]
    max_lag = int(N_or_t/N)-1
    assert true_dag.shape[1] == res_dag.shape[1]
    assert true_dag.shape[0] == res_dag.shape[0]
    if not has_time_lags:
        assert N_or_t == N

    f1, tp, tn, fn, fp = directional_f1(true_dag, res_dag)

    f1check, _, _, _, _ = directional_f1(true_dag, true_dag)
    assert f1check == 1

    den = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    mcc = 1 if den == 0 else (tp * tn - fp * fn) / np.sqrt(den)
    mcc_norm = (mcc+1)/2

    tpr = 0 if (tp + fn == 0) else tp / (tp + fn)
    fpr = 0 if (tn + fp == 0) else fp / (tn + fp)
    fdr = 0 if (tp + fp == 0) else fp / (tp + fp)

    true_ones = np.array(
        [[1 if i != 0 else 0 for i in true_dag[j]] for j in range(len(true_dag))]
    )
    shd = metrics.SHD(true_ones, res_dag)
    shd_norm = shd / (N**2)
    sid, sid_norm = -1, -1

    if enable_SID_call and not has_time_lags: #call to R script not working on some machines
        sid = np.float(metrics.SID(true_ones, res_dag))
        sid_norm = sid / (N * (N-1))

    checksum = tp + tn + fn + fp
    assert checksum == true_dag.shape[0] * true_dag.shape[1]

    n_edges = tp + fn
    dtop, dtop_norm = -1, -1
    is_acyclic=False
    if not has_time_lags:
        is_acyclic = nx.is_directed_acyclic_graph(nx.from_numpy_array(res_dag, create_using=nx.DiGraph))
        eval_top_ordering = is_acyclic
        dtop, dtop_norm = n_edges, 1
        if eval_top_ordering:
            #if not is_acyclic:
            #    res_graph = nx.from_numpy_array(pdag2dag(convert_npadj_to_causallearn_cpdag(res_dag)).graph, create_using=nx.DiGraph)
            res_graph = nx.from_numpy_array(res_dag, create_using=nx.DiGraph())

            # Topological Ordering Divergence (SCORE)
            res_order = list(nx.topological_sort(res_graph))
            dtop = sum([sum([true_dag[pa][i] for pa in range(len(true_dag))
                        if res_order.index(pa) > res_order.index(i)])
                        for i in range(len(true_dag))])
            n_edges = tp+fn
            dtop_norm = 0 if n_edges == 0 else dtop / n_edges

    suff = '-t' if has_time_lags else ''
    res = {
        "shd"+suff: shd,
        "sid"+suff: sid,
        "shd_norm"+suff: shd_norm,
        "sid_norm"+suff: sid_norm,
        "f1"+suff: f1,
        "tpr"+suff: tpr,
        "fpr"+suff: fpr,
        "fdr"+suff: fdr,
        "tp"+suff: tp,
        "fp"+suff: fp,
        "tn"+suff: tn,
        "fn"+suff: fn,
        "mcc"+suff: mcc,
        "mcc_norm"+suff: mcc_norm,
        "dtop"+suff: dtop,
        "dtop_norm"+suff: dtop_norm,
        "is_acyclic"+suff: is_acyclic
    }
    return res



# Other
def get_adj_from_true_links(true_links, tau_max, N, timed):
    if timed:
        true_adj = np.zeros((N * (tau_max + 1), N))
        for j in range(N):
            for entry in true_links[j]:
                ((i, lag), _, _) = entry
                assert lag <= 0 and -lag <= tau_max
                if i != j:
                    index = N * -lag + i
                    true_adj[index][j] = 1
    else:
        true_adj = np.zeros((N, N))
        for j in range(N):
            for entry in true_links[j]:
                ((i, lag), _, _) = entry
                if i != j:
                    true_adj[i][j] = 1
    return true_adj

# time series TODO update
def _compare_adj_to_links(untimed_graph, truths):
    res_adj = np.array(
        [[1 if untimed_graph.has_edge(j, i) else 0 for i in untimed_graph.nodes] for j in
         untimed_graph.nodes])
    return _compare_adj_to_links(
        res_adj, truths.true_links, truths.tau_max, False
    )



def _compare_timed_adj_to_links(timed_graph, untimed_graph, truths):
    res_adj = np.zeros((len(timed_graph.nodes), len(untimed_graph.nodes)))
    for i in timed_graph.nodes:
        for j in untimed_graph.nodes:
            if timed_graph.has_edge(i, (j, 0)):
                idx = i[1] * truths.tau_max
                res_adj[idx][j] = 1
    return _compare_adj_to_links(
        res_adj, truths.true_links, truths.tau_max, True
    )


def _compare_adj_to_links(res_adj, true_links, tau_max, timed):
    N = res_adj.shape[1]
    if timed:
        assert res_adj.shape[0] == N * (tau_max + 1)
        true_adj = np.zeros((N * (tau_max + 1), N))
        for j in range(N):
            for entry in true_links[j]:
                ((i, lag), _, _) = entry
                assert lag <= 0 and -lag <= tau_max
                if i != j:
                    index = N * -lag + i
                    true_adj[index][j] = 1
        res = compare_np_digraph_to_dag(true_adj, res_adj, False)
    else:
        assert res_adj.shape[0] == N
        true_adj = np.zeros((N, N))
        for j in range(N):
            for entry in true_links[j]:
                ((i, lag), _, _) = entry
                if i != j:
                    true_adj[i][j] = 1

        res = compare_np_digraph_to_dag(true_adj, res_adj, False)

    # log.info(
    #    f"\t{name}\t\t(f1={np.round(res['f1'], 2)}, mcc={np.round(res['mcc'], 2)})\t(shd={np.round(res['shd'], 2)}, "
    #    f"sid={np.round(res['sid'], 2)})\t(tp={res['tp']}, tn={res['tn']}, fp={res['fp']}, fn={res['fn']})"
    # )

    return_dict = dict()

    for name, val in res.items():
        suff = "-timed" if timed else ""
        return_dict[name + suff] = val
    return return_dict





