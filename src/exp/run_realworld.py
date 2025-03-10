import os
import time

from exp.util.ty import ExpType
from exp.util.eval import compare_np_digraph_to_dag

from collections import defaultdict
from types import SimpleNamespace

import networkx as nx
import numpy as np

from exp.util.case_results import CaseReslts
from exp.util.eval import convert_nx_graph_to_adj

import pandas as pd


def run_realworld(options):
    np.random.seed(options.seed)
    if options.exp_type == ExpType.REGED:
        run_reged(options)
    elif options.exp_type == ExpType.SACHS:
        run_sachs(options)
    elif options.exp_type == ExpType.TUEBINGEN:
        run_tuebingen(options)
    else:
        raise ValueError(ExpType)


def run_realworld_instance(data, truths, options, info, weight=1):
    options.logger.info(info)  # f"\n*** Tuebingen Pair {i} ***" f"\n\tParams: {params} ")
    n_samp, n_nodes = data.shape
    params = {"C": 1, "S": n_samp, "N": n_nodes, "R": 1, "P": 0.3, "TM": 2}

    # Run methods
    metrics = defaultdict(SimpleNamespace)
    for method in options.methods:
        options.logger.info(f'\tMethod: {method.nm()}')
        time_st = time.perf_counter()
        dag = np.zeros((n_nodes, n_nodes))
        try:
            method.fit(method, data, truths, params, options)
            dag = method.dag
            graph: np.array = convert_nx_graph_to_adj(method.dag)
        except Exception:
            options.logger.info("Run failed")
            print (f"Time: {time.perf_counter() - time_st}")
            method.dag = nx.from_numpy_array(dag, create_using=nx.DiGraph)
            graph = dag

        method.eval_results(method, truths, options)
        method.metrics["weight"] = weight
        metrics[method.nm()] = SimpleNamespace(
            obj=method,
            metrics=method.metrics,
            res_dag=graph,
            true_dag=truths.adj,
            got_timed_result=False)

        print(f"Time: {time.perf_counter() - time_st}")
    return metrics, params


def run_reged(options, rdpath="datasets/reged/"):
    if not os.path.exists(rdpath):
        return
    wrpath = options.out_dir + "reged/"
    if not os.path.exists(wrpath):
        os.makedirs(wrpath)
    exps = []
    for rdfile in os.listdir(rdpath):
        if rdfile.endswith(".txt"):
            exp_id = rdfile.split(".txt")[0]
            exp_id = exp_id.split("_")[0]
            if exp_id not in exps:
                exps.append(exp_id)

    reslts = [CaseReslts(exp_id) for exp_id in exps]
    options.reps = 1
    for exp_id, res in zip(exps, reslts):

        metrics, params = run_reged_methods(exp_id, options, rdpath)
        res.add_reps([metrics])
        res.plot_case(params, options)
        res.write_case(params, options)


def run_reged_methods(exp_id, options, rdpath):
    data = np.genfromtxt(rdpath + exp_id + '.txt', delimiter=',')
    n_samp, n_nodes = data.shape
    true_edges = pd.read_csv(rdpath + exp_id + "_truth.txt", header=None, delimiter=r"\s+").to_numpy()
    node_ids = np.genfromtxt(rdpath + exp_id + '_header.txt', delimiter=',')  #
    assert len(node_ids) == n_nodes
    assert all([i in node_ids for i in np.unique(true_edges)]) and all([i in np.unique(true_edges) for i in node_ids])
    true_adj = np.zeros((n_nodes, n_nodes))
    for n in range(n_nodes):
        for m in range(n_nodes):
            if any([edge[0] == node_ids[n] and edge[1] == node_ids[m] for edge in true_edges]):
                true_adj[n][m] = 1

    true_graph = nx.from_numpy_array(true_adj, create_using=nx.DiGraph())
    truths = SimpleNamespace(
        adj=true_adj,
        graph=true_graph,
        order=list(nx.topological_sort(true_graph)),
        is_true_edge= lambda i: lambda j: ("causal " if true_adj[i][j] != 0
                                        else "anticausal" if true_adj[j][i] != 0 else "spurious")
    )
    return run_realworld_instance(data, truths, options, f"*** REGED {exp_id} ")


def read_reged_res():
    adj = pd.read_csv("results_reged/adjs/adj_TOPCont.csv", header=None)
    adjo = pd.read_csv("results_reged/adjs/adj_TOPOracleCont.csv", header=None)
    adj_true = pd.read_csv("results_reged/adjs/adj_true.csv", header=None)
    compare_np_digraph_to_dag(np.array(adj), np.array(adj_true))
    compare_np_digraph_to_dag(np.array(adjo), np.array(adj_true))


def run_tuebingen(options, path="datasets/tuebingen_pairs/"):
    meta = pd.read_csv(f'{path}pairmeta.txt', sep='\s').to_numpy()
    weights = []
    res = CaseReslts(f"tuebingen")
    params = {"C": 1, "S": 500, "N": 2,   "R": 1, "P": 0.3, "TM": 2}
    metrics = []

    for i in range(1, 108):
        st = f"000{i}" if i < 10 else f"00{i}" if i < 100 else f"0{i}"
        if i == 65:
            continue
        try:
            data_i = pd.read_csv(f'{path}pair{st}.txt', sep='\s').to_numpy()
            with open(f'{path}pair{st}_des.txt', 'r') as file:
                info_i = file.read().replace('\n', '--')
        except Exception:
            continue
        cause1, cause2, eff1, eff2, weight = None, None, None, None, None
        for (stt, c1, c2, e1, e2, w) in meta:
            if int(st) == int(stt):
                cause1, cause2, eff1, eff2, weight = c1, c2, e1, e2, w
                break
        if not all([cause1, cause2, eff1, eff2, weight]):
            continue
        if cause1 != cause2 or eff1 != eff2:
            continue
        adj = np.zeros((2, 2))
        adj[0][1] = 1 if int(cause1) == 1 and int(eff1) == 2 else 0
        adj[1][0] = 1 if int(cause1) == 2 and int(eff1) == 1 else 0
        n_samp, n_nodes = data_i.shape

        if n_nodes != 2:
            continue
        is_true_edge = lambda i: lambda j: "causal" if adj[i][j] != 0 else "anticausal" if adj[j][i] != 0 else "spurious"
        params = {"C": 1, "S": n_samp, "N": n_nodes,   "R": 1, "P": 0.3, "TM": 2}

        truths = SimpleNamespace(
            adj = adj,
            graph=nx.from_numpy_array(adj, create_using=nx.DiGraph),
            order=[],
            is_true_edge=is_true_edge, windows_T=[(3, n_samp)]
         )

        metrics_pair, _ = run_realworld_instance(data_i, truths, options, f"*** Tuebingen Pair {i}", weight)
        metrics.append(metrics_pair)
        weights.append(weight)

    res.add_reps(metrics)  # list of all Tuebingen pairs
    res.plot_case(params, options)
    res.write_case(params, options)

    # Print weighted results
    options.logger.info("\n*** Tuebingen Pairs (weighted results) ***")
    f1s = {}
    for entry, weight in zip(metrics, weights):
        for method in entry:
            if method not in f1s:
                f1s[method] = {}

            for meas in entry[method].metrics:
                if meas not in f1s[method]:
                    f1s[method][meas] = []
                    f1s[method][meas+ "-weighted"] = []
                f1s[method][meas + "-weighted"].append(entry[method].metrics[meas] * weight)
                f1s[method][meas].append(entry[method].metrics[meas] * 1)

    for meas in ["f1", "dtop_norm", "sid_norm", "shd_norm"]:
        for method in f1s:
            options.logger.info(f"{method}, {meas}: {sum(f1s[method][meas]) / len(weights):.2f}")

    options.logger.info("\n*** Tuebingen Pairs (unweighted) ***")
    for meas in ["f1", "dtop_norm", "sid_norm", "shd_norm"]:
        for method in f1s:
            options.logger.info(f"{method}, {meas}-weighted: {sum(f1s[method][meas+'-weighted']) / sum(weights):.2f}")


def run_sachs(options, rdpath="datasets/sachs"):
    data_1 = pd.read_csv(f"{rdpath}/sachs_1.csv")
    node_ids = data_1.columns

    import cdt
    datacdt, graph = cdt.data.load_dataset('sachs')
    node_ids_cdt = datacdt.columns
    node_ids_transposed = np.array(['pakts473', 'p44/42', 'pjnk', 'pmek', 'P38', 'PIP2', 'PIP3', 'PKA',
                                    'PKC', 'plcg', 'praf'])  # to match data columns
    assert all([n in node_ids_cdt for n in node_ids_transposed])
    options.logger.info(
        f"SACHS node IDs (make sure they match): \n\t{','.join(node_ids)}\n\t{','.join(node_ids_transposed)}")
    datacdt = np.array(datacdt)
    n_samp, n_nodes = datacdt.shape
    true_adj = np.zeros((n_nodes, n_nodes))
    true_g = nx.DiGraph()
    true_g.add_nodes_from(set(range(n_nodes)))
    for (n, m) in graph.edges():
        ixn, ixm = int(np.where(node_ids_transposed == n)[0]), int(np.where(node_ids_transposed == m)[0])
        true_adj[ixn][ixm] = 1
        true_g.add_edge(ixn, ixm)

    is_true_edge = lambda i: lambda j: ("causal " if true_adj[i][j] != 0
                                        else "anticausal" if true_adj[j][i] != 0 else "spurious")
    truths = SimpleNamespace(
        graph=true_g,
        adj = true_adj,
        order=[],  # list(nx.topological_sort(graph)),
        is_true_edge=is_true_edge
    )
    run_sachs_combined(options)



def run_sachs_combined(options):
    import cdt
    datafr, graph = cdt.data.load_dataset('sachs')
    print(datafr.head())
    node_ids = datafr.columns
    data = np.array(datafr)
    n_samp, n_nodes = data.shape
    true_adj = np.zeros((n_nodes, n_nodes))
    true_g = nx.DiGraph()
    true_g.add_nodes_from(set(range(n_nodes)))
    for (n, m) in graph.edges():
        ixn, ixm = int(np.where(node_ids == n)[0]), int(np.where(node_ids == m)[0])
        true_adj[ixn][ixm] = 1
        true_g.add_edge(ixn, ixm)

    res = CaseReslts("sachs_combined")
    is_true_edge = lambda i: lambda j: ("causal " if true_adj[i][j] != 0
                                        else "anticausal" if true_adj[j][i] != 0 else "spurious")
    params = {"C": 1, "S": n_samp, "N": n_nodes,  "R": 1, "P": 0.3}
    truths = SimpleNamespace(
        graph=true_g,
        order=[],  # list(nx.topological_sort(graph)),
        is_true_edge=is_true_edge
    )

    options.logger.info(f"\n*** SACHS ***"
                        f"\n\tParams: {params}\n\tTrue Edges: " +
                        ", ".join([f"{node_ids[e1]} -> {node_ids[e2]}" for (e1, e2) in truths.graph.edges]))

    # Run methods
    metrics = defaultdict(SimpleNamespace)
    for method in options.methods:
        options.logger.info(f'\tMethod: {method.nm()}')
        method.fit(method, data, truths, params, options)
        method.eval_results(method, truths, options)
        graph: np.array = convert_nx_graph_to_adj(method.dag)
        metrics[method.nm()] = SimpleNamespace(
            obj=method,
            metrics=method.metrics,
            res_dag=graph,
            true_dag=convert_nx_graph_to_adj(truths.graph),
            got_timed_result=False)
        if method.dag is not None:
            options.logger.info(f"\n*** RESULT ***\n" + "\n".join(
                [f"{node_ids[e1]} -> {node_ids[e2]} : {true_adj[e1][e2]}, {true_adj[e2][e1]}" for (e1, e2) in
                 method.dag.edges]))

    print(metrics)
    res.add_reps([metrics])  # list of one bc one rep
    res.plot_case(params, options)
    res.write_case(params, options)
