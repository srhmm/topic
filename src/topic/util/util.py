from sklearn import preprocessing
import numpy as np
from cdt.metrics import SHD, SID


def is_insignificant(gain, alpha=0.05):
    return gain < 0 or 2 ** (-gain) > alpha

def cantor_pairing(x, y):
    return int((x + y) * (x + y + 1) / 2 + y)

def data_scale(y):
    scaler = preprocessing.StandardScaler().fit(y)
    return (scaler.transform(y))

def compare_adj(true_adj, my_adj):
    tp, fp, fn = 0, 0, 0
    for i in my_adj:
        for j in my_adj[i]:
            if j in true_adj[i]:
                tp += 1
            else:
                fp += 1
    for i in true_adj:
        for j in true_adj[i]:
            if not (j in my_adj[i]):
                fn += 1
    den = tp + 1 / 2 * (fp + fn)
    f1 = tp / den if den > 0 else 1
    return dict(f1=f1, tp=tp, fp=fp, fn=fn)

def dag_n_edges(adj):
    assert adj.shape[0] == adj.shape[1]
    return sum([len(np.where(adj[i] != 0)[0]) for i in range(len(adj))])


def directional_f1(true_dag, test_dag):
    tp = sum([sum([1 if (test_dag[i][j] != 0 and true_dag[i][j] != 0) else 0
                   for j in range(len(true_dag[i]))]) for i in range(len(true_dag))])
    tn = sum([sum([1 if (test_dag[i][j] == 0 and true_dag[i][j] == 0) else 0
                   for j in range(len(true_dag[i]))]) for i in range(len(true_dag))])
    fp = sum([sum([1 if (test_dag[i][j] != 0 and true_dag[i][j] == 0) else 0
                   for j in range(len(true_dag[i]))]) for i in range(len(true_dag))])
    fn = sum([sum([1 if (test_dag[i][j] == 0 and true_dag[i][j] != 0) else 0
                   for j in range(len(true_dag[i]))]) for i in range(len(true_dag))])
    den = tp + 1 / 2 * (fp + fn)
    if den > 0:
        f1 = tp / den
    else:
        f1 = 1
    return f1, tp, tn, fn, fp


def match_dags(true_dag, test_dag):
    return (not False in [(not False in [((test_dag[i][j] == 0 and true_dag[i][j] == 0)
                                          or (test_dag[i][j] != 0 and true_dag[i][j] != 0))
                                         for j in range(len(test_dag[i]))]) for i in range(len(test_dag))])



def eval_dag(true_dag, res_dag, N, enable_SID_call):
    assert (true_dag.shape[1] == res_dag.shape[1] and res_dag.shape[1] == N)
    assert (true_dag.shape[0] == res_dag.shape[0])

    f1, tp, tn, fn, fp = directional_f1(true_dag, res_dag)

    tpr = 0 if (tp + fn == 0) else tp / (tp + fn)
    fpr = 0 if (tn + fp == 0) else fp / (tn + fp)
    fdr = 0 if (tp + fp == 0) else fp / (tp + fp)

    acc = match_dags(true_dag, res_dag)

    true_ones = np.array([[1 if i != 0 else 0 for i in true_dag[j]] for j in range(len(true_dag))])
    shd = SHD(true_ones, res_dag)
    if enable_SID_call:
        sid = np.float(SID(true_ones, res_dag))
    else:
        sid = 1  # call to R script currently not working

    checksum = tp + tn + fn + fp
    assert (checksum == true_dag.shape[0] * true_dag.shape[1])

    return acc, shd, sid, f1, tpr, fpr, fdr, tp, tn, fn, fp

def gen_lagged_target_links(covariates, i):
    """Generates the links for a single effect, with given time lags"""
    links = []
    # Add causal parents
    for j, lag in covariates:
        links.append(((j, -lag), 1, None))
    # Add self links
    links.append(((i, -1), 1, None))
    return links

def gen_instantaneous_target_links(covariates, i, fixed_lag = 0):
    """Generates the links for a single effect, here, all instantaneous"""
    links = []
    # Add causal parents
    for j in covariates:
        links.append(((j, fixed_lag), 1, None))
    # Add self links
    links.append(((i, -1), 1, None))
    return links

def gen_links_from_lagged_dag(dag, N, random_state, funs):
    """
    Converts DAG with lagged effects to link list (as used in tigramite data generation)

    :param dag: true DAG including self transitions
    :param N: n nodes
    :param random_state: rand
    :param funs: list of possible functional forms for each causal relationship
    """
    # TODO: each variable should have only one list (which is a problem in case of DAG ><><>< like (Jilles example))
    links = dict()
    cnode = dag.nodes[-2]
    snode = dag.nodes[-1]

    for i in range(N):
        links[i] = []
        fun_pa = random_state.choice(funs, size=len(dag.parents_of(i)))

        # Add causal parents in dag
        for index_j, j in enumerate(dag.parents_of(i)):
            if j != cnode and j != snode: # Skip context & spatial links
                lag = int(i//N-j//N)
                w = dag.weight_mat[j][i]
                assert (w != 0)
                j_t = j+lag*N
            else: # context & spatial links
                lag = 0
                w = dag.weight_mat[j][i]
                assert (w != 0)
                j_t = N+(j%N)-1
            links[i].append(((j_t, lag), w, fun_pa[index_j][1]))

    links[N] = [] # context node
    links[N+1] = [] # spatial node
    return links