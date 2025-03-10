import networkx as nx
import numpy as np

from topic.scoring.fitting import fit_functional_model
from topic.scoring.fitting import DataType
from topic.util.util import gen_lagged_target_links


def compute_edge_score(
        X,
        data_type: DataType,
        covariates: list,
        target: int,
        **scoring_params) -> int:
    """ Computes the score for target node given parents.

    :param X: data
    :param data_type: type
    :param covariates: parents of target
    :param target: node
    :return: edge score
    """
    if data_type.is_time(): # supports mcontext-time, mregime-time, or time
        return fit_functional_model_time(
            X, covariates, target, **scoring_params)
    elif data_type.is_multicontext():
        assert data_type == DataType.CONT_MCONTEXT
        return fit_functional_model_context(
            X, covariates, target, **scoring_params)
    else:
        return fit_functional_model_continuous(
            X, covariates, target, **scoring_params)


def fit_functional_model_context(X, parents, target, **kwargs):
    contexts = list(range(len(X)))
    C = len(X)
    M = X[0].shape[1]
    Xgrp, ygrp = {}, {}
    Xcon, ycon = {}, {}

    pval_mat = np.ones((C, C))
    for c1 in range(C):
        for c2 in range(c1 + 1, C):
            data_pa_1, data_node_1 = X[c1][:, parents], X[c1][:, target]
            data_pa_2, data_node_2 = X[c2][:, parents], X[c2][:, target]
            if len(parents) == 0:
                data_pa_1 = np.random.normal(size=X[0][:, [target]].shape)
                data_pa_2 = np.random.normal(size=X[0][:, [target]].shape)
            score_group = fit_functional_model(
                {0: np.vstack([data_pa_1, data_pa_2])}, {0: np.hstack([data_node_1, data_node_2])}, M,
                **kwargs)

            score_sep = fit_functional_model(
                {0: data_pa_1, 1: data_pa_2}, {0: data_node_1, 1: data_node_2}, M, **kwargs)
            pval_mat[c1, c2] = 1 if score_sep > score_group else 0

    map = pval_to_map(pval_mat, alpha=0.5, strong=False)

    n_groups = len(np.unique(map))
    assert max(map) == n_groups-1 and min(map) == 0

    for context,gr in zip(set(contexts), map):
        grp = int(gr)
        data_pa_i, data_node_i = X[context][:,parents], X[context][:,target]
        if len(parents) == 0:
            data_pa_i = np.random.normal(size=X[0][:, [target]].shape)
        if grp not in Xgrp:
            Xgrp[grp] = data_pa_i
            ygrp[grp] = data_node_i
        else:
            Xgrp[grp] = np.vstack([Xgrp[grp], data_pa_i])
            ygrp[grp] = np.hstack([ygrp[grp], data_node_i])
        Xcon[context] = data_pa_i
        ycon[context] = data_node_i
    score_group = fit_functional_model(
        Xgrp, ygrp, M, **kwargs)

    _score_context = fit_functional_model(
        Xcon, ycon, M, **kwargs)

    return score_group


def fit_functional_model_continuous(X, covariates, target, **scoring_params):
    assert len(X) == 1
    M = X[0].shape[1]
    X_parents = {0: np.random.normal(size=X[0][:, [target]].shape)} if len(covariates) == 0 else {
        0: X[0][:, covariates]}
    X_target = {0: X[0][:, target]}
    return fit_functional_model(X_parents, X_target, M, **scoring_params)


def fit_functional_model_time(X, covariates, target, **kwargs):
    """Fit functional model in each time window (for each time window in each context)"""
    skip = kwargs.get("max_lag", 3)
    regime_windows = kwargs.get("regimes", [[skip, X[0].shape[0]]])
    contexts = kwargs.get("context_partition", None)
    regimes = kwargs.get("regime_partition", None)
    target_links = gen_lagged_target_links(covariates, target)
    score = 0

    contexts = list(range(len(X))) if contexts is None else contexts
    regimes = list(range(len(regime_windows))) if regimes is None else regimes

    for context in set(contexts):
        for regime in set(regimes):
            pa_i = [(var, lag) for (var, lag), _, _ in target_links]
            M = len(pa_i)
            data_pa_i, data_all, data_node_i = get_autoregressive_sample(
                X, target_links, context, regime, contexts, regimes, target, regime_windows)
            if M == 0:
                data_pa_i = data_node_i.reshape(-1, 1)
            mdl = fit_functional_model(
                [data_pa_i], [data_node_i], X[0].shape[1], **kwargs)
            score += mdl
    return score


def get_autoregressive_sample(data_C, target_links, context, regime, contexts, regimes, target, windows_T):
    pa_i = [(var, lag) for (var, lag), _, _ in target_links]
    M = len(pa_i)
    N = data_C[0].shape[1]
    assert all([N == data_C[k].shape[1] for k in data_C])

    data_pa_i = np.zeros((1, M), dtype='float32')
    data_all = np.zeros((1, N + 1), dtype='float32')
    data_node_i = np.zeros((1), dtype='float32')

    for dataset in [d for d in range(len(data_C)) if contexts[d] == context]:
        data = data_C[dataset]

        for window, (t0, tn) in enumerate(windows_T):
            if regimes[window] != regime: continue
            T = tn - t0
            data_pa_i_w = np.zeros((T, M), dtype='float32')
            data_all_w = np.zeros((T, N + 1), dtype='float32')

            for j, (var, lag) in enumerate(pa_i):
                if var == target:
                    data_all_w[:, N] = data[t0 + lag:tn + lag, var]
                data_pa_i_w[:, j] = data[t0 + lag:tn + lag, var]
                data_all_w[:, var] = data[t0 + lag:tn + lag, var]

            data_pa_i = np.concatenate([data_pa_i, data_pa_i_w])
            data_all = np.concatenate([data_all, data_all_w])
            data_node_i = np.concatenate([data_node_i, data[t0:tn, target]])

    data_pa_i = data_pa_i[1:]
    data_all = data_all[1:]
    data_node_i = data_node_i[1:]

    return data_pa_i, data_all, data_node_i


def pval_to_map(pval_mat, alpha=0.05, strong=True):
    n_c = pval_mat.shape[0]
    mp = np.zeros(n_c)
    groups = [[e, e] for e in range(n_c)]
    for c1 in range(n_c):
        for c2 in range(c1+1, n_c):
            if pval_mat[c1][c2] > alpha:
                groups.append([c1, c2])
    if strong:
        G = nx.DiGraph()
        G.add_edges_from(groups)
        clusters = list(nx.strongly_connected_components(G))
    else:
        G = nx.Graph()
        G.add_edges_from(groups)
        clusters = list(nx.connected_components(G))
    for i, c in enumerate(clusters): mp[list(c)] = i
    return mp




