import random
from itertools import product, combinations
from types import SimpleNamespace

import causaldag as cd
import networkx as nx
import numpy as np
from graphical_models.rand import unif_away_zero

from .exp.util.ty import NoiseType
from .exp.gen.context_model import ContextModelTSWithRegimes
from .exp.util.eval import get_adj_from_true_links, convert_wcg_to_nx_digraph
from .exp.util.utils_conversions import convert_contexts_to_stack, convert_contexts_to_labelled_stack

from .topic.scoring.fitting import DataType


def r_partition_to_windows_T(r_partition, skip):
    """
    Convert an r_partition into a windows_T
    :param r_partition: [(begin, duration, regime)]
    :param skip:
    :return: [(begin, end)]
    """
    return [(b + skip, b + d) for (b, d, r) in r_partition]


def partition_t(T, R, n, min_dur=100, equal_dur=True):
    """
    Generate a partition of n chunks over R different regimes and T datapoints
    :param T: Total length of the time series
    :param R: Number of different regimes
    :param n: Number of chunks
    :param min_dur: Minimal duration of the chunks
    :param equal_dur: If all chunks should have the same length (except the last)
    :return:
    """
    success = True
    if not equal_dur:
        cuts, success = generate_cuts(T, n, min_dur)
        p = [(cuts[i], cuts[i + 1] - cuts[i], None) for i in range(len(cuts) - 1)]
    if equal_dur or not success:
        dur = T // n
        p = [(i * dur, dur, None) for i in range(n - 1)]
        p.append(((n - 1) * dur, dur + T % n, None))
    p = [(p[i][0], p[i][1], r) for i, r in enumerate(generate_seq(R, n))]
    return p


def generate_cuts(T, n, min_dur=100, max_attempts=100):
    """
    Cur a range (0, T) into n bins of minimal width min_dur
    :param T: Length of the range
    :param n: Number of bins
    :param min_dur: Minimal bin width
    :param max_attempts: Maximal number of attempts to obtain the correct minimal width
    :return: The cutpoints, a flag indicating the success or failure of the task
    """
    attempts = 0
    durs = [False]
    while not all(durs) and attempts < max_attempts:
        cuts = [0] + sorted(random.sample(range(T), n - 1)) + [T]
        durs = [(cuts[i + 1] - cuts[i]) >= min_dur for i in range(len(cuts) - 1)]
        attempts += 1
    return cuts, all(durs)


def generate_seq(R, n):
    """
    Generate a sequence of length n from R symbols with each symbol at least once and no consecutive repetition
    :param R: Number of different symbols
    :param n: Length of the sequence
    :return:
    """
    assert n >= R
    seq = list(range(R))
    random.shuffle(seq)
    for i in range(R, n):
        choices = set(range(R))
        seq.append(random.choice(list(choices - {seq[i - 1]})))
    return seq


def gen_time_data(params, seeds, seed, _depth=99):
    """ Generates data in multiple contexts and regimes
    :param options: RunOptions
    :param params: test case, entries R: n_regimes, C: n_contexts, D: n_datasets, T: n_timeseriessamples, F: functional form
    :random_state: random state
    :param _depth: recursion depth if invalid data generated
    """
    true_max_lag = params['TM']
    true_min_dur = 20

    regime_drift = False
    random_state = np.random.default_rng(seeds[_depth])
    #if params["NS"] == NoiseType.UNIF:
    #    raise NotImplementedError
    # Hyperparams
    nb_of_chunks = params['R']  # random_state.integers(params['R'], params['T'] / options.true_min_dur)
    equal_dur = random_state.choice([True, False])
    nb_edges = random_state.integers(1, params['N'])  # number of (lagged) links between different variables
    # todo edge p

    skip = params['TM']+ 1  # nn of datapoints to skip at the beginning of a regime
    params['D'] = 1

    ### MODEL GENERATION
    weights, intervention_targets_c = gen_time_dag(params, nb_edges, random_state)
    regimes_partition = partition_t(params['S'], params['R'], nb_of_chunks, true_min_dur, equal_dur)
    windows_T = r_partition_to_windows_T(regimes_partition, skip)
    truth = [r for (b, d, r) in regimes_partition]

    ### DATA GENERATION WITH TIGRAMITE
    cnode = params['N']
    snode = params['N'] + 1
    node_classification = {
        cnode: 'time_context',  # unused
        snode: 'space_context'  # unused
    }
    for n in range(params['N']): node_classification[n] = 'system'

    noises = [NoiseType.to_noise(params["NS"])[1](random_state) for _ in
              range(params['N'] + 2)]  # same for all regimes and contexts

    func = dict.fromkeys(weights[(0, 0)].arcs)
    for k in func.keys():
        func[k] = params['F']

    datasets = dict()
    datasets_without_dummynodes = dict()

    cpt = 0
    invalid_data = False
    data = dict.fromkeys(set(product(set(range(params['C'])), set(range(params['D'])))))
    for c in range(params['C']):
        links = dict.fromkeys(range(params['R']))
        for r in range(params['R']):
            dag = weights[(r, c)]
            links[r] = gen_links_from_lagged_dag(dag, params['N'], random_state, func)
        n_drift = 0.1 * true_min_dur if regime_drift else 0
        contextmodel = ContextModelTSWithRegimes(links_regimes=links, node_classification=node_classification,
                                                 noises=noises, seed=seed)  # , noises=noises
        data_ens, nonstationary = contextmodel.generate_data_with_regimes(
            params['D'], params['S'], regimes_partition, n_drift=n_drift)
        if nonstationary: invalid_data = True
        for d in range(params['D']):
            data[(c, d)] = data_ens[d]
            datasets[cpt] = data[(c, d)]
            datasets_without_dummynodes[cpt] = data[(c, d)][:, :params['N']]
            cpt += 1
    if invalid_data and _depth > 0:
        return gen_time_data(params, seeds, _depth - 1)

    for ky in data:
        assert (data[ky].shape[1] == params['N'] + 2)


    # DATA
    data_summary = SimpleNamespace(
        datasets=datasets_without_dummynodes,
        datasets_combined=convert_contexts_to_stack(datasets),
        datasets_labelled=convert_contexts_to_labelled_stack(datasets),
        tau_max=params['TM'],
        datasets_with_dummynodes=datasets,
        cnode=cnode,
        snode=snode,
        node_classification=node_classification
    )
    # GROUND TRUTH
    sdag = get_adj_from_true_links(links[0], params['TM'], params['N'], False)
    wcg = get_adj_from_true_links(links[0], params['TM'], params['N'], True)
    assert sdag.shape == (params["N"], params["N"])
    assert wcg.shape == (params["N"] * (params["TM"] + 1), params["N"])

    timed_graph = convert_wcg_to_nx_digraph(params['N'], params['TM'], wcg)
    timed_order = list(nx.topological_sort(timed_graph))
    targets = {i: [ n for (x, n) in intervention_targets_c[i]] for i in intervention_targets_c}
    truths = SimpleNamespace(
        data_type=DataType.TIME_MCONTEXT,
        timed_graph=timed_graph,
        timed_order=timed_order,
        timed_dag=wcg,
        graph=sdag,
        windows_T=windows_T,
        tau_max=params['TM'],
        true_links=links[0],
        is_true_edge=links_to_is_true_edge(links[0]),
        true_r_partition=truth,
        targets=targets,
        true_contexts=datasets_without_dummynodes,
        true_regimes=regimes_partition,
        skip=skip
    )
    return data_summary, truths, not invalid_data


def gen_time_dag(params, nb_edges, random_state):
    ## Define DAG structure
    intervention_nb = params["I"]
    arcs = cd.rand.directed_erdos(((params['TM'] + 1) * params['N']) + 2, 0)

    # across variable links

    pairs = random_state.choice(list(combinations(range(params['N']), 2)), nb_edges, replace=False)
    pairs = [(j, i) if random_state.random() > .5 else (i, j) for (i, j) in pairs]
    lags = random_state.choice(range(params['TM'] + 1), nb_edges)
    for i in range(nb_edges):
        arcs = add_edge(pairs[i][0], pairs[i][1], -lags[i], params['N'], arcs)

    for i in range(params['N']):
        # self links
        arcs = add_edge(i, i, -1, params['N'], arcs)

    ## For each context define intervention targets
    intervention_targets = dict.fromkeys(set(range(params['C'])))
    for c in intervention_targets.keys():
        # interventions shouldn't be on edges from spatial or context variable
        intervention_targets[c] = random_state.choice(list(arcs.arcs), size=intervention_nb,
                                                      replace=False)  # TODO: assert different from a context to another?
        intervention_targets[c] = list(tuple(l) for l in intervention_targets[c])


    ## For each regime define intervention targets
    intervention_targets_regimes = dict.fromkeys(set(range(params['R'])))
    for r in intervention_targets_regimes.keys():
        nb = random_state.integers(1, params['N'] + 1)
        # interventions shouldn't be on edges from spatial or context variable
        intervention_targets_regimes[r] = random_state.choice(list(arcs.arcs), size=nb,
                                                              replace=False)  # TODO: assert different from a context to another?
        intervention_targets_regimes[r] = list(tuple(l) for l in intervention_targets_regimes[r])

    ## For each regime, define general weights and special weights for intervention tagets in contexts
    weights = dict.fromkeys(set(product(set(range(params['R'])), set(range(params['C'])))))  # key = (regime, context)

    initial_weights = cd.rand.rand_weights(arcs)

    c_weights = {c: {t: unif_away_zero()[0] for t in intervention_targets[c]} for c in intervention_targets.keys()}
    for r in range(params['R']):
        r_weights = dict()
        for t in intervention_targets_regimes[r]:
            w = unif_away_zero()[0]
            while abs(w - initial_weights.arc_weights[t]) < .1:
                w = unif_away_zero()[0]
            r_weights[t] = w
        # r_weights = {t: unif_away_zero()[0] for t in intervention_targets_regimes[r]}
        for c in range(params['C']):
            weights[(r, c)] = cd.rand.rand_weights(arcs)
            for arc in initial_weights.arcs:
                if arc in intervention_targets_regimes[r] and arc in intervention_targets[c]:
                    w = unif_away_zero()[0]
                    while abs(w - initial_weights.arc_weights[arc]) < .1 or abs(w - r_weights[arc]) < .1:
                        w = unif_away_zero()[0]
                    weights[(r, c)].set_arc_weight(arc[0], arc[1], w)
                elif arc not in intervention_targets_regimes[r] and arc in intervention_targets[c]:
                    weights[(r, c)].set_arc_weight(arc[0], arc[1], c_weights[c][arc])
                elif arc not in intervention_targets_regimes[r] and arc not in intervention_targets[c]:
                    weights[(r, c)].set_arc_weight(arc[0], arc[1], initial_weights.arc_weights[arc])
                elif arc in intervention_targets_regimes[r] and arc not in intervention_targets[c]:
                    weights[(r, c)].set_arc_weight(arc[0], arc[1], r_weights[arc])
                else:
                    print('oups')  # TODO: remove

    return weights, intervention_targets


def add_edge(node_i_t, node_j_t, lag, N, arcs):
    """
    Add arc to the DAG from a parent to the child at time t

    :param node_i_t: parent node name at time t
    :param node_j_t: child node name at time t
    :param lag: lag (negative value)
    :param N: number of variables
    :param arcs: DAG
    :return: DAG
    """
    arcs.add_arc(node_i_t + (-lag * N), node_j_t)
    return arcs


def links_to_is_true_edge(rlinks):
    """ For info during DAG search """

    def is_true_edge(parent):
        def fun(j):
            i, lag = parent
            info = ''
            if True in [rlinks[j][k][0][0] == i and rlinks[j][k][0][1] == -lag for k in
                        range(len(rlinks[j]))]:
                info += '[caus]'
            elif True in [rlinks[j][k][0][0] == i for k in
                          range(len(rlinks[j]))]:
                info += '[caus]'
            if True in [rlinks[i][k][0][0] == j and rlinks[i][k][0][1] == -lag for k in range(len(rlinks[i]))]:
                info += '[rev]'
            elif True in [rlinks[i][k][0][0] == j for k in range(len(rlinks[i]))]:
                info += '[rev]'
            if len(info) == 0:
                info += '[spu]'
            return info

        return fun

    return is_true_edge


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
        # fun_pa = random_state.choice(funs, size=len(dag.parents_of(i)))

        # Add causal parents in dag
        for index_j, j in enumerate(dag.parents_of(i)):
            if j != cnode and j != snode:  # Skip context & spatial links
                lag = int(i // N - j // N)
                w = dag.weight_mat[j][i]
                assert (w != 0)
                j_t = j + lag * N
            else:  # context & spatial links
                lag = 0
                w = dag.weight_mat[j][i]
                assert (w != 0)
                j_t = N + (j % N) - 1
            links[i].append(((j_t, lag), w, funs[(j, i)][1]))

    links[N] = []  # context node
    links[N + 1] = []  # spatial node
    return links
