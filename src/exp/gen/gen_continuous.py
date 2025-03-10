import random
from itertools import product
from types import SimpleNamespace

import networkx as nx
from matplotlib import pyplot as plt
from pygam import GAM, s, te
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures


from exp.util.HEC import *
from exp.util.ty import NoiseType, FunType, DagType
from exp.util.utils_conversions import convert_contexts_to_stack, convert_hard_interventions_to_idls

from topic.scoring.fitting import DataType


def gen_intervention_targets(params, graph, random_state):
    params["R"] = 1
    intervention_nb =  params["R"]
    skip_observational = 1
    intervention_targets = dict.fromkeys(set(range(params["C"])))
    intervention_target_nodes = dict.fromkeys(set(range(params["C"])))

    choices = random_state.choice(
        list(range(params["N"])),  # list(arcs.arcs),
        size=min(intervention_nb * (params["C"] - skip_observational), params["N"]),
        replace=False,  # (params["N"] < intervention_nb * params["C"]),
    )
    rest = intervention_nb * (params["C"] - skip_observational) - params["N"]
    if rest > 0:
        remaining_choices = random_state.choice(
            list(range(params["N"])),  # list(arcs.arcs),
            size=rest,
            replace=True,
        )
        choices = np.concatenate([choices, remaining_choices])
    ct = 0
    for c in range(params["C"]):
        intervention_targets[c] = []
        intervention_target_nodes[c] = []
    for c in range(skip_observational, params["C"]):
        for ib in range(intervention_nb):
            if choices[ct] not in intervention_target_nodes[c]:
                intervention_target_nodes[c].append(choices[ct])
            for arc in graph.edges:
                if arc[1] == choices[ct]:
                    intervention_targets[c].append(arc)
            # intervention_targets[c].append((choices[ct][0], choices[ct][1]))
            ct += 1
    # print("INTERVENED ARCS")
    # print(intervention_targets)
    print("Data Gen: Intervened nodes in each context: ",  ", ".join([f"context {ci}: targets  {intervention_target_nodes[ci]}" for ci in  intervention_target_nodes]))
    return intervention_target_nodes


def _gen_context_dag( params, nb_edges, intervention_nb, random_state):
    params["R"] = 1
    ## Define DAG structure
    import causaldag as cd
    from graphical_models.rand import unif_away_zero

    arcs = cd.rand.directed_erdos((params["N"]),  params["P"])
    #graph = gen_causal_graph(params)

    skip_observational = 1  # if want to kave an observational context
    # todo replace invervention_nb with sparsity parameter and choose frac of nodes.

    intervention_targets = dict.fromkeys(set(range(params["C"])))
    intervention_target_nodes = dict.fromkeys(set(range(params["C"])))

    choices = random_state.choice(
        list(range(params["N"])),  # list(arcs.arcs),
        size=min(intervention_nb * (params["C"] - skip_observational), params["N"]),
        replace=False,  # (params["N"] < intervention_nb * params["C"]),
    )
    rest = intervention_nb * (params["C"] - skip_observational) - params["N"]
    if rest > 0:
        remaining_choices = random_state.choice(
            list(range(params["N"])),  # list(arcs.arcs),
            size=rest,
            replace=True,
        )
        choices = np.concatenate([choices, remaining_choices])
    ct = 0
    for c in range(params["C"]):
        intervention_targets[c] = []
        intervention_target_nodes[c] = []
    for c in range(skip_observational, params["C"]):
        for ib in range(intervention_nb):
            if choices[ct] not in intervention_target_nodes[c]:
                intervention_target_nodes[c].append(choices[ct])
            for arc in arcs.edges:
                if arc[1] == choices[ct]:
                    intervention_targets[c].append(arc)
            # intervention_targets[c].append((choices[ct][0], choices[ct][1]))
            ct += 1
    # print("INTERVENED ARCS")
    # print(intervention_targets)
    print("Data Gen: Intervened nodes in each context")
    print("\t", ";".join([f"\n\tContext {ci}: targets  {intervention_target_nodes[ci]}" for ci in  intervention_target_nodes]))


    intervention_targets_regimes = {0: []}

    ## For each regime, define general weights and special weights for intervention tagets in contexts
    weights = dict.fromkeys(
        set(product(set(range(params["R"])), set(range(params["C"]))))
    )  # key = (regime, context)

    initial_weights = cd.rand.rand_weights(arcs)

    c_weights = {
        c: {t: unif_away_zero()[0] for t in intervention_targets[c]}
        for c in intervention_targets.keys()
    }
    for r in range(params["R"]):
        r_weights = dict()
        for t in intervention_targets_regimes[r]:
            w = unif_away_zero()[0]
            while abs(w - initial_weights.arc_weights[t]) < 0.1:
                w = unif_away_zero()[0]
            r_weights[t] = w
        # r_weights = {t: unif_away_zero()[0] for t in intervention_targets_regimes[r]}
        for c in range(params["C"]):
            weights[(r, c)] = cd.rand.rand_weights(arcs)
            for arc in initial_weights.arcs:
                if (
                        arc in intervention_targets_regimes[r]
                        and arc in intervention_targets[c]
                ):
                    w = unif_away_zero()[0]
                    while (
                            abs(w - initial_weights.arc_weights[arc]) < 0.1
                            or abs(w - r_weights[arc]) < 0.1
                    ):
                        w = unif_away_zero()[0]
                    weights[(r, c)].set_arc_weight(arc[0], arc[1], w)
                elif (
                        arc not in intervention_targets_regimes[r]
                        and arc in intervention_targets[c]
                ):
                    weights[(r, c)].set_arc_weight(arc[0], arc[1], c_weights[c][arc])
                elif (
                        arc not in intervention_targets_regimes[r]
                        and arc not in intervention_targets[c]
                ):
                    weights[(r, c)].set_arc_weight(
                        arc[0], arc[1], initial_weights.arc_weights[arc]
                    )
                elif (
                        arc in intervention_targets_regimes[r]
                        and arc not in intervention_targets[c]
                ):
                    weights[(r, c)].set_arc_weight(arc[0], arc[1], r_weights[arc])
                else:
                    raise ValueError("wrong case")

    return weights, intervention_target_nodes



class PolynomialRegression():
    def __init__(self, degree=3):
        self.degree = degree

    def fit(self, X, y):
        self.poly = Pipeline([('poly', PolynomialFeatures(degree=self.degree)),
                              ('linear', LinearRegression(fit_intercept=False))])
        self.poly.fit(X, y)
        return self

    def predict(self, X):
        return self.poly.predict(X)

    def get_params(self, deep=True):
        return {'degree': self.degree}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


def plot_case(result_dict):
    plt.subplot(1, 3, 1)
    plt.bar(result_dict.keys(), [result_dict[key]["f1"] for key in result_dict.keys()],
            yerr=[result_dict[key]["f1-std"] for key in result_dict.keys()])
    plt.title("F1")
    plt.subplot(1, 3, 2)
    plt.bar(result_dict.keys(), [result_dict[key]["shd"] for key in result_dict.keys()],
            yerr=[result_dict[key]["shd-std"] for key in result_dict.keys()])
    plt.title("SHD")
    plt.subplot(1, 3, 3)
    plt.bar(result_dict.keys(), [result_dict[key]["sid"] for key in result_dict.keys()],
            yerr=[result_dict[key]["sid-std"] for key in result_dict.keys()])
    plt.title("SID")

    plt.subplot(1, 3, 1)
    plt.bar(result_dict.keys(), [result_dict[key]["f1"] for key in result_dict.keys()],
            yerr=[result_dict[key]["f1-std"] for key in result_dict.keys()])
    plt.title("F1")
    plt.subplot(1, 3, 2)
    plt.bar(result_dict.keys(), [result_dict[key]["shd"] for key in result_dict.keys()],
            yerr=[result_dict[key]["shd-std"] for key in result_dict.keys()])
    plt.title("SHD")
    plt.subplot(1, 3, 3)
    plt.bar(result_dict.keys(), [result_dict[key]["sid"] for key in result_dict.keys()],
            yerr=[result_dict[key]["sid-std"] for key in result_dict.keys()])
    plt.title("SID")

    plt.show()


# make gam with default pairwise terms
class PairwiseGAM():
    def fit(self, X, y):
        terms = s(0)
        for i in range(X.shape[1]):
            if i > 0:
                terms = terms + s(i)
            for j in range(i + 1, X.shape[1]):
                terms = terms + te(i, j)
        self.model = GAM(terms=terms).fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)


def BIC(mse, n, p):
    # actually AIC now
    return 2 * p + n * np.log(mse)
    return n * np.log2(mse) + p * np.log2(n)


def generate_random_dag(num_nodes, num_edges):
    """Generate a random Directed Acyclic Graph (DAG) with the given number of nodes and edges."""
    # Generate a random permutation of nodes
    nodes = list(range(num_nodes))
    random.shuffle(nodes)

    # Initialize a DAG
    dag = nx.DiGraph()

    # Add nodes to the DAG
    dag.add_nodes_from(nodes)

    # Add edges to the DAG ensuring acyclic property
    while len(dag.edges) < num_edges:
        # Randomly select two distinct nodes
        node1, node2 = random.sample(nodes, 2)
        # Add edge if it doesn't introduce a cycle
        if not nx.has_path(dag, node2, node1):
            dag.add_edge(node1, node2)

    return dag



def sample_random_multivariate_taylor_series(num_terms, num_variables, center, radius, coefficient_range=(-1, 1)):
    """Sample a random multivariate Taylor series."""
    # Generate random coefficients for each term
    coefficients = np.random.uniform(coefficient_range[0], coefficient_range[1], size=(num_terms,))

    # Generate random exponents for each variable in each term
    exponents = np.random.randint(0, 5, size=(num_terms, num_variables))

    # Construct the Taylor series function
    def taylor_series(*args):
        series_sum = 0
        for i in range(num_terms):
            term = coefficients[i]
            for j in range(num_variables):
                term *= (args[j] - center[j]) ** exponents[i, j]
            series_sum += term
        return series_sum

    return taylor_series

def gen_causal_graph(params: dict)-> nx.DiGraph:
    if params['DG'] == DagType.ERDOS_CDT:
        import cdt.data as tb
        exp_deg = 2
        generator = tb.AcyclicGraphGenerator(
            'polynomial', nodes=params["N"], npoints=1, noise='uniform' if params["NS"]==NoiseType.UNIF else 'gaussian',
             noise_coeff=0.3, dag_type='erdos', expected_degree=exp_deg)
        _, graph = generator.generate()
        graph = nx.relabel_nodes(graph, mapping={f"V{i}": i for i in range(params["N"])})

    elif params['DG'] == DagType.ERDOS_CD:
        import causaldag as cd
        arcs = cd.rand.directed_erdos(params['N'], params['P'])
        nodes = list(range(params["N"]))
        graph = nx.DiGraph()
        graph.add_nodes_from(nodes)
        _ = [graph.add_edge(n1, n2) for n1 in nodes for n2 in nodes if (n1, n2) in arcs.arcs]

    elif params['DG'] == DagType.SCALE_FREE:
        G = nx.directed.scale_free_graph(
            params["N"],
            alpha=0.41, beta=0.54, gamma=0.05,
            delta_in=0.2, delta_out=0)
        G = G.to_directed()
        _G = nx.DiGraph()
        for u, v, _ in G.edges:
            if (u, v) not in _G.edges:
                _G.add_edge(u, v)
        try:
            while True:
                cycle = nx.find_cycle(_G)
                e = cycle.pop()
                _G.remove_edge(*e)
        except nx.NetworkXNoCycle:
            pass
        graph = _G
    elif params['DG'] == DagType.RANDOM:
        # todo set these dep. on edge p
        avg_edges = (params['N']) // 3
        n_edges = (params['N'] * avg_edges) // 2
        graph = generate_random_dag(params['N'], n_edges)
    else:
        raise ValueError(f"{params['DG']}")
    return graph


def gen_continuous_data(params, random_state, use_cd=False):
    graph = gen_causal_graph(params)

    data = gen_data_from_graph(graph, params['S'], random_state, params['NS'], params['F'][0], params['F'][1])

    truths = SimpleNamespace(
        data_type=DataType.CONTINUOUS,
        graph=graph,
        order=list(nx.topological_sort(graph)),
        is_true_edge=lambda node: lambda other: 'causal' if graph.has_edge(node, other) else (
        'anticausal' if graph.has_edge(other, node) else 'spurious')
    )
    return data, truths


def gen_context_data(params, random_state):
    graph = gen_causal_graph(params)
    nodes = graph.nodes

    intervention_targets_c = gen_intervention_targets(params, graph, random_state)
    data_C = {}
    for c in range(params["C"]):
        graph_c = nx.DiGraph()
        graph_c.add_nodes_from(nodes)
        _ = [graph_c.add_edge(n1, n2) for n1 in nodes for n2 in nodes if (n1, n2) in graph.edges and n2 not in intervention_targets_c[c]]
        data_C[c] = gen_data_from_graph(graph_c, params['S'], random_state, params['NS'], params['F'][0], params['F'][1])

    data_summary = SimpleNamespace(
        datasets=data_C,
        datasets_combined=convert_contexts_to_stack(data_C),
        #datasets_labelled=convert_contexts_to_labelled_stack(data_C),
    )
    truths = SimpleNamespace(
        data_type=DataType.CONT_MCONTEXT,
        graph=graph,
        order=list(nx.topological_sort(graph)),
        is_true_edge=lambda node: lambda other: 'causal' if graph.has_edge(node, other) else (
        'anticausal' if graph.has_edge(other, node) else 'spurious'),
        targets=intervention_targets_c,
        idls=convert_hard_interventions_to_idls(data_summary.datasets, intervention_targets_c)
    )
    return data_summary, truths

def gen_data_from_graph(
        G, n_samples, random_state,
        noise_dist: NoiseType,
        function_type: FunType, fn,
        noise_std=0.2):
    """Generate data from a given graph."""
    # todo could add intervention type
    noise_fun = lambda sz, std: (
        random_state.normal(size=sz, scale=std)) if noise_dist == NoiseType.GAUSS \
        else (random_state.uniform(size=sz) * 2 - 1)

    order = list(nx.topological_sort(G))
    n_nodes = len(order)
    X = np.zeros((n_samples, n_nodes))
    for node in order:
        parents = list(G.predecessors(node))
        if len(parents) == 0:
            X[:, node] = noise_fun(n_samples, noise_std)
        elif function_type == FunType.TAYLOR:
            center = np.zeros(len(parents))
            radius = 1
            f = sample_random_multivariate_taylor_series(10, len(parents), center, radius)
            # make list of parent vectors
            in_data = [X[:, parents[i]] for i in range(len(parents))]
            intermediate_pred = f(*in_data)
            # standardize
            intermediate_pred = (intermediate_pred - np.mean(intermediate_pred)) / np.std(intermediate_pred)

            X[:, node] = intermediate_pred + noise_fun(n_samples, noise_std)
            # normalize to -1 and 1
            X[:, node] = (X[:, node] - np.min(X[:, node])) / (np.max(X[:, node]) - np.min(X[:, node])) * 2 - 1

        elif function_type == FunType.LIN:
            # sample random coefficients between -1 and 1
            coefficients = random_state.uniform(-1, 1, size=len(parents)) # todo away from zero
            intermediate_pred = np.dot(X[:, parents], coefficients)
            intermediate_pred = (intermediate_pred - np.mean(intermediate_pred)) / np.std(intermediate_pred)

            X[:, node] = intermediate_pred + noise_fun(n_samples, noise_std)
            X[:, node] = (X[:, node] - np.min(X[:, node])) / (np.max(X[:, node]) - np.min(X[:, node])) * 2 - 1
        else:
            in_data = [X[:, parents[i]] for i in range(len(parents))]
            intermediate_pred = fn(*in_data)
            # standardize
            intermediate_pred = (intermediate_pred - np.mean(intermediate_pred)) / np.std(intermediate_pred)

            X[:, node] = intermediate_pred + noise_fun(n_samples, noise_std)
            # normalize to -1 and 1
            X[:, node] = (X[:, node] - np.min(X[:, node])) / (np.max(X[:, node]) - np.min(X[:, node])) * 2 - 1
    return X


def sample_random_taylor_series(num_terms, center, radius, coefficient_range=(-1, 1)):
    """Sample a random Taylor series."""
    # Generate random coefficients for each term
    coefficients = np.random.uniform(coefficient_range[0], coefficient_range[1], size=num_terms)

    # Construct the Taylor series function
    def taylor_series(x):
        series_sum = 0
        for i in range(num_terms):
            series_sum += coefficients[i] * (x - center) ** i
        return series_sum

    return taylor_series

