import time
from itertools import product

import numpy as np
import pandas as pd
from numpy.linalg import LinAlgError

from exp.util.ty import TimeMethod, ValidMethodType, DAGType
from exp.util.dag_utils import convert_npadj_to_causallearn_cpdag, convert_np_timed_to_causallearn_cpdag
from exp.util.eval import convert_nx_timed_graph_to_adj

from topic.scoring.fitting import DataType
from topic.topic_variants import TopicTimed

""" Direction 3: OBSERVATIONAL, time series """


class TopTimeMethod(TimeMethod):
    """ Main method for top. cd. from multivariate time series """

    @staticmethod
    def nm(): return 'TOPTime'

    @staticmethod
    def ty(): return ValidMethodType.TopTime

    @staticmethod
    def dag_ty(): return [DAGType.TDAG, DAGType.WCG]

    def fit(self, data, truths, params, options):
        super(TopTimeMethod, self).check_data(data, truths, params)

        obj = TopicTimed(
            data.datasets, DataType.TIMESERIES,
            options.score_type, params["T"],
            options.verbosity, options.logger,
            truths.is_true_edge, truths.windows_T, params['TM']  # provide true tau max
        )
        time_st = time.perf_counter()
        self.untimed_graph, self.timed_graph, self.top_order = obj.fit()
        self.metrics = {'time': time.perf_counter() - time_st}
        self.wcg, self.tdag = convert_nx_timed_graph_to_adj(self.timed_graph)
        self.obj = obj


class OracleTimeMethod(TimeMethod):
    """ Oracle: known true causal ordering, consider nodes in true order and add outgoing edges"""

    @staticmethod
    def nm(): return 'TOPOracleTime'

    @staticmethod
    def ty(): return ValidMethodType.OracleTime

    @staticmethod
    def dag_ty(): return [DAGType.TDAG, DAGType.WCG]

    def fit(self, data, truths, params, options):
        super(OracleTimeMethod, self).check_data(data, truths, params)

        obj = TopicTimed(
            data.datasets, DataType.TIMESERIES,
            options.score_type, params["T"],
            options.verbosity, options.logger,
            truths.is_true_edge, truths.windows_T, params['TM']  # provide true tau max
        )
        obj.add_true_top_order(truths.timed_order)
        time_st = time.perf_counter()
        self.untimed_graph, self.timed_graph, self.top_order = obj.fit()
        self.metrics = {'time': time.perf_counter() - time_st}
        self.wcg, self.tdag = convert_nx_timed_graph_to_adj(self.timed_graph)
        self.obj = obj


class PCMCIPLUSMethod(TimeMethod):

    @staticmethod
    def nm(): return 'PCMCI'

    @staticmethod
    def ty(): return ValidMethodType.PCMCI

    @staticmethod
    def dag_ty(): return [DAGType.TDAG, DAGType.WCG_PAG]

    def fit(self, data, truths, params, options):
        super(PCMCIPLUSMethod, self).check_data(data, truths, params)
        from tigramite import data_processing as pp
        from tigramite.pcmci import PCMCI
        from sklearn.gaussian_process import GaussianProcessRegressor
        from tigramite.independence_tests.gpdc import GPDC

        # Hyperparams
        cond_ind_test = GPDC()  # ParCorrMult(significance='analytic') if fun == lin
        pred_model = GaussianProcessRegressor()  # LinearRegression()
        time_st = time.perf_counter()
        wcg_pags, tdags = [], []
        for ci in range(len(data.datasets)):
            dataset = pp.DataFrame(data.datasets[ci])
            pcmci_obj = PCMCI(dataframe=dataset, cond_ind_test=cond_ind_test, verbosity=0)

            results = pcmci_obj.run_pcmciplus(tau_min=params["TM"], tau_max=params["TM"], pc_alpha=0.2)

            graph = results["graph"]
            assert graph.shape == (params["N"], params["N"], params["TM"] + 1)

            wcg_pag, tdag = convert_pcmci_links_to_dag(graph, params)
            wcg_pags.append(wcg_pag)
            tdags.append(tdag)
            self.obj = pcmci_obj
        self.wcg_pag, self.tdag = convert_list_to_dag(wcg_pags, tdags, params)
        self.metrics = {'time': time.perf_counter() - time_st}


def convert_list_to_dag(wcg_pags, tdags, params):
    """ report edge optimistically, if found in any context"""
    N, tau_max = params["N"], params["TM"]
    wcg_pag = np.zeros((N * (tau_max + 1), N))
    pag = np.zeros((N, N))
    for wcg, pg in zip(wcg_pags, tdags):
        for i, j, lag in zip3rng(N, N, tau_max + 1):
            index = N * lag + i
            wcg_pag[index][j] = 1 if wcg[index][j] == 1 else 0
            pag[i][j] = 1 if pg[i][j] == 1 else 0

    return wcg_pag, pag


def convert_pcmci_links_to_pag(graph, params):
    N, tau_max = params["N"], params["TM"]
    wcg_pag = np.zeros((N * (tau_max + 1), N))
    pag = np.zeros((N, N))
    for i, j, lag in zip3rng(N, N, tau_max + 1):
        index = N * lag + i
        if graph[i][j][lag] == "-->":
            wcg_pag[index][j] = 1
            pag[i][j] = 1
        if graph[j][i][lag] == "o-o":
            assert lag == 0  # contemporaneous bidirected edge
            wcg_pag[index][j] = 1
            pag[i][j] = 1
        if graph[j][i][lag] == "x-x":
            wcg_pag[index][j] = 1
            pag[i][j] = 1
    return wcg_pag, pag


def convert_pcmci_links_to_dag(graph, params):
    N, tau_max = params["N"], params["TM"]
    wcg_pag = np.zeros((N * (tau_max + 1), N))
    dag = np.zeros((N, N))
    for i, j, lag in zip3rng(N, N, tau_max + 1):
        index = N * lag + i
        if graph[i][j][lag] == "-->":
            wcg_pag[index][j] = 1
            if i != j:
                dag[i][j] = 1
        if graph[j][i][lag] == "o-o":
            assert lag == 0  # contemporaneous bidirected edge
            wcg_pag[index][j] = 0
        if graph[j][i][lag] == "x-x":
            wcg_pag[index][j] = 0
            dag[i][j] = 0
    return wcg_pag, dag


class DyNOTEARSMethod(TimeMethod):

    @staticmethod
    def nm(): return 'DyNOTEARS'

    @staticmethod
    def ty(): return ValidMethodType.DyNOTEARS

    @staticmethod
    def dag_ty(): return [DAGType.TPAG, DAGType.WCG_PAG]

    def fit(self, data, truths, params, options):
        super(DyNOTEARSMethod, self).check_data(data, truths, params)
        from causalnex.structure.dynotears import from_pandas_dynamic
        wcg_pags, tdags = [], []
        time_st = time.perf_counter()
        for ci in range(len(data.datasets)):
            dataset = data.datasets[ci]
            dynotears_obj = from_pandas_dynamic(pd.DataFrame(dataset), params['TM'])
            self.metrics = {'time': time.perf_counter() - time_st}
            self.obj = dynotears_obj
            wcg_pag, tpag = convert_dynotears_to_pag(dynotears_obj, params)
            wcg_pags.append(wcg_pag)
            tdags.append(tpag)

        self.wcg_pag, self.tpag = convert_list_to_dag(wcg_pags, tdags, params)
        self.metrics = {'time': time.perf_counter() - time_st}


def convert_dynotears_to_pag(model, params):
    N, tau_max = params["N"], params["TM"]
    timed_adj = np.zeros((N * (tau_max + 1), N))
    adj = np.zeros((N, N))

    for (node1, node2) in model.edges():
        n, nlg = node1.split('_lag')
        m, mlg = node2.split('_lag')
        n, m, nlg, mlg = int(n), int(m), int(nlg), int(mlg)
        assert (mlg == 0 and nlg >= 0)
        index = N * nlg + n
        timed_adj[index][m] = 1
        if (m != n) and not (nlg == 0 and timed_adj[m][n] != 0):  # skip instantaneous cycles
            adj[n][m] = 1

    return timed_adj, adj


class VarLINGAMMethod(TimeMethod):
    @staticmethod
    def nm():
        return 'VarLINGAM'

    @staticmethod
    def ty():
        return ValidMethodType.VarLINGAM

    @staticmethod
    def dag_ty():
        return [DAGType.TDAG, DAGType.WCG_PAG]

    def fit(self, data, truths, params, options):
        super(VarLINGAMMethod, self).check_data(data, truths, params)
        from causallearn.search.FCMBased import lingam

        time_st = time.perf_counter()
        wcg_pags, tdags = [], []
        for ci in range(len(data.datasets)):
            dataset = data.datasets[ci]
            varlingam_obj = lingam.VARLiNGAM(lags=params["TM"])
            from sklearn import preprocessing
            dataset = preprocessing.normalize(dataset)
            try:
                varlingam_obj.fit(dataset)
                wcg_pag, tdag = convert_varlingam_to_dag(varlingam_obj, params)
            except LinAlgError:
                wcg_pag, tdag = np.zeros(((params["TM"] + 1), params["N"])), np.zeros((params["N"], params["N"]))
            wcg_pags.append(wcg_pag)
            tdags.append(tdag)

        self.wcg_pag, self.tdag = convert_list_to_dag(wcg_pags, tdags, params)
        self.metrics = {'time': time.perf_counter() - time_st}
        self.obj = varlingam_obj


def convert_varlingam_to_dag(model, params):
    tadj = np.hstack(model.adjacency_matrices_).T
    # adj = np.sum(model.adjacency_matrices_, axis=0).T
    # From the documentation:
    from causallearn.search.FCMBased import lingam
    dlingam = lingam.DirectLiNGAM()
    dlingam.fit(model.residuals_)
    adj = dlingam.adjacency_matrix_.T

    timed_adj = np.zeros(((params["TM"] + 1) * params["N"], params["N"]))
    untimed_adj = np.zeros((params["N"], params["N"]))
    for i in range(len(tadj)):
        for j in range(len(tadj[i])):
            if j < params["N"] and i < params["N"]:  # instantaneous edge
                # decide directions of bidirected edges
                if (adj[i][j] != 0 and adj[j][i] == 0) or (adj[i][j] != 0 and adj[i][j] > adj[j][i]):
                    timed_adj[i][j] = 1
                    timed_adj[i][j] = 0
            elif tadj[i][j] != 0:  # lagged edge
                timed_adj[i][j] = 1

    assert timed_adj.shape == ((params["TM"] + 1) * params["N"], params["N"])
    assert adj.shape == (params["N"], params["N"])

    for i, j in ziprng(len(adj), len(adj)):
        untimed_adj[i][i] = 0
        if (adj[i][j] != 0 and adj[j][i] == 0) or (adj[i][j] != 0 and adj[i][j] > adj[j][i]):
            untimed_adj[i][j] = 1
            untimed_adj[j][i] = 0

    return timed_adj, untimed_adj


class CDNODTMethod(TimeMethod):
    @staticmethod
    def nm(): return 'CDNODT'

    @staticmethod
    def ty(): return ValidMethodType.CDNODT

    @staticmethod
    def dag_ty(): return [DAGType.TPAG]  # no wcg returned

    def fit(self, data, truths, params, options):
        super(CDNODTMethod, self).check_data(data, truths, params)
        from causallearn.search.ConstraintBased.CDNOD import cdnod

        cdnod_indep_test = 'kci'
        cdnod_alpha = 0.05
        time_st = time.perf_counter()
        wcg_pags, tdags = [], []
        for ci in range(len(data.datasets)):
            dataset = data.datasets[ci]
            dataset = data.datasets[0]
            # Add exogenous variable with time indices
            c_indx = np.array([i for i in range(dataset.shape[0])]).reshape(-1, 1)

            time_st = time.perf_counter()
            cdnod_obj = cdnod(dataset, c_indx, indep_test=cdnod_indep_test, alpha=cdnod_alpha)
            self.obj = cdnod_obj
            tdags.append(convert_cdnod_to_pag(cdnod_obj, params))
            wcg_pags.append(
                np.zeros((params["N"] * (params["TM"] + 1), params["N"])))  # not used further, simlifies code
        self.metrics = {'time': time.perf_counter() - time_st}
        _, self.tpag = convert_list_to_dag(wcg_pags, tdags, params)


def convert_cdnod_to_pag(model, params):
    assert params["N"] == len(model.G.graph) - 1
    adj = np.zeros((len(model.G.graph) - 1, len(model.G.graph) - 1))
    ivs = np.zeros(len(model.G.graph) - 1)
    nodes = range(len(model.G.graph))
    model.labels = {i: model.G.nodes[i].get_name() for i in nodes}
    model.nx_graph.add_nodes_from(nodes)
    directed = model.find_fully_directed()
    bidirected = model.find_bi_directed()
    for (i, j) in directed:
        if i == len(model.G.graph) - 1:
            if j != len(model.G.graph) - 1:
                ivs[j] = 1
        elif j == len(model.G.graph) - 1:
            if i != len(model.G.graph) - 1:
                ivs[i] = 1
        elif (i, j) not in bidirected:
            adj[i][j] = 1

    return adj


class RhinoMethod(TimeMethod):
    @staticmethod
    def nm(): return 'Rhino'

    @staticmethod
    def ty(): return ValidMethodType.Rhino

    @staticmethod
    def dag_ty(): return [DAGType.TPAG, DAGType.WCG_PAG]

    def fit(self, data, truths, params, options):
        super(RhinoMethod, self).check_data(data, truths, params)

        dataset = data.datasets[0]

        time_st = time.perf_counter()
        raise NotImplementedError

        self.metrics = {'time': time.perf_counter() - time_st}
        wcg_pag = convert_np_timed_to_causallearn_cpdag(tadj)
        pag = convert_npadj_to_causallearn_cpdag(adj)


def ziprng(i: int, j: int):
    return set(product(set(range(i)), set(range(j))))


def zip3rng(i: int, j: int, k: int):
    return set(product(set(range(i)), set(range(j)), set(range(k))))
