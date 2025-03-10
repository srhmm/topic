import time

import networkx as nx
import numpy as np
import pandas as pd

from exp.util.ty import ContinuousMethod, ChangeMethod, ValidMethodType, TimeMethod, DAGType, ExpType
from exp.methods.methods_time import convert_cdnod_to_pag, convert_pcmci_links_to_dag
from exp.util.dag_utils import convert_npadj_to_causallearn_cpdag
from exp.util.eval import convert_nx_graph_to_adj, convert_nx_timed_graph_to_adj

from topic.scoring.fitting import DataType
from topic.topic import Topic

""" Direction 2: INTERVENTIONAL """


class TopChangesContMethod(ChangeMethod):
    """Main method for top. c.d. using causal mechanism changes"""

    @staticmethod
    def nm(): return 'TOPChangesCont'

    @staticmethod
    def ty(): return ValidMethodType.TopChangesCont

    @staticmethod
    def dag_ty(): return [DAGType.DAG]  # could evaluate the detected interventions

    def fit(self, data, truths, params, options):
        super(TopChangesContMethod, self).check_data(data, truths, params)
        obj = TopologicalChangesCD(
            data.datasets, DataType.CONT_MCONTEXT,
            options.score_type, params["T"],
            options.verbosity, options.logger,
            truths.is_true_edge)

        assert not obj.is_oracle and not obj.known_order
        time_st = time.perf_counter()
        self.dag, _ = obj.fit()
        self.metrics = {'time': time.perf_counter() - time_st}
        self.obj = obj


class OracleChangesContMethod(ChangeMethod):
    """Oracle: known causal mechanism changes"""

    @staticmethod
    def nm(): return 'TOPOracleChangesCont'

    @staticmethod
    def ty(): return ValidMethodType.OracleChangesCont

    @staticmethod
    def dag_ty(): return [DAGType.DAG]

    def fit(self, data, truths, params, options):
        super(OracleChangesContMethod, self).check_data(data, truths, params)
        obj = TopologicalChangesCD(
            data.datasets, DataType.CONT_MCONTEXT,  # datasets not used here
            options.score_type, params["T"],
            options.verbosity, options.logger,
            truths.is_true_edge)

        assert truths.targets is not None
        obj.add_truths(truths)
        assert obj.is_oracle and not obj.known_order

        time_st = time.perf_counter()
        self.dag, _ = obj.fit()
        self.metrics = {'time': time.perf_counter() - time_st}
        self.obj = obj


class TopChangesTimeMethod(ChangeMethod):
    """Top. c.d. using causal mechanism changes for time series"""

    @staticmethod
    def nm(): return 'TOPChangesTime'

    @staticmethod
    def ty(): return ValidMethodType.TopChangesTime

    @staticmethod
    def dag_ty(): return [DAGType.TDAG, DAGType.WCG]

    @staticmethod
    def valid_exp_type():
        return ExpType.TIMECHANGES

    def fit(self, data, truths, params, options):
        super(TopChangesTimeMethod, self).check_data(data, truths, params)
        obj = TopologicalChangesTimedCD(
            data.datasets, DataType.TIME_MCONTEXT,
            options.score_type, params["T"],
            options.verbosity, options.logger,
            truths.is_true_edge)

        assert not obj.known_order
        time_st = time.perf_counter()
        self.untimed_graph, self.timed_graph, self.top_order = obj.fit()
        self.metrics = {'time': time.perf_counter() - time_st}
        self.wcg, self.tdag = convert_nx_timed_graph_to_adj(self.timed_graph)
        self.obj = obj


class OracleChangesTimeMethod(ChangeMethod):
    """Oracle: known causal mechanism changes"""

    @staticmethod
    def nm(): return 'TOPOracleChangesTime'

    @staticmethod
    def ty(): return ValidMethodType.OracleChangesTime

    @staticmethod
    def dag_ty(): return [DAGType.TDAG, DAGType.WCG]

    @staticmethod
    def valid_exp_type():
        return ExpType.TIMECHANGES

    def fit(self, data, truths, params, options):
        super(OracleChangesTimeMethod, self).check_data(data, truths, params)
        obj = TopologicalChangesTimedCD(
            data.datasets, DataType.TIME_MCONTEXT,
            options.score_type, params["T"],
            options.verbosity, options.logger,
            truths.is_true_edge)

        assert truths.targets is not None
        obj.add_truths(truths)
        assert obj.is_oracle and not obj.known_order

        time_st = time.perf_counter()
        self.untimed_graph, self.timed_graph, self.top_order = obj.fit()
        self.metrics = {'time': time.perf_counter() - time_st}
        self.wcg, self.tdag = convert_nx_timed_graph_to_adj(self.timed_graph)
        self.obj = obj


class JCIPCMethod(ChangeMethod):
    @staticmethod
    def nm():
        return 'JCI-PC'

    @staticmethod
    def ty():
        return ValidMethodType.JCI_PC

    @staticmethod
    def dag_ty():
        return [DAGType.DAG]

    @staticmethod
    def valid_exp_type():
        return ExpType.CHANGES

    def fit(self, data, truths, params, options):
        super(JCIPCMethod, self).check_data(data, truths, params)
        data_labelled = data.datasets_labelled
        from causallearn.search.ConstraintBased.PC import pc
        options.methodparams_pc_indep_test = 'kci'  # todo change for linear exp

        time_st = time.perf_counter()
        pc_obj = pc(data_labelled, indep_test=options.methodparams_pc_indep_test)
        self.metrics = {'time': time.perf_counter() - time_st}
        adj = pc_obj.G.graph
        self.adj = np.zeros((params["N"], params["N"]))
        for i in range(params["N"]):
            for j in range(params["N"]):
                if adj[j][i] == 1 and adj[i][j] == -1:
                    self.adj[i][j] = 1
        self.dag = nx.from_numpy_array(self.adj, create_using=nx.Graph)
        self.obj = pc_obj


class JCIFCIMethod(ChangeMethod):
    @staticmethod
    def nm():
        return 'JCI-FCI'

    @staticmethod
    def ty():
        return ValidMethodType.JCI_FCI

    @staticmethod
    def dag_ty():
        return [DAGType.DAG]

    @staticmethod
    def valid_exp_type():
        return ExpType.CHANGES

    def fit(self, data, truths, params, options):
        super(JCIFCIMethod, self).check_data(data, truths, params)
        data_labelled = data.datasets_labelled
        from causallearn.search.ConstraintBased.FCI import fci

        time_st = time.perf_counter()
        pag, edges = fci(data_labelled, independence_test_method='kci')
        self.metrics = {'time': time.perf_counter() - time_st}
        self.obj = pag, edges
        adj = pag.graph
        self.adj = np.zeros((params["N"], params["N"]))
        for i in range(params["N"]):
            for j in range(params["N"]):
                if adj[j][i] == 1 and adj[i][j] == -1:
                    self.adj[i][j] = 1
        self.dag = nx.from_numpy_array(self.adj, create_using=nx.Graph)


class CDNODMethod(ChangeMethod):
    @staticmethod
    def nm(): return 'CDNOD'

    @staticmethod
    def ty(): return ValidMethodType.CDNOD

    @staticmethod
    def dag_ty(): return [DAGType.DAG]

    @staticmethod
    def valid_exp_type():
        return ExpType.CHANGES

    def fit(self, data, truths, params, options):
        super(CDNODMethod, self).check_data(data, truths, params)
        from causallearn.search.ConstraintBased.CDNOD import cdnod
        cdnod_indep_test = 'kci'
        cdnod_alpha = 0.05

        # Add exogenous variable with context indices and combine datasets
        c_indx = None
        for ci, c in enumerate(data.datasets):
            c_indx = ci * np.ones(data.datasets[c].shape[0]) if c_indx is None else np.vstack(
                (c_indx, ci * np.ones(data.datasets[c].shape[0])))

        time_st = time.perf_counter()
        cdnod_obj = cdnod(data.datasets_combined, c_indx.reshape(-1, 1), indep_test=cdnod_indep_test, alpha=cdnod_alpha)
        self.metrics = {'time': time.perf_counter() - time_st}
        self.dag = nx.from_numpy_array(convert_cdnod_to_pag(cdnod_obj, params), create_using=nx.DiGraph)
        self.obj = cdnod_obj


class MGLINGAMMethod(ChangeMethod):
    @staticmethod
    def nm(): return 'MGLINGAM'

    @staticmethod
    def ty(): return ValidMethodType.MGLINGAM

    @staticmethod
    def dag_ty(): return [DAGType.DAG]

    @staticmethod
    def valid_exp_type():
        return ExpType.CHANGES

    def fit(self, data, truths, params, options):
        super(MGLINGAMMethod, self).check_data(data, truths, params)
        import lingam
        X_list = [dataset for dataset in data.datasets.values()]
        if len(X_list) == 1:
            X_list = [data.datasets[0], data.datasets[0]]
        mglingam_obj = lingam.MultiGroupDirectLiNGAM()
        time_st = time.perf_counter()
        mglingam_obj.fit(X_list)
        self.metrics = {'time': time.perf_counter() - time_st}
        adj = mglingam_obj.adjacency_matrices_[0].T
        weights = np.where(adj != 0, 1, 0)
        # add edge if it exists in any context
        # for ci in range(len(mglingam_obj.adjacency_matrices_)):
        #    adj = mglingam_obj.adjacency_matrices_[ci].T
        #    weights += np.where(adj != 0, 1, 0)

        self.dag = nx.from_numpy_array(weights, create_using=nx.DiGraph)
        self.obj = mglingam_obj


class UTIGSPMethod(ChangeMethod):
    @staticmethod
    def nm(): return 'UTIGSP'

    @staticmethod
    def ty(): return ValidMethodType.UTIGSP

    @staticmethod
    def dag_ty(): return [DAGType.PAG]

    @staticmethod
    def valid_exp_type():
        return ExpType.CHANGES

    def fit(self, data, truths, params, options):
        super(UTIGSPMethod, self).check_data(data, truths, params)
        from causaldag import unknown_target_igsp
        from conditional_independence import gauss_invariance_suffstat, MemoizedCI_Tester, MemoizedInvarianceTester, \
            gauss_invariance_test
        from conditional_independence import partial_correlation_suffstat, partial_correlation_test, kci_test

        obs_samples = np.array(data.datasets[0])
        iv_samples_list = [np.array(data.datasets[i]) for i in range(len(data.datasets)) if i != 0]
        num_settings = len(iv_samples_list)
        num_nodes = obs_samples.shape[1]
        assert num_nodes < obs_samples.shape[0]
        nodes = set(range(num_nodes))

        # Form sufficient statistics
        obs_suffstat = partial_correlation_suffstat(obs_samples)
        invariance_suffstat = gauss_invariance_suffstat(obs_samples, iv_samples_list)

        # Create conditional independence tester and invariance tester
        alpha = 1e-3
        alpha_inv = 1e-3
        ci_tester = MemoizedCI_Tester(kci_test,
                                      obs_suffstat, alpha=alpha)
        invariance_tester = MemoizedInvarianceTester(gauss_invariance_test, invariance_suffstat, alpha=alpha_inv)

        # check why kci_test is not working
        # parametric version:
        suffstat = partial_correlation_suffstat(obs_samples)
        suffstat_inv = gauss_invariance_suffstat(obs_samples, iv_samples_list)
        ci_tester = MemoizedCI_Tester(partial_correlation_test, suffstat, alpha=alpha)
        inv_tester = MemoizedInvarianceTester(gauss_invariance_test, suffstat_inv, alpha=alpha_inv)

        # Run UT-IGSP
        setting_list = [dict(known_interventions=[]) for _ in range(num_settings)]
        time_st = time.perf_counter()
        est_dag, est_targets_list = unknown_target_igsp(setting_list, nodes, ci_tester, invariance_tester)
        self.metrics = {'time': time.perf_counter() - time_st}
        adj = np.zeros((params["N"], params["N"]))
        for i, j in est_dag.arcs:
            adj[i][j] = 1
        self.pag = convert_npadj_to_causallearn_cpdag(adj)

        self.obj = est_dag, est_targets_list


class GIESMethod(ChangeMethod):
    @staticmethod
    def nm(): return 'GIES'

    @staticmethod
    def ty(): return ValidMethodType.GIES

    def fit(self, data, truths, params, options):
        super(GIESMethod, self).check_data(data, truths, params)
        from cdt.causality.graph import GIES
        gies_obj = GIES()

        raise NotImplementedError
        # GIES assumes known intervention targets

        time_st = time.perf_counter()
        # gies_obj = obj.predict(data_combined)
        nx_graph = gies_obj.create_graph_from_data(pd.DataFrame(data))
        self.metrics = {'time': time.perf_counter() - time_st}
        self.pag = convert_npadj_to_causallearn_cpdag(convert_nx_graph_to_adj(nx_graph))
        self.obj = gies_obj


class JPCMCIMethod(TimeMethod):
    @staticmethod
    def nm(): return 'JPCMCI'

    @staticmethod
    def ty(): return ValidMethodType.JPCMCI

    @staticmethod
    def dag_ty(): return [DAGType.TDAG, DAGType.WCG_PAG]

    def fit(self, data, truths, params, options):
        super(JPCMCIMethod, self).check_data(data, truths, params)
        from tigramite.independence_tests.gpdc import GPDC
        jpcmci_test = GPDC()

        jpcmci_dsets = data.datasets_with_dummynodes
        jpcmci_n_contexts = params['C']

        datasets_jpcmci, node_classification = gen_jpcmci_data(
            jpcmci_n_contexts, params['N'], min([jpcmci_dsets[k].shape[0] for k in jpcmci_dsets]),
            data.cnode, data.snode,
            data.node_classification, jpcmci_dsets)  # converts data to jpcmci format

        time_st = time.perf_counter()

        # Create a J-PCMCI+ object, passing the dataframe and (conditional)
        # independence test objects, as well as the observed temporal and spatial context nodes
        # and the indices of the dummies.
        from tigramite.jpcmciplus import JPCMCIplus
        JPCMCIplus = JPCMCIplus(dataframe=datasets_jpcmci,
                                cond_ind_test=jpcmci_test,
                                node_classification=node_classification,
                                verbosity=0)

        # Run J-PCMCI+
        jpcmci_obj = JPCMCIplus.run_jpcmciplus(tau_min=0,
                                               tau_max=params["TM"],
                                               pc_alpha=0.2)
        self.metrics = {'time': time.perf_counter() - time_st}

        graph = jpcmci_obj["graph"]
        assert graph.shape[2] == (params["TM"] + 1)
        self.wcg_pag, self.tdag = convert_pcmci_links_to_dag(graph, params)
        self.obj = jpcmci_obj


def gen_jpcmci_data(C, N, T, cnode, snode, node_classification, data_C):
    import tigramite.data_processing as pp

    """ Put data generated with gen_data in format for JPCMCI class """
    nb_domains = C
    system_indices = list(range(N))
    # decide which context variables should be latent, and which are observed
    observed_indices_time = [cnode]
    latent_indices_time = []
    observed_indices_space = [snode]
    latent_indices_space = []

    # all system variables are also observed, thus we get the following observed data
    observed_indices = system_indices + observed_indices_time + observed_indices_space
    data_observed = {key: data_C[key][:, observed_indices] for key in data_C}

    # Add one-hot-encoding of time-steps and dataset index to the observational data.
    # These are the values of the time and space dummy variables.
    dummy_data_time = np.identity(T)

    data_dict = {}
    for i in range(nb_domains):
        dummy_data_space = np.zeros((T,
                                     nb_domains))
        dummy_data_space[:, i] = 1.
        data_dict[i] = np.hstack((data_observed[i], dummy_data_time, dummy_data_space))

    # Define vector-valued variables including dummy variables as well as observed (system and context) variables
    nb_observed_context_nodes = len(observed_indices_time) + len(observed_indices_space)
    N = len(system_indices)
    process_vars = system_indices
    observed_temporal_context_nodes = list(range(N, N + len(observed_indices_time)))
    observed_spatial_context_nodes = list(range(N + len(observed_indices_time),
                                                N + len(observed_indices_time) + len(observed_indices_space)))
    time_dummy_index = N + nb_observed_context_nodes
    space_dummy_index = N + nb_observed_context_nodes + 1
    time_dummy = list(range(time_dummy_index, time_dummy_index + T))
    space_dummy = list(range(time_dummy_index + T, time_dummy_index + T + nb_domains))

    vector_vars = {i: [(i, 0)] for i in
                   process_vars + observed_temporal_context_nodes + observed_spatial_context_nodes}
    vector_vars[time_dummy_index] = [(i, 0) for i in time_dummy]
    vector_vars[space_dummy_index] = [(i, 0) for i in space_dummy]

    # Name all the variables and initialize the dataframe object
    # Be careful to use analysis_mode = 'multiple'
    sys_var_names = ['X_' + str(i) for i in process_vars]
    context_var_names = ['t-C_' + str(i) for i in observed_indices_time] + ['s-C_' + str(i) for i in
                                                                            observed_indices_space]
    var_names = sys_var_names + context_var_names + ['t-dummy', 's-dummy']

    dataframe = pp.DataFrame(
        data=data_dict,
        vector_vars=vector_vars,
        analysis_mode='multiple',
        var_names=var_names
    )

    # Classify all the nodes into system, context, or dummy
    node_classification_jpcmci = {i: node_classification[var] for i, var in enumerate(observed_indices)}
    node_classification_jpcmci.update({time_dummy_index: "time_dummy", space_dummy_index: "space_dummy"})

    return dataframe, node_classification_jpcmci


## Continuous MI attempt
class _TopChangesContinuousMethod(ContinuousMethod):
    @staticmethod
    def nm(): return 'TopChangesContinuousMethod'

    def fit(self, data, truths, params, options):
        raise DeprecationWarning
        super(_TopChangesContinuousMethod, self).check_data(data, truths, params)
        obj = TopologicalChangesCD(
            data, DataType.CONTINUOUS,
            options.score_type, params["T"],
            options.verbosity, options.logger,
            truths.is_true_edge)

        options.logger.info(f'TRUE Order {truths.order}')

        assert not obj.is_oracle and not obj.known_order
        self.topic_graph, self.top_order = obj.fit()
        super(_TopChangesContinuousMethod, self).eval_results(self, truths, options.logger)


class _OracleOrderChangesContinuousMethod(ContinuousMethod):
    @staticmethod
    def nm(): return 'OracleChangesContinuousMethod (O: true order)'

    def fit(self, data, truths, params, options):
        raise DeprecationWarning
        super(_OracleOrderChangesContinuousMethod, self).check_data(data, truths, params)
        obj = TopologicalChangesCD(
            data, DataType.CONTINUOUS,
            options.score_type, params["T"],
            options.verbosity, options.logger,
            truths.is_true_edge)

        options.logger.info(f'TRUE Order {truths.order}')

        obj.add_true_top_order(truths.order)
        assert not obj.is_oracle and obj.known_order
        self.topic_graph, self.top_order = obj.fit()
        super(_OracleOrderChangesContinuousMethod, self).eval_results(self, truths, options.logger)


# could use this alternative to TopChanges: TODO TODO  idea to combine both
class _TopContextMethod(ChangeMethod):
    @staticmethod
    def nm(): return 'TOPContext'

    def fit(self, data, truths, params, options):
        raise DeprecationWarning
        super(_TopContextMethod, self).check_data(data, truths, params)
        obj = Topic(
            data, options.score_type, DataType.CONT_MCONTEXT,
            options.score_type, params["T"],
            options.verbosity, options.logger,
            truths.is_true_edge)
        self.topic_graph, self.top_order = obj.fit()
        super(_TopContextMethod, self).eval_results(self, truths, options.logger)
