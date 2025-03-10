import time
from abc import ABC

import networkx as nx
import numpy as np
import pandas as pd
import torch
from causallearn.graph import GeneralGraph

from exp.util.ty import ContinuousMethod, ValidMethodType, DAGType
from exp.util.dag_utils import dag2cpdag, convert_npadj_to_causallearn_cpdag
from exp.util.eval import convert_nx_graph_to_adj
from topic.scoring.fitting import DataType
from topic.topic import Topic

""" Direction 1: OBSERVATIONAL, continuous """


class TopContMethod(ContinuousMethod, ABC):
    """ Main method for top. cd. from continuous data """

    @staticmethod
    def nm(): return 'TOPCont'

    @staticmethod
    def ty(): return ValidMethodType.TopCont

    @staticmethod
    def dag_ty(): return [DAGType.DAG]

    def fit(self, data, truths, params, options):
        super(TopContMethod, self).check_data(data, truths, params)
        hypparams = dict(score_type=options.score_type,vb=options.verbosity,lg=options.logger,true_graph=truths.graph)
        top = Topic(**hypparams)
        time_st = time.perf_counter()
        self.dag, self.top_order = top.fit(data)
        self.metrics = {'time': time.perf_counter() - time_st}
        self.obj = top


class OracleContMethod(ContinuousMethod, ABC):
    """ Oracle: known true causal ordering, consider nodes in true order and add outgoing edges"""

    @staticmethod
    def nm(): return 'TOPOracleCont'

    @staticmethod
    def ty(): return ValidMethodType.OracleCont

    @staticmethod
    def dag_ty(): return [DAGType.DAG]

    def fit(self, data, truths, params, options):
        super(OracleContMethod, self).check_data(data, truths, params)
        obj = Topic(
            data, DataType.CONTINUOUS,
            options.score_type, params["T"],
            options.verbosity, options.logger,
            truths.is_true_edge)

        obj.add_true_top_order(truths.order)
        time_st = time.perf_counter()
        self.dag, self.top_order = obj.fit()
        self.metrics = {'time': time.perf_counter() - time_st}
        self.obj = obj


class PCMethod(ContinuousMethod, ABC):
    """causal-learn implementation, PC HSIC"""

    @staticmethod
    def nm():
        return 'PC'

    @staticmethod
    def ty():
        return ValidMethodType.PC

    @staticmethod
    def dag_ty(): return [DAGType.PAG]

    def fit(self, data, truths, params, options):
        super(PCMethod, self).check_data(data, truths, params)
        from causallearn.search.ConstraintBased.PC import pc
        options.methodparams_pc_indep_test = 'kci'  # todo change for linear exp

        time_st = time.perf_counter()
        pc_obj = pc(data, indep_test=options.methodparams_pc_indep_test)
        self.pag: GeneralGraph = pc_obj.G
        self.metrics = {'time': time.perf_counter() - time_st}
        self.obj = pc_obj
class GESNLMethod(ContinuousMethod, ABC):
    @staticmethod
    def nm(): return 'GESCV'

    @staticmethod
    def ty(): return ValidMethodType.GES_CV

    @staticmethod
    def dag_ty(): return [DAGType.PAG]

    def fit(self, data, truths, params, options):
        super(GESNLMethod, self).check_data(data, truths, params)
        # cdt implementation:
        # import cdt.causality.graph as algs
        # obj = algs.GES()
        # datafr = pd.DataFrame(data)
        # self.untimed_graph = obj.predict(datafr)

        from causallearn.search.ScoreBased.GES import ges
        options.methodparams_ges_score =  "local_score_CV_multi" #"local_score_marginal_multi"
        options.logger.info(f"GES: using {options.methodparams_ges_score} ")
        options.methodparams_ges_maxP = None  # needed?
        time_st = time.perf_counter()
        ges_obj = ges(data, options.methodparams_ges_score)  # , maxP, parameters)
        self.pag: GeneralGraph = ges_obj['G']
        self.metrics = {'time': time.perf_counter() - time_st}
        self.obj = ges_obj
class GESMargMethod(ContinuousMethod, ABC):
    @staticmethod
    def nm(): return 'GESMARG'

    @staticmethod
    def ty(): return ValidMethodType.GES_MARG

    @staticmethod
    def dag_ty(): return [DAGType.PAG]

    def fit(self, data, truths, params, options):
        super(GESMargMethod, self).check_data(data, truths, params)
        # cdt implementation:
        # import cdt.causality.graph as algs
        # obj = algs.GES()
        # datafr = pd.DataFrame(data)
        # self.untimed_graph = obj.predict(datafr)

        from causallearn.search.ScoreBased.GES import ges
        options.methodparams_ges_score =  "local_score_marginal_multi"
        options.logger.info(f"GES: using {options.methodparams_ges_score} ")
        options.methodparams_ges_maxP = None  # needed?
        time_st = time.perf_counter()
        ges_obj = ges(data, options.methodparams_ges_score)  # , maxP, parameters)
        self.pag: GeneralGraph = ges_obj['G']
        self.metrics = {'time': time.perf_counter() - time_st}
        self.obj = ges_obj

class GESMethod(ContinuousMethod, ABC):
    @staticmethod
    def nm(): return 'GES'

    @staticmethod
    def ty(): return ValidMethodType.GES

    @staticmethod
    def dag_ty(): return [DAGType.PAG]

    def fit(self, data, truths, params, options):
        super(GESMethod, self).check_data(data, truths, params)
        # cdt implementation:
        # import cdt.causality.graph as algs
        # obj = algs.GES()
        # datafr = pd.DataFrame(data)
        # self.untimed_graph = obj.predict(datafr)

        from causallearn.search.ScoreBased.GES import ges
        options.methodparams_ges_score = "local_score_BIC"  # "local_score_marginal_multi"   "local_score_cv_multi"
        options.logger.info(f"GES: using {options.methodparams_ges_score} ")
        options.methodparams_ges_maxP = None  # needed?
        time_st = time.perf_counter()
        ges_obj = ges(data, options.methodparams_ges_score)  # , maxP, parameters)
        self.pag: GeneralGraph = ges_obj['G']
        self.metrics = {'time': time.perf_counter() - time_st}
        self.obj = ges_obj



class R2SORTMethod( ContinuousMethod, ABC):
    @staticmethod
    def nm(): return 'R2SORT'

    @staticmethod
    def ty(): return ValidMethodType.R2SORT

    @staticmethod
    def dag_ty(): return [DAGType.DAG]

    def fit(self, data, truths, params, options):
        from CausalDisco.baselines import (
            random_sort_regress,
            var_sort_regress,
            r2_sort_regress
        )

        super(R2SORTMethod, self).check_data(data, truths, params)

        time_st = time.perf_counter()
        self.dag = nx.from_numpy_array(r2_sort_regress(data), create_using=nx.DiGraph)
        self.metrics = {'time': time.perf_counter() - time_st}

class VARSORTMethod( ContinuousMethod, ABC):
    @staticmethod
    def nm(): return 'VARSORT'

    @staticmethod
    def ty(): return ValidMethodType.VARSORT

    @staticmethod
    def dag_ty(): return [DAGType.DAG]

    def fit(self, data, truths, params, options):
        super(VARSORTMethod, self).check_data(data, truths, params)

        time_st = time.perf_counter()
        self.dag = nx.from_numpy_array(var_sort_regress(data), create_using=nx.DiGraph)
        self.metrics = {'time': time.perf_counter() - time_st}

class RANDSORTMethod( ContinuousMethod, ABC):
    @staticmethod
    def nm(): return 'RANDSORT'

    @staticmethod
    def ty(): return ValidMethodType.RANDSORT

    @staticmethod
    def dag_ty(): return [DAGType.DAG]

    def fit(self, data, truths, params, options):
        super(RANDSORTMethod, self).check_data(data, truths, params)

        time_st = time.perf_counter()
        self.dag = nx.from_numpy_array(random_sort_regress(data), create_using=nx.DiGraph)
        self.metrics = {'time': time.perf_counter() - time_st}

class GGESMethod(GESMethod, ContinuousMethod, ABC):
    @staticmethod
    def nm(): return 'GGES'

    @staticmethod
    def ty(): return ValidMethodType.GGES

    def fit(self, data, truths, params, options):
        super(GGESMethod, self).check_data(data, truths, params)

        from causallearn.search.ScoreBased.GES import ges

        options.methodparams_ges_score = "local_score_marginal_multi"  # or "local_score_cv_multi"
        options.logger.info(f"GGES: using {options.methodparams_ges_score}")
        time_st = time.perf_counter()
        ges_obj = ges(data, options.methodparams_ges_score)  # , maxP, parameters)
        self.pag: GeneralGraph = ges_obj['G']
        self.metrics = {'time': time.perf_counter() - time_st}
        self.obj = ges_obj


class FCIMethod(ContinuousMethod, ABC):
    @staticmethod
    def nm():
        return 'FCI'

    @staticmethod
    def ty():
        return ValidMethodType.FCI

    @staticmethod
    def dag_ty(): return [DAGType.PAG]

    def fit(self, data, truths, params, options):
        super(FCIMethod, self).check_data(data, truths, params)
        from causallearn.search.ConstraintBased.FCI import fci

        time_st = time.perf_counter()
        self.pag, edges = fci(data,
                              independence_test_method='kci')  # todo change for linear exp        # customized parameters
        # PARAMETERS: fci(data, independence_test_method, alpha, depth, max_path_length,
        #               verbose, background_knowledge, cache_variables_map)
        self.metrics = {'time': time.perf_counter() - time_st}
        self.obj = self.pag, edges


class RESITMethod(ContinuousMethod, ABC):
    @staticmethod
    def nm(): return 'RESIT'

    @staticmethod
    def ty(): return ValidMethodType.RESIT

    @staticmethod
    def dag_ty(): return [DAGType.DAG]

    def fit(self, data, truths, params, options):
        super(RESITMethod, self).check_data(data, truths, params)
        from sklearn.ensemble import RandomForestRegressor
        import lingam as lng

        reg = RandomForestRegressor(max_depth=4, random_state=0)
        resit_obj = lng.RESIT(regressor=reg)
        time_st = time.perf_counter()
        resit_obj.fit(data)
        self.metrics = {'time': time.perf_counter() - time_st}
        self.dag = nx.from_numpy_array(resit_obj.adjacency_matrix_.T, create_using=nx.DiGraph())

        self.top_order = resit_obj.causal_order_
        self.obj = resit_obj


class CAMMethod(ContinuousMethod, ABC):
    """ cdt """

    @staticmethod
    def nm(): return 'CAM'

    @staticmethod
    def ty(): return ValidMethodType.CAM

    @staticmethod
    def dag_ty(): return [DAGType.DAG]

    def fit(self, data, truths, params, options):
        super(CAMMethod, self).check_data(data, truths, params)
        from cdt.causality.graph import CAM
        obj = CAM(score="nonlinear")
        time_st = time.perf_counter()
        cam_obj = obj.predict(pd.DataFrame(data))
        self.metrics = {'time': time.perf_counter() - time_st}
        self.dag, self.obj = cam_obj, cam_obj
        self.pag = dag2cpdag(convert_nx_graph_to_adj(self.dag))


class CAMUVMethod(ContinuousMethod, ABC):
    """ causal discovery toolbox """

    @staticmethod
    def nm():
        return 'CAMUV'

    @staticmethod
    def ty():
        return ValidMethodType.CAM_UV

    @staticmethod
    def dag_ty():
        return [DAGType.PAG]

    def fit(self, data, truths, params, options):
        num_explanatory_vals_prev = 3
        max_interactions = max([len([1 if j == n else 0 for (i, j) in truths.graph.edges]) for n in truths.graph.nodes])
        print("Setting max interactions to: ", max_interactions)
        num_explanatory_vals = max_interactions

        super(CAMUVMethod, self).check_data(data, truths, params)
        from causallearn.search.FCMBased.lingam import CAMUV

        time_st = time.perf_counter()
        P, U = CAMUV.execute(data, 0.05, num_explanatory_vals)
        self.metrics = {'time': time.perf_counter() - time_st}

        nx_pag = nx.DiGraph()
        nx_pag.add_nodes_from(set(range(len(P))))
        for i, result in enumerate(P):
            if not len(result) == 0:
                print("child: " + str(i) + ",  parents: " + str(result))
                for j in result:
                    nx_pag.add_edge(j, i)  # may contain cycles

        self.pag = convert_npadj_to_causallearn_cpdag(convert_nx_graph_to_adj(nx_pag))
        self.obj = P, U


class GLOBEMethod(ContinuousMethod, ABC):
    @staticmethod
    def nm(): return 'GLOBE'

    @staticmethod
    def ty(): return ValidMethodType.GLOBE

    @staticmethod
    def dag_ty(): return [DAGType.DAG]

    def fit(self, data, truths, params, options):
        from baselines.globe.globeWrapper import GlobeWrapper
        super(GLOBEMethod, self).check_data(data, truths, params)
        # Setting the max interactions to the ground truth
        # worse performance: max_interactions = max([len([1 if j==n else 0 for (i,j ) in truths.graph.edges] ) for n in truths.graph.nodes])

        max_interactions = 3
        print("Setting max interactions to: ", max_interactions)

        obj = GlobeWrapper(max_interactions, False, True)
        data = pd.DataFrame(data)
        data.to_csv("temp.csv", header=False, index=False)
        obj.loadData("temp.csv")
        time_st = time.perf_counter()
        adjacency = obj.run()
        self.metrics = {'time': time.perf_counter() - time_st}

        self.dag = nx.from_numpy_array(adjacency, create_using=nx.DiGraph())
        self.obj = obj
class SCORE2Method(ContinuousMethod, ABC):
    @staticmethod
    def nm(): return 'SCORE2'

    @staticmethod
    def ty(): return ValidMethodType.SCORE2

    @staticmethod
    def dag_ty(): return [DAGType.DAG]

    def fit(self, data, truths, params, options):
        super(SCORE2Method, self).check_data(data, truths, params)

        from baselines.dodiscover.toporder.score import SCORE
        from baselines.dodiscover import make_context
        score = SCORE()  # or DAS() or NoGAM() or CAM()
        time_st = time.perf_counter()

        score_context = make_context().variables(data=pd.DataFrame(data)).build()
        score.learn_graph(pd.DataFrame(data), score_context)

        # SCORE estimates a directed acyclic graph (DAG) and the topoological order
        # of the nodes in the graph. SCORE is consistent in the infinite samples
        # limit, meaning that it might return faulty estimates due to the finiteness
        # of the data.
        top_order_SCORE = score.order_graph_

        # `score_full_dag.png` visualizes the fully connected DAG representation of
        # the inferred topological ordering.
        # `score_dag.png` visualizes the fully connected DAG after pruning with
        # sparse regression.
        #dot_graph = draw(graph, name="DAG after pruning")
        #dot_graph.render(outfile="score_dag.png", view=True)

        #dot_graph = draw(order_graph, name="Fully connected DAG")
        #dot_graph.render(outfile="score_full_dag.png", view=True)

        self.metrics = {'time': time.perf_counter() - time_st}

        self.dag = score.graph_
        adj = nx.to_numpy_array(score.graph_)
        self.pag = dag2cpdag(adj)
        self.obj = adj, top_order_SCORE
class DASMethod(ContinuousMethod, ABC):
    @staticmethod
    def nm(): return 'DAS'

    @staticmethod
    def ty(): return ValidMethodType.DAS

    @staticmethod
    def dag_ty(): return [DAGType.DAG]

    def fit(self, data, truths, params, options):
        super(DASMethod, self).check_data(data, truths, params)

        from baselines.dodiscover.toporder.das import DAS
        from baselines.dodiscover import make_context
        score = DAS()  # or DAS() or NoGAM() or CAM()
        time_st = time.perf_counter()
        score_context = make_context().variables(data=pd.DataFrame(data)).build()
        score.learn_graph(pd.DataFrame(data), score_context)

        top_order_SCORE = score.order_graph_
        self.metrics = {'time': time.perf_counter() - time_st}

        self.dag = score.graph_
        adj = nx.to_numpy_array(score.graph_)
        self.pag = dag2cpdag(adj)
        self.obj = adj, top_order_SCORE
class CAM2Method(ContinuousMethod, ABC):
    @staticmethod
    def nm(): return 'CAM2'

    @staticmethod
    def ty(): return ValidMethodType.CAM2

    @staticmethod
    def dag_ty(): return [DAGType.DAG]

    def fit(self, data, truths, params, options):
        super(CAM2Method, self).check_data(data, truths, params)

        from baselines.dodiscover.toporder.cam import CAM
        from baselines.dodiscover import make_context
        score = CAM()  # or DAS() or NoGAM() or CAM()
        time_st = time.perf_counter()
        score_context = make_context().variables(data=pd.DataFrame(data)).build()
        score.learn_graph(pd.DataFrame(data), score_context)

        top_order_SCORE = score.order_graph_
        self.metrics = {'time': time.perf_counter() - time_st}

        self.dag = score.graph_
        adj = nx.to_numpy_array(score.graph_)
        self.pag = dag2cpdag(adj)
        self.obj = adj, top_order_SCORE

class NOGAMMethod(ContinuousMethod, ABC):
    @staticmethod
    def nm(): return 'NOGAM'

    @staticmethod
    def ty(): return ValidMethodType.NOGAM

    @staticmethod
    def dag_ty(): return [DAGType.DAG]

    def fit(self, data, truths, params, options):
        super(NOGAMMethod, self).check_data(data, truths, params)

        from baselines.dodiscover.toporder.nogam import NoGAM
        from baselines.dodiscover import make_context
        score = NoGAM()  # or DAS() or NoGAM() or CAM()
        time_st = time.perf_counter()
        score_context = make_context().variables(data=pd.DataFrame(data)).build()
        score.learn_graph(pd.DataFrame(data), score_context)
        top_order_SCORE = score.order_graph_
        self.metrics = {'time': time.perf_counter() - time_st}


        self.dag = score.graph_
        adj = nx.to_numpy_array(score.graph_)
        self.pag = dag2cpdag(adj)
        self.obj = adj, top_order_SCORE


class SCOREMethod(ContinuousMethod, ABC):
    @staticmethod
    def nm(): return 'SCORE'

    @staticmethod
    def ty(): return ValidMethodType.SCORE

    @staticmethod
    def dag_ty(): return [DAGType.DAG]

    def fit(self, data, truths, params, options):
        super(SCOREMethod, self).check_data(data, truths, params)
        from baselines.score.stein import SCORE
        import torch

        # SCORE hyper-parameters
        eta_G = 0.001
        eta_H = 0.001
        cam_cutoff = 0.001

        time_st = time.perf_counter()
        A_SCORE, top_order_SCORE = SCORE(torch.from_numpy(data), eta_G, eta_H, cam_cutoff)
        self.metrics = {'time': time.perf_counter() - time_st}

        self.dag = nx.from_numpy_array(A_SCORE, create_using=nx.DiGraph())
        self.pag = dag2cpdag(A_SCORE)
        self.obj = A_SCORE, top_order_SCORE


class ICALINGAMMethod(ContinuousMethod, ABC):
    """causallearn implementation"""

    @staticmethod
    def nm(): return 'ICALINGAM'

    @staticmethod
    def ty(): return ValidMethodType.ICA_LINGAM

    @staticmethod
    def dag_ty(): return [DAGType.DAG]

    def fit(self, data, truths, params, options):
        super(ICALINGAMMethod, self).check_data(data, truths, params)
        from causallearn.search.FCMBased import lingam

        model = lingam.ICALiNGAM()  # random_state, max_iter=1000)
        time_st = time.perf_counter()
        model.fit(data)
        self.metrics = {'time': time.perf_counter() - time_st}

        self.dag = nx.from_numpy_array(model.adjacency_matrix_, create_using=nx.DiGraph())
        self.top_order = model.causal_order_
        self.obj = model


class DirectLINGAMMethod(ContinuousMethod, ABC):
    """causallearn implementation"""

    @staticmethod
    def nm(): return 'DirectLINGAM'

    @staticmethod
    def ty(): return ValidMethodType.Direct_LINGAM

    @staticmethod
    def dag_ty(): return [DAGType.DAG]

    def fit(self, data, truths, params, options):
        super(DirectLINGAMMethod, self).check_data(data, truths, params)
        from causallearn.search.FCMBased import lingam

        model = lingam.DirectLiNGAM()  # random_state, prior_knowledge, apply_prior_knowledge_softly, measure)
        time_st = time.perf_counter()
        model.fit(data)
        self.metrics = {'time': time.perf_counter() - time_st}

        self.dag = nx.from_numpy_array(model.adjacency_matrix_, create_using=nx.DiGraph())
        self.top_order = model.causal_order_
        self.obj = model


class LINGAMMethod(ContinuousMethod, ABC):
    """cdt implementation"""

    @staticmethod
    def nm(): return 'LINGAM'

    @staticmethod
    def ty(): return ValidMethodType.LINGAM

    @staticmethod
    def dag_ty(): return [DAGType.DAG]

    def fit(self, data, truths, params, options):
        super(LINGAMMethod, self).check_data(data, truths, params)

        from cdt.causality.graph import LiNGAM
        obj = LiNGAM()  # no hyperparams needed
        time_st = time.perf_counter()
        dag = obj.predict(pd.DataFrame(data))
        self.metrics = {'time': time.perf_counter() - time_st}
        self.dag = dag
        self.obj = obj


class NOTEARSMethod(ContinuousMethod, ABC):
    # https://arxiv.org/abs/2104.05441
    @staticmethod
    def nm(): return 'NOTEARS'

    @staticmethod
    def ty(): return ValidMethodType.NOTEARS

    @staticmethod
    def dag_ty(): return [DAGType.DAG]

    def fit(self, data, truths, params, options):
        super(NOTEARSMethod, self).check_data(data, truths, params)
        from baselines.notears.notears.nonlinear import notears_nonlinear, NotearsMLP
        lambda1 = 0.1  # not working: 0.01
        lambda2 = 0.1  # not working: 0.01
        # todo give true n edges here:
        expected_num_edges = max(len(truths.graph.edges),
                                 1)  # give true number of edges as background knowledge, min 1 to avoid div problems

        time_st = time.perf_counter()
        notears_obj = NotearsMLP(dims=[params["N"], expected_num_edges, 1], bias=True)
        data_float32 = np.array(torch.from_numpy(data).to(torch.float32))
        dag_adj = notears_nonlinear(notears_obj, data_float32, lambda1, lambda2)
        self.metrics = {'time': time.perf_counter() - time_st}
        self.dag = nx.from_numpy_array(dag_adj, create_using=nx.DiGraph())

        self.obj = notears_obj


class GranDAGMethod(ContinuousMethod, ABC):
    # https://github.com/kurowasan/GraN-DAG
    @staticmethod
    def nm(): return 'GranDAG'

    @staticmethod
    def ty(): return ValidMethodType.GranDAG

    @staticmethod
    def dag_ty(): return [DAGType.PAG]

    def fit(self, data, truths, params, options):
        super(GranDAGMethod, self).check_data(data, truths, params)
        raise NotImplementedError
        obj = None
        data = pd.DataFrame(data)
        self.topic_graph = obj.predict(data)
        self.top_order = list(nx.topological_sort(self.topic_graph))
        self.obj = obj


class DAGGNNMethod(ContinuousMethod, ABC):
    # https://github.com/fishmoon1234/DAG-GNN
    @staticmethod
    def nm(): return 'DAGGNN'

    @staticmethod
    def ty(): return ValidMethodType.DAGGNN

    @staticmethod
    def dag_ty(): return [DAGType.PAG]

    def fit(self, data, truths, params, options):
        super(DAGGNNMethod, self).check_data(data, truths, params)
        raise NotImplementedError
        obj = None
        data = pd.DataFrame(data)
        self.topic_graph = obj.predict(data)
        self.top_order = list(nx.topological_sort(self.topic_graph))
        self.obj = obj
