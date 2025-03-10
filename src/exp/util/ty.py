from abc import abstractmethod, ABC
from enum import Enum

import networkx as nx
import numpy as np
from causallearn.graph import GeneralGraph

from exp.util.eval import compare_nx_digraph_to_dag, compare_ggnx_pag_to_dag, compare_np_digraph_to_dag, \
    convert_nx_graph_to_adj


class ExpType(Enum):
    """ Decides how data is generated """
    CONTINUOUS = 0
    TIME = 1
    CHANGES = 2
    TIMECHANGES = 3
    REGED = 4
    SACHS = 5
    TUEBINGEN = 6

    def is_synthetic(self):
        return self.value < 4

    def is_continuous(self):
        return self.value in [
            ExpType.CONTINUOUS.value, ExpType.REGED.value,
            ExpType.SACHS.value, ExpType.TUEBINGEN.value]

    def is_time(self):
        return self.value in [ExpType.TIME.value, ExpType.TIMECHANGES.value]

    def __eq__(self, other):
        return self.value == other.value


class ValidMethodType(Enum):
    TopCont = 0
    OracleCont = 1
    TopTime = 2
    OracleTime = 3
    TopChangesCont = 4
    OracleChangesCont = 5
    TopChangesTime = 6
    OracleChangesTime = 7
    PC = 8
    GES = 9
    GGES = 10
    FCI = 11
    CAM = 12
    CAM_UV = 13
    RESIT = 14
    GLOBE = 15
    SCORE = 16
    ICA_LINGAM = 17
    Direct_LINGAM = 18
    LINGAM = 19
    NOTEARS = 20
    GranDAG = 21
    DAGGNN = 22
    VarLINGAM = 23
    PCMCI = 24
    DyNOTEARS = 25
    Rhino = 26
    JPCMCI = 27
    JCI_FCI = 28
    JCI_PC = 29
    CDNOD = 30
    CDNODT = 31
    UTIGSP = 32
    GIES = 33
    MGLINGAM = 34
    DAS = 35
    NOGAM = 36
    SCORE2 = 37
    CAM2 = 38
    GES_CV = 39
    GES_MARG = 40
    R2SORT = 41
    VARSORT = 42
    RANDSORT = 43

    def __eq__(self, other):
        return self.value == other.value


class DAGType(Enum):
    """ result that a method returns, DAG or PAG and/or window DAG """
    DAG = 0
    PAG = 1
    WCG = 2
    TDAG = 3
    WCG_PAG = 4
    TPAG = 5
    IV_IND = 6  # interventions: dimension n_nodes x n_contexts, 1 if node i changes in context j


class MethodType(ABC):

    @staticmethod
    @abstractmethod
    def dag_ty() -> DAGType:
        pass

    @staticmethod
    @abstractmethod
    def ty() -> ExpType:
        pass

    @staticmethod
    @abstractmethod
    def nm() -> str:
        pass

    def __init__(self):
        self.dag: nx.DiGraph = None
        self.pag: GeneralGraph = None
        self.tdag: np.array = None
        self.tpag: np.array = None
        self.wcg: np.array = None
        self.wcg_pag: GeneralGraph = None
        self.metrics: dict = {}
        self.obj = None

    @abstractmethod
    def fit(self, data, truths, params, options):
        pass

    def get_result_dag(self):
        res_dag: np.array = convert_nx_graph_to_adj(self.dag) if DAGType.DAG in self.dag_ty() \
            else self.wcg if DAGType.WCG in self.dag_ty() \
            else self.wcg_pag if DAGType.WCG_PAG in self.dag_ty() \
            else self.pag.graph if DAGType.PAG in self.dag_ty() else self.tpag if DAGType.TPAG in self.dag_ty() else None
        assert res_dag is not None
        return res_dag

    def add_result_dag(self, res_dag):
        if DAGType.DAG in self.dag_ty():
            self.dag = nx.from_numpy_array(res_dag, create_using=nx.DiGraph)
        elif DAGType.WCG in self.dag_ty():
            self.wcg = res_dag
        elif DAGType.WCG_PAG in self.dag_ty():
            self.wcg_pag = res_dag
        elif DAGType.PAG in self.dag_ty():
            self.pag = res_dag
        elif DAGType.TPAG in self.dag_ty():
            self.tpag = res_dag

    def eval_results(self, truths, options):
        if DAGType.DAG in self.dag_ty():
            assert self.dag is not None
            self.metrics.update(compare_nx_digraph_to_dag(self.dag, truths.graph, not options.skip_sid))
        # if method returns a partially directed dag
        if DAGType.PAG in self.dag_ty():
            assert self.pag is not None
            self.metrics.update(compare_ggnx_pag_to_dag(self.pag, truths.graph, options.enable_SID_call))

        #  if method also returns a window dag
        if DAGType.WCG in self.dag_ty():
            assert self.wcg is not None
            self.metrics.update(
                compare_np_digraph_to_dag(self.wcg, truths.timed_dag, options.enable_SID_call, has_time_lags=True))

        if DAGType.WCG_PAG in self.dag_ty():
            assert self.wcg_pag is not None
            self.metrics.update(
                compare_np_digraph_to_dag(
                    self.wcg_pag, truths.timed_dag, options.enable_SID_call, allow_cycles=True, has_time_lags=True))

        # time series
        if DAGType.TDAG in self.dag_ty():
            assert self.tdag is not None
            self.metrics.update(
                compare_np_digraph_to_dag(
                    self.tdag, truths.graph, options.enable_SID_call, allow_cycles=False, has_time_lags=False))

        if DAGType.TPAG in self.dag_ty():
            assert self.tpag is not None
            self.metrics.update(
                compare_np_digraph_to_dag(
                    self.tpag, truths.graph, options.enable_SID_call, allow_cycles=True, has_time_lags=False))

        log_results(self.metrics, self.nm(), options.logger, options.exp_type.is_time())


class ContinuousMethod(MethodType, ABC):
    def __init__(self):
        super().__init__()
        self.metrics = {}
        self.dag = nx.DiGraph()
        self.pag = nx.DiGraph()
        self.top_order = []

    @staticmethod
    def check_data(data, truths, params):
        assert data.shape == (params['S'], params['N'])
        assert len(truths.graph.nodes) == params['N']

    @staticmethod
    def valid_exp_type():
        return ExpType.CONTINUOUS


class TimeMethod(MethodType, ABC):
    """ used both for continuous x time and changes x time"""

    def __init__(self):
        super().__init__()
        self.metrics = {}
        self.wcg = np.zeros((1, 1))
        self.wcg_pag = np.zeros((1, 1))
        self.tdag = np.zeros((1, 1))
        self.pag = np.zeros((1, 1))
        self.top_order = []

    @staticmethod
    def check_data(data, truths, params):
        """ takes multiple datasets as input, continous x time -> use datasets[0], changes x time -> use all """
        assert len(data.datasets) == params['C']
        assert all(data.datasets[c].shape == (params['S'], params['N']) for c in range(params['C']))

    @staticmethod
    def valid_exp_type():
        return ExpType.TIME


class ChangeMethod(MethodType, ABC):
    def __init__(self):
        super().__init__()
        self.metrics = {}
        self.dag = nx.DiGraph()
        self.pag = nx.DiGraph()
        self.result_obj = None
        self.top_order = []

    @staticmethod
    def check_data(data, truths, params):
        assert len(data.datasets) == params['C']
        assert all(data.datasets[c].shape == (params['S'], params['N']) for c in range(params['C']))
        # assert len(truths.dag) == params['N']

    @staticmethod
    def valid_exp_type():
        return ExpType.CHANGES


class NoiseType(Enum):
    """ Decides additive noise type for data generation"""
    GAUSS = 0
    UNIF = 1
    HTSKD = 2

    def to_noise(self):
        vals = [
            (NoiseType.GAUSS, lambda random_state: random_state.standard_normal),
            (NoiseType.UNIF, lambda random_state: random_state.uniform),
            (None, None)  # not used yet
        ]
        return vals[self.value]

    def __eq__(self, other):
        return self.value == other.value

    def __str__(self):
        names = ["gaussian", "uniform", "hskd"]
        return names[self.value]


class InterventionType(Enum):
    CoefficientChange = 0
    HardIntervention = 1
    NoiseShift = 2
    NoiseScaling = 3
    NoiseHeteroskedasticity = 4

    def __eq__(self, other):
        return self.value == other.value

class DagType(Enum):
    ERDOS_CDT = 'erdos_cdt'
    ERDOS_CD = 'erdos_cd'
    SCALE_FREE = 'sf'
    RANDOM = 'rand'

class FunType(Enum):
    LIN = 0
    TAYLOR = 1
    NLIN = 2

    def __str__(self):
        names = ["linear", "nonlin_cont", "nonlin_time"]
        return names[self.value]

    def to_fun(self):
        vals = [
            (FunType.LIN, lambda x: x),
            (FunType.TAYLOR, lambda x: _taylor_fun(x)),
            (FunType.NLIN, lambda x: x + 5.0 * x ** 2 * np.exp(-(x ** 2) / 20.0))
        ]
        return vals[self.value]

    def __eq__(self, other):
        return self.value == other.value


def _taylor_fun(x):
    raise NotImplementedError


def log_results(metrics, nm, logger, is_time):
    logger.info(
        f"\tResult: {nm}" + f"\tf1: {metrics['f1']:.2f}\t" + ", ".join(
            [f'{met}: {metrics[met]:.2f}' for met in metrics]))
