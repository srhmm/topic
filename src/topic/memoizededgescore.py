from topic.scoring.fitting import DataType, ScoreType
from topic.scoring.scoring import compute_edge_score


class MemoizedEdgeScore:
    data_type: DataType
    score_type: ScoreType

    def __init__(self, X, **scoring_parameters):
        self.X = X
        self.defaultargs = {
            "data_type": DataType.CONTINUOUS,
            "score_type": ScoreType.GAM,
            "regimes": None, "max_lag": 1, "n_ffs": 50,
            "is_true_edge": lambda i: lambda j: "",
            "oracle": False, "known_true_order": False, "true_idls": None, "true_A": None, "true_top_order": [],
            "lg": None, "vb": 0}

        assert all([arg in self.defaultargs.keys() for arg in scoring_parameters.keys()])
        self.__dict__.update((k, v) for k, v in self.defaultargs.items() if k not in scoring_parameters.keys())
        self.__dict__.update((k, v) for k, v in scoring_parameters.items() if k in self.defaultargs.keys())
        self._info = lambda st: (self.lg.info(st) if self.lg is not None else print(st)) if self.vb > 0 else None

        # Memoized info
        self.mdl_cache = {}
        self.idl_cache = {}

        # params needed for scoring edges
        self.mdl_params = dict(score_type=self.score_type,regimes=self.regimes,max_lag=self.max_lag,n_ffs=self.n_ffs,vb=self.vb,lg=self.lg)
        self.idl_params = dict(
            score_type=self.score_type,
            oracle=self.oracle,
            true_idls=self.true_idls,
            true_A=self.true_A,
            vb=self.vb,lg=self.lg
            )

    def score_edge(self, j, pa) -> int:
        """
        Evaluates score for a causal relationship pa(Xj)->Xj.

        :param j: Xj
        :param pa: pa(Xj)
        :return: score_up=score(Xpa->Xj)
        """
        hash_key = f'j_{str(j)}_pa_{str(pa)}'

        if self.mdl_cache.__contains__(hash_key):
            return self.mdl_cache[hash_key]


        score = compute_edge_score(
            self.X, self.data_type, covariates=pa, target=j, **self.mdl_params)

        self.mdl_cache[hash_key] = score
        return score
