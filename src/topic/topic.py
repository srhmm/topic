import itertools

import networkx as nx
import numpy as np

from topic.memoizededgescore import MemoizedEdgeScore
from topic.scoring.fitting import ScoreType, DataType
from topic.util.util import is_insignificant, compare_adj


class Topic:
    data_type: DataType
    score_type: ScoreType
    known_true_order: False
    extra_refinement: True
    true_graph: None
    true_top_order: []
    candidates: []
    scores: MemoizedEdgeScore

    def __init__(self, **kwargs):
        r""" Topological Causal Discovery
        :param optargs: optional arguments

        :Keyword Arguments:
        * *data_type* (``DataType``) -- continuous, time series, or multi-context data
        * *score_type* (``ScoreType``) -- information-theoretic score
        * *true_graph* (``nx.DiGraph``) -- for logging
        * *known_true_order* -- oracle version
        * *extra_refinement* -- refine parent sets, slower but better specificity
        * *lg* (``logging``) -- logger if verbosity>0
        * *vb* (``int``) -- verbosity level
        """
        self.defaultargs = {
            "data_type": DataType.CONTINUOUS,
            "score_type": ScoreType.GAM,
            "true_graph": None,
            "known_true_order": False,
            "extra_refinement": True,
            "lg": None, "vb": 0}

        #assert all([arg in self.defaultargs.keys() for arg in kwargs.keys()])
        self.__dict__.update((k, v) for k, v in self.defaultargs.items() if k not in kwargs.keys())
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in self.defaultargs.keys())

        def _info(st, strength=0):
            (self.lg.info(st) if self.lg is not None else print(st)) if self.vb + strength > 0 else None

        self._info = _info
        self.is_true_edge = ( lambda i: lambda j: "") if self.true_graph is None else  \
            (lambda node: lambda other: 'causal' if self.true_graph.has_edge(node, other) else (
                'rev' if self.true_graph.has_edge(other, node) else 'spurious'))
        self.true_top_order = None if self.true_graph is None else list(nx.topological_sort(self.true_graph))
        assert (not self.known_true_order or self.true_graph is not None)

        self.topic_graph = nx.DiGraph()
        self.topological_order = []

        self.scoring_params = dict(
            data_type=self.data_type,
            score_type=self.score_type
        )

    def fit(self, X):
        self._info(f"***TOPIC {self.data_type.value, self.score_type.value}***")
        self._check_X(X)
        self.topic_graph.add_nodes_from(range(self.N))
        self.candidates = list(range(self.N))
        self.scores = MemoizedEdgeScore(self.X, **self.scoring_params)

        self.order_nodes()

        if self.true_graph is not None:
            self._info(f"Result: {', '.join([f'{ky}:{val}' for ky, val in self.get_metrics(self.true_graph).items()])}")
        self._info("")
        return self.topic_graph, self.topological_order

    def order_nodes(self):
        it = 0
        while it < self.N:
            source = self.get_next_node(self.candidates if not self.known_true_order else self.true_top_order[it])
            self.candidates.remove(source)
            self.topological_order.append(source)
            it += 1
            self._info(f"\t{it}. Source: {source}\t current {self.topological_order}, true {self.true_top_order}", -1)

            self.add_edges(source)
            self.refine_edges(source)

        if self.extra_refinement:
            self.refinement_phase()

    def add_edges(self, source):
        for node in self.candidates:  # todo could use prio q here
            if node in self.topological_order or node == source or self.has_cycle(source, node):
                continue
            gain = self._addition_gain(node, source)
            if self._significant(gain):
                self._add_edge(source, node)

    def _addition_gain(self, node, source):
        parents = list(self.topic_graph.predecessors(node)).copy()
        old_score = self._score(parents, node)
        parents.append(source)
        new_score = self._score(parents, node)
        gain = self._gain(new_score, old_score)
        return gain

    def refine_edges(self, source):
        parents = list(self.topic_graph.predecessors(source))
        n_removed = 0
        while n_removed < len(parents):
            best_diff = np.inf
            best_parent = None
            old_score = self._score(parents, source)

            for parent in parents:
                new_parents = parents.copy()
                new_parents.remove(parent)
                if len(new_parents) == 0:
                    continue
                new_score = self._score(new_parents, source)
                diff = new_score - old_score

                if diff < best_diff and self._significant(diff):
                    best_diff = diff
                    best_parent = parent

            if best_parent is not None:
                self._remove_edge(best_parent, source)
                parents.remove(best_parent)
                n_removed += 1
            else:
                break

    """ utils """

    def _add_edge(self, parent, child, vb=-1):
        self.topic_graph.add_edge(parent, child)
        self._info(f"\tAdding {self.is_true_edge(parent)(child)} edge {parent} -> {child}", vb)

    def _remove_edge(self, parent, child, vb=-1):
        self.topic_graph.remove_edge(parent, child)
        self._info(f"\tRemoving {self.is_true_edge(parent)(child)} edge {parent} -> {child}", vb)

    def _score(self, parents, child, efficient=False):
        return self.scores_efficient.score_edge(child, parents) if efficient else self.scores.score_edge(child, parents)

    def _gain(self, new_score, old_score):
        return new_score + 4 - old_score if self.score_type == ScoreType.GAM else new_score - old_score

    def _improvement(self, new_score, old_score):
        return new_score - old_score

    def _significant(self, gain):
        return not is_insignificant(-gain) if self.score_type == ScoreType.GAM else not is_insignificant(
            -gain)  # convention neg. gains

    def _check_X(self, X):
        if not (self.data_type.is_multicontext() or self.data_type.is_time()):
            assert len(X.shape) == 2
            X = {0: X}  # ensure same format used for all data types
        else:
            assert all([len(X[k].shape) == 2 for k in range(len(X))])
        self.X = X
        self.D, self.N = [X[k].shape[0] for k in X][0], [X[k].shape[1] for k in X][0]

    def has_cycle(self, source, node):
        G_hat = self.topic_graph.copy()
        G_hat.add_edge(source, node)
        try:
            _ = nx.find_cycle(G_hat, orientation="original")
        except nx.exception.NetworkXNoCycle:
            return False
        return True

    def get_next_node(self, candidates):
        if self.known_true_order:
            n = len(self.topological_order)
            self._info(f"\tTrue Next Node: {self.true_top_order[n]}", -2)
            return self.true_top_order[n]

        improvement = self.get_improvement_matrix(self.topic_graph, candidates)
        delta = improvement - improvement.T
        # find the node with the smallest possible delta
        np.fill_diagonal(delta, -np.inf)
        best_delta = np.max(delta, axis=1)
        worst = np.argmin(best_delta)

        self._info(f"\tNext Node: {candidates[worst]}, order {self.topological_order} ", -2)
        return candidates[worst]

    def get_improvement_matrix(self, graph, candidates):
        improvement_matrix = np.zeros((len(candidates), len(candidates)))
        for cause in candidates:
            for effect in candidates:
                if cause == effect:
                    continue
                parents = list(graph.predecessors(effect))
                old_score = self._score(
                    parents, effect, efficient=(self.score_type == ScoreType.GP))  # speed this step up if using GPs
                parents.append(cause)
                new_score = self._score(parents, effect, efficient=False)
                improvement_matrix[candidates.index(cause), candidates.index(effect)] = \
                    self._improvement(new_score, old_score)
        return improvement_matrix

    def refinement_phase(self, min_parent_set_size = 0):
        # smallest subset of parents with insignificant score gain
        for j in self.topic_graph.nodes:
            parents = list(self.topic_graph.predecessors(j))
            if len(parents) == 0:
                continue

            best_size = np.inf
            arg_max = None

            old_score = self._score(parents, j)
            old_parents = parents.copy()

            for k in range(min_parent_set_size, len(parents) + 1 - 1):
                parent_sets = itertools.combinations(parents, k)
                for parent_set in parent_sets:

                    new_score = self._score(parent_set, j)
                    gain = self._gain(new_score, old_score)

                    if is_insignificant(np.abs(gain)) and len(parent_set) < best_size:  # favor smaller parent sets
                        best_size = len(parent_set)
                        arg_max = parent_set

            if arg_max is None:
                continue
            self._info(f'\trefine {parents} to {arg_max} -> {j}', -2)
            for p in old_parents:
                if p not in arg_max:
                    self._remove_edge(p, j)


    def get_metrics(self, true_graph):
        return compare_adj(true_graph.adj, self.topic_graph.adj)
