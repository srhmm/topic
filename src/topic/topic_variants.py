import itertools
from typing import List

import networkx as nx
import numpy as np
from pygam import GAM

from topic.memoizededgescore import MemoizedEdgeScore
from topic.topic import Topic
from topic.scoring.fitting import ScoreType, DataType
from topic.util.util import is_insignificant


class TopicChanges(Topic):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data_type = DataType.CONT_MCONTEXT



class TopicTimed(Topic):
    regimes: List

    def __init__(self, max_lag: int, **kwargs):
        self.max_lag = max_lag
        self.defaultargs = {
            "data_type": DataType.TIMESERIES,
            "score_type": ScoreType.GAM,
            "regimes": None,
        }
        self.__dict__.update((k, v) for k, v in self.defaultargs.items() if k not in kwargs.keys())
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in self.defaultargs.keys())

        super().__init__(**kwargs)
        assert self.data_type.is_time()

        self.visited = []
        self.untimed_graph = nx.DiGraph()

    def _add_edge(self, node, others, gain, new_score):
        self.topic_graph.add_edge(node, (others[0], 0))  # convention:target alw has time lag 0
        ##self.scores.add_edge(node, others[0], gain, new_score)
        self.untimed_graph.add_edge(node[0], others[0])

        self._info(f"\tAdding {self.is_true_edge(node)(others[0])} edge {node} -> {others[0]} ", -1)

    def _remove_edge(self, best_parent, node):
        pa, lag = best_parent
        tg, _ = node
        if not self.topic_graph.has_edge(best_parent, node):
            print("something went wrong")
            return
        self.topic_graph.remove_edge(best_parent, (tg, 0))  # convention:target alw has time lag 0
        ##self.scores.remove_edge(best_parent, tg)

        other_lags = sum([1 if self.topic_graph.has_edge((pa, lg), node) else 0 for lg in range(self.max_lag)])
        if other_lags <= 1:
            if self.topic_graph.has_edge(pa, tg):
                self.untimed_graph.remove_edge(pa, tg)

        self._info(f"\tRemoving {self.is_true_edge(best_parent)(tg)} edge {best_parent} -> {tg}", -1)

    def _score(self, parents, others, efficient=False):
        return self.scores_efficient.score_edge(others[0], parents) if efficient \
            else self.scores.score_edge(others[0], parents)

    def fit(self, X):
        self._info(f"***TOPIC {self.data_type.value, self.score_type.value}***")
        assert all([len(X[k].shape) == 2 for k in range(len(X))])
        self.X = X
        self.D, self.N = [X[k].shape[0] for k in X][0], [X[k].shape[1] for k in X][0]
        self.regimes = [[3, self.D]] if self.regimes is None else self.regimes
        self.candidates = [(i, l) for i in range(self.N) for l in range(self.max_lag)]
        scoring_params = dict(
            data_type=self.data_type,
            score_type=self.score_type,
            vb=self.vb-1, lg=self.lg,
            regimes=self.regimes
        )

        self.scores_efficient = MemoizedEdgeScore(self.X, **scoring_params)
        self.scores_efficient.score_type = ScoreType.GAM
        self.scores = MemoizedEdgeScore(self.X, **scoring_params)

        self.untimed_graph.add_nodes_from(range(self.N))
        self.topic_graph.add_nodes_from(self.candidates)

        X_pred = self.X.copy()

        for _ in range(self.N):
            node = self.get_next_node(self.candidates, self.X )
            nd = node[0]
            if nd in self.visited:
                continue

            self.topological_order.append(nd)
            lags = range(self.max_lag) if self.data_type.is_time() else range(1)

            for lag in lags:
                node = (nd, lag)
                for others in self.candidates:
                    # if (self.data_type.is_time() and node[0] == others[0]) or (not self.data_type.is_time() and node == others):
                    if nd == others[0]:  # or
                        continue

                    parents = list(self.topic_graph.predecessors((others[0], 0)))  # convention target time lag 0

                    old_score = self._score(parents, others)
                    old_parents = parents.copy()
                    parents.append(node)
                    new_score = self._score(parents, others)
                    gain = self._gain(new_score, old_score)

                    if self._significant(gain):
                        self._add_edge(node, others, gain, new_score)

                self.candidates.remove(node)

                # prune unnecessary incoming edges
                its_parents = list(self.topic_graph.predecessors((nd, 0)))

                if len(its_parents) == 0:
                    continue

                while True:
                    best_diff = np.inf
                    best_parent = None
                    old_score = self._score(its_parents, node)

                    new_model = None
                    for parent in its_parents:

                        # make new model without parent
                        parents = its_parents.copy()
                        parents.remove(parent)
                        if len(parents) == 0:
                            continue
                        new_score = self._score(parents, node)
                        diff = new_score - old_score

                        if diff < best_diff \
                                and self._significant(diff):
                            best_diff = diff
                            best_parent = parent

                    if best_parent is not None:
                        self._remove_edge(best_parent, node)
                        its_parents.remove(best_parent)
                    else:
                        if new_model is not None:
                            X_pred[:, node] = new_model.predict(self.X[:, parents])
                        break

            self.visited.append(node[0])

        self._info(self.topological_order, -1)

        if self.extra_refinement:
            pass
            # self.refinement_phase()

        return self.untimed_graph, self.topic_graph, self.topological_order

    def get_next_node(self, candidates, X ):

        if self.known_true_order:
            n = len(self.topological_order)
            self._info(f"\tTrue Next Node: {self.true_top_order[n]}", 2)
            return self.true_top_order[n]

        graph = self.topic_graph
        improvement = self.get_improvement_matrix(graph, candidates)

        delta = improvement - improvement.T
        # find the node with the smallest possible delta

        # set diagoonal to -inf
        np.fill_diagonal(delta, -np.inf)

        best_delta = np.max(delta, axis=1)
        worst = np.argmin(best_delta)

        if self.data_type.is_time():
            pred = [p for p in range(self.N) if
                    self.is_true_edge((p, 0))(candidates[worst][0]) in ['[caus]', '[caus-any-lg]']]
        else:
            pred = [p for p in range(self.N) if self.is_true_edge(p)(candidates[worst]) in ['causal']]
        self._info(f"\tNext Node: {candidates[worst]},"
                   f" predec {pred}, order {self.topological_order} ", -2)
        return candidates[worst]

    def refinement_phase(self):
        raise NotImplementedError #todo fix
        # currently: smallest subset of parents with insignificant score difference
        self._info('\tRefinement', -1)

        for j in self.topic_graph.nodes:
            parents = list(self.topic_graph.predecessors(j))
            if len(parents) == 0:
                continue

            best_size = np.inf
            best_gain = np.inf
            arg_max, arg_newscore, arg_gain = None, None, None

            old_score = self._score(parents, j)
            old_parents = parents.copy()

            timed_parents = list([(pa, i) for pa in parents for i in range(self.max_lag)])

            min_size = 0
            for k in range(min_size, len(timed_parents) + 1 - 1):
                parent_sets = itertools.combinations(timed_parents, k)
                for parent_set in parent_sets:
                    # check whether parent set induces cyvles
                    new_score = self._score(parent_set, j)
                    gain = self._gain(new_score, old_score)
                    # print(gain)
                    # if is_insignificant(np.abs(gain)) and len(parent_set)<best_size: #favor smaller parent sets
                    if is_insignificant(np.abs(gain)) and gain < best_gain:
                        best_size, best_gain = len(parent_set), gain
                        arg_max = parent_set
                        arg_newscore, arg_gain = new_score, gain
                    # print(f'\tconsidering {parent_set} -> {j}, {np.round(gain[0][0],2)}')
            if (arg_max is not None):
                self._info(f'\trefine {parents} to {arg_max} -> {j}', -1)
                for p in old_parents:
                    self._remove_edge(p, j)
                for p in arg_max:
                    self._add_edge(p, j, arg_gain, arg_newscore)

class TopicTimedChanges(TopicTimed):
    """ wrapper for multi-context time TOPIC, not tested much"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data_type = DataType.TIME_MCONTEXT
