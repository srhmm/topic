import os
import statistics
from collections import defaultdict
from types import SimpleNamespace

import networkx as nx
import pandas as pd
from joblib import Parallel, delayed
from numpy.random import SeedSequence

import numpy as np

from exp.experiment import Experiment
from exp.gen.gen_data_type import gen_data
from exp.util.eval import convert_nx_graph_to_adj


##from exp.util.case_results import CaseReslts, write_cases



def run(experiment: Experiment):
    """ Main Method for running our experiments """
    from datetime import datetime
    experiment.logger.info(f"RUN AT: {str(datetime.now())}")

    for exp in experiment.get_experiment_cases():
        exp_cases = experiment.get_cases()
        experiment.logger.info(f"Experiment {exp}")

        for case in exp_cases.items():
            run_case_safe(experiment, exp, case) if experiment.safe else run_case(experiment, exp, case)

        for attr in experiment.fixed:
            write_cases(experiment, None, attr, base_attributes=experiment.fixed)

def run_case_safe(options, exp, case):
    try:
        run_case(options, exp, case)
    except Exception as e:
        print(f"case failed {e}, skipping it")
        options.inc_seed()


def run_case(options, exp, test_case):
    """ Runs a case with fixed parameters (n nodes, n timepoints ...) """
    ss = SeedSequence(options.seed)
    cs = ss.spawn(options.reps)

    case, params = test_case
    for e,val in exp.items():
        params[e] = val
    run_rep = lambda rep: run_repetition(options, params, case, rep)

    if options.n_jobs != 1:
        reslts = Parallel(n_jobs=options.n_jobs)(delayed(
            run_rep)(rep_seed) for rep_seed in enumerate(cs))
    else:
        original_out_dir = options.out_dir
        options.out_dir = original_out_dir + 'intermediate/'
        reslts = []
        for rep_seed in enumerate(cs):
            res = run_rep(rep_seed)
            reslts.append(res)
        options.out_dir = original_out_dir

    case_results = CaseReslts(case)
    case_results.add_reps(reslts)
    case_results.write_case(params, options)


def run_repetition(experiment, params, case, rep_seed):
    import warnings
    warnings.filterwarnings("ignore")

    rep_random_state = np.random.default_rng(rep_seed[1])
    experiment.logger.info(f'*** Rep {rep_seed[0] + 1}/{experiment.reps}, seed: {rep_seed[0]}***')
    experiment.logger.info(f'Params: {case}')

    # Generate Data
    data, truths = gen_data(experiment, params, rep_seed[1])

    # Run methods
    #return {method.value: run_method(experiment, params, data, truths, method) for method in experiment.methods}


    # Run methods
    metrics = defaultdict(SimpleNamespace)

    for method in experiment.methods:
        metrics[method.nm()] = run_method(experiment, params, case, rep_seed, method, data, truths)

    return metrics


def run_method(experiment, params, case, rep_seed, method, data, truths):
    experiment.logger.info(f'\tMethod: {method.nm()}')
    graph_file = os.path.join(
        experiment.read_dir + f"{experiment.exp_type}/tikzfiles/graphs/{case}_m_{method.nm()}_graph_{rep_seed[0]}.tsv")
    true_dag_file = os.path.join(
        experiment.read_dir + f"{experiment.exp_type}/tikzfiles/graphs/{case}_m_{method.nm()}_trueDAG_{rep_seed[0]}.tsv")
    true_tdag_file = os.path.join(
        experiment.read_dir + f"{experiment.exp_type}/tikzfiles/graphs/{case}_m_{method.nm()}_trueWCG_{rep_seed[0]}.tsv")

    # Read precomputed results
    if os.path.exists(graph_file):
        experiment.logger.info(f'\tReading results from: {graph_file}')
        res_dag = pd.read_csv(graph_file, sep=",", header=None).to_numpy()
        true_dag = pd.read_csv(true_dag_file, sep=",", header=None).to_numpy()
        true_tdag = np.zeros((0, 0))
        if os.path.exists(true_tdag_file):
            true_tdag = pd.read_csv(true_tdag_file, sep=",", header=None).to_numpy()
        method.add_result_dag(method, res_dag)
        method.metrics = {'time': -1}  # means method was not fit
        method.eval_results(
            method, truths=
            SimpleNamespace(graph=nx.from_numpy_array(true_dag, create_using=nx.DiGraph), timed_dag=true_tdag),
            options=experiment)

    # or compute results
    else:
        method.fit(method, data, truths, params, experiment)
        method.eval_results(method, truths, experiment)
        res_dag = method.get_result_dag(method)
        assert res_dag is not None
        true_dag: np.array = convert_nx_graph_to_adj(truths.graph) if not truths.data_type.is_time() else truths.graph

    method_results = SimpleNamespace(
        obj=method,
        metrics=method.metrics,
        res_dag=res_dag,
        true_dag=true_dag,
        got_timed_result=False)
    if truths.data_type.is_time():
        method_results.true_tdag = truths.timed_dag
        method_results.got_timed_result = True
    return method_results


class CaseMethodReslts:
    def __init__(self, case, nm):
        self.case = case
        self.nm = nm
        self.metrics = defaultdict(list)
        self.objects = []
        self.dags = []
        self.true_dags = []
        self.true_tdags = []

    def add_method_rep(self, method_results):
        assert method_results.obj.nm() == self.nm
        for met in method_results.metrics:
            self.metrics[met] += [method_results.metrics[met]]
        self.objects.append(method_results.obj)
        self.dags.append(method_results.res_dag)
        self.true_dags.append(method_results.true_dag)
        if method_results.got_timed_result:
            self.true_tdags.append(method_results.true_tdag)


class CaseReslts:
    def __init__(self, case):
        self.case = case
        self.method_results = {}

    def add_reps(self, all_results):
        for rep in range(len(all_results)):
            for method_nm in all_results[rep]:
                one_result = all_results[rep][method_nm]
                self._add_rep(one_result)

    def _add_rep(self, one_result):
        nm = one_result.obj.nm()
        if nm not in self.method_results:
            self.method_results[nm] = CaseMethodReslts(self.case, nm)
        self.method_results[nm].add_method_rep(one_result)

    def write_case(self, params, options):
        table = self.method_results

        path = options.out_dir + str(options.exp_type) + "/tikzfiles/all/"
        graphpath = options.out_dir + str(options.exp_type) + "/tikzfiles/graphs/"
        os.makedirs(path, exist_ok=True)
        os.makedirs(graphpath, exist_ok=True)

        # print all results
        methods = table.keys()
        for mth in methods:
            fl = os.path.join(path, f"{self.case}_m_{mth}.tsv")
            write_file = open(fl, 'w')
            write_file.write(f'X')

            for met in table[mth].metrics.keys():
                write_file.write(f'\t{mth}_{met}')
            for r in range(options.reps):
                write_file.write(f'\n{r}')
                for met in table[mth].metrics.keys():
                    if len(table[mth].metrics[met]) >= r:
                        write_file.write(f'\t{table[mth].metrics[met][r]} ')
                    else:
                        write_file.write(f'\t{-1}')
            write_file.close()

            # Store result and true graphs
            for g_i, graph in enumerate(table[mth].dags):
                dag_file = os.path.join(graphpath, f"{self.case}_m_{mth}_graph_{g_i}.tsv")
                pd.DataFrame(graph).to_csv(dag_file, header=None, index=False, sep=",")

            for g_i, graph in enumerate(table[mth].true_dags):
                true_dag_file = os.path.join(graphpath, f"{self.case}_m_{mth}_trueDAG_{g_i}.tsv")
                pd.DataFrame(graph).to_csv(true_dag_file, header=None, index=False, sep=",")

            for g_i, graph in enumerate(table[mth].true_tdags):
                true_tdag_file = os.path.join(graphpath, f"{self.case}_m_{mth}_trueWCG_{g_i}.tsv")
                pd.DataFrame(graph).to_csv(true_tdag_file, header=None, index=False, sep=",")

        # Print averages of all runs for this case
        path = options.out_dir + str(options.exp_type) + "/tikzfiles/avg/"
        if not os.path.exists(path):
            os.makedirs(path)

        for mth in methods:
            fl = os.path.join(path, f"{self.case}_m_{mth}.tsv")
            write_file = open(fl, 'w')
            write_file.write(f'X')

            for met in table[mth].metrics.keys():
                write_file.write(f'\t{mth}_{met}\t{mth}_{met}_var\t{mth}_{met}_std')
            write_file.write(f'\n0')
            for met in table[mth].metrics.keys():
                if len(table[mth].metrics[met]) > 2:
                    write_file.write(
                        f'\t{statistics.mean(table[mth].metrics[met])}'
                        f'\t{statistics.variance(table[mth].metrics[met])}'
                        f'\t{statistics.stdev(table[mth].metrics[met])}')

                elif len(table[mth].metrics[met]) == 1:
                    write_file.write(f'\t{table[mth].metrics[met][0]}\t0\t0')
                else:
                    write_file.write(f'\t{-1}\t0\t0')
            write_file.close()

    def plot_case(self, params, options):
        metrics = ['f1', 'shd', 'tpr', 'fpr']
        if options.enable_SID_call:
            metrics = ['f1', 'shd', 'sid', 'fpr']
        if options.exp_type.is_time():
            metrics = ["f1", "f1-t", "shd", "shd-t"]
            if options.enable_SID_call:
                metrics = ["f1", "f1-t", "sid", "sid-t"]
        table = self.method_results
        if "shd" in metrics:
            shdmax = max([max(table[mth].metrics['shd']) for mth in table])
        methods = table.keys()

        fig, axs = plt.subplots(nrows=len(metrics), ncols=max(len(methods), 2), figsize=(
        2 * len(methods), 2 * len(metrics)))  # only one method-> show 2 cols to avoid index problems

        fig.suptitle(f'Case: {self.case}')
        fig.tight_layout()
        for i, m in enumerate(methods):
            axs[0][i].set_title(m)

        # metrics that should be shown in the rows
        for row, dag_metric in enumerate(metrics):
            scores = [table[m].metrics[dag_metric] if dag_metric in table[m].metrics else [0.0] for m in methods]
            n = options.reps

            successes = {
                m: f'[{len(table[m].metrics[dag_metric])}/{n}]' if dag_metric in table[m].metrics else f'[0/{n}]' for m
                in methods}
            labels = [successes[m] for m in methods]

            if len(methods) == 1:  # hack one method
                labels += [successes[m] for m in methods]

            for i, m in enumerate(methods):
                axs[row][i].boxplot(scores[i])

            # adding horizontal grid lines
            for i, ax in enumerate(axs[row]):
                ax.yaxis.grid(True)
                metric = dag_metric
                ax.set_xticks([0], labels=[labels[i]])
                ax.set_xlabel('')
                if metric == 'shd':
                    ax.set_ylim([0, shdmax])
                else:
                    ax.set_ylim([0, 1])

                ax.set_ylabel(metric)

        path = options.out_dir + str(options.exp_type) + "/plots/" + getfolder(params, options)
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(path + self.case + '.png')


def getfolder(params, options):
    if not options.exp_type.is_synthetic():
        return ''
    fixed = {"C": options.base_C, "N": options.base_N, "S": options.base_S, "R": options.base_R, "P": options.base_P,
             "TM": options.base_taumax}
    attrs = {"C": options.contexts_C, "N": options.nodes_N, "S": options.samples_S, "R": options.regimes_R,
             "P": options.edge_P, "TM": options.true_tau_max}
    s = 'base/'
    for attr in attrs:
        if params[attr] != fixed[attr]:
            s = f'change_{attr}/'
    return s


def create_plot_files(
        out_dir,
        fixed={"C": 10, "N": 5, "S": 500, "R": 1, "TM": 2},  # example usage
        varied={"C": [10], "N": [5], "S": [500], "R": [1], "TM": [2]},  # example usage
        sep="\t"
):
    """ Aggregate results: for each variable VAR in fixed, create tables vry_VAR_... forall possible values of VAR in varied, keep other variables fixed"""
    rdpath = out_dir + "/tikzfiles/avg/"
    if not os.path.exists(rdpath):
        return
    wrpath = out_dir + "/tikzfiles/vry/"
    if not os.path.exists(wrpath):
        os.makedirs(wrpath)

    for vr in fixed:
        relevant = {}
        assert vr in varied
        for rdfile in os.listdir(rdpath):
            if rdfile.endswith(".tsv"):
                case_info = rdfile.split(".tsv")[0]
                case = options_case_deconstruct(case_info)
                # Collect cases where the variable of interest, var, is in the varied range and all others are fixed
                if vr == "P":  # hacky
                    if float(case[vr]) not in varied[vr]:
                        continue
                else:
                    if int(case[vr]) not in varied[vr]:
                        continue
                if any([case[vvar] != str(fixed[vvar]) for vvar in fixed if vvar != vr]):
                    continue
                # read
                table = pd.DataFrame(pd.read_csv(os.path.join(rdpath, rdfile), sep='\t'))
                table[vr] = [case[vr]]

                # only combine files of the same category (e.g. do not mix linear and nonlinear case)
                categ = f"EXP_{case['EXP']}_F_{case['F']}_NS_{case['NS']}_M_{case['M']}"
                if not categ in relevant:
                    relevant[categ] = []
                relevant[categ].append(table)

            for categ in relevant:
                table = pd.concat(relevant[categ])
                if len(table) < 2:
                    continue
                if "X" in table:
                    table = table.drop("X", axis=1)
                table = table.sort_values(by=[vr])  # so that tikz plots it correctly #TODO double check
                table.to_csv(os.path.join(wrpath, f"vry_{vr}_{categ}.tsv"), sep=sep, index=False)
