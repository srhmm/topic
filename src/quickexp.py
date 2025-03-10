
import statistics
from collections import defaultdict

import numpy as np

from exp.gen import gen_data_type
from exp.util.ty import NoiseType, FunType, DagType

from topic.scoring.fitting import ScoreType, DataType
from topic.topic import Topic


def repeat_experiment(run_def, reps):
    metrs = defaultdict()
    resu = defaultdict()
    for i in range(reps):
        met = run_def(i)
        for m in met:
            if m not in metrs:
                metrs[m] = []
            metrs[m].append(met[m])
    for m in metrs:
        resu[m] = statistics.mean(metrs[m]), statistics.median(metrs[m]), statistics.stdev(metrs[m])
    return resu

def one_repetition(params, hypparams, seed):
    np.random.seed(seed)
    data, truths = gen_data_type(hypparams["data_type"], params, seed)

    top = Topic(**hypparams)
    _, _ = top.fit(data)
    return top.get_metrics(truths.graph)


if __name__ == "__main__":

    params = {
        'N': 5, 'S': 1000, 'P': 0.3,
        'NS': NoiseType.GAUSS, 'F': FunType.TAYLOR.to_fun(), 'DG': DagType.ERDOS_CD,
        'C': 1, 'R': 1, 'I': 2}
    hypparams = dict(vb=0, data_type=DataType.CONTINUOUS)

    resu = repeat_experiment(lambda seed: one_repetition(params, hypparams, seed), 100)
    print(resu)
