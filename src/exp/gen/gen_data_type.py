import numpy as np
from numpy.random import SeedSequence

from .exp.experiment import Experiment
from .exp.util.ty import ExpType
from .exp.gen.gen_continuous import gen_continuous_data, gen_context_data
from .exp.gen.gen_time import gen_time_data
from .topic.scoring.fitting import DataType


def gen_data(experiment: Experiment, params, seed):
    data_type = DataType.CONTINUOUS if experiment.exp_type == ExpType.CONTINUOUS else \
        DataType.CONT_MCONTEXT if experiment.exp_type == ExpType.CHANGES else \
            DataType.TIMESERIES if experiment.exp_type == ExpType.TIME else None
    if data_type is None:
        raise ValueError("synthetic exp type expected")

    return gen_data_type(data_type, params, seed)

def gen_data_type(data_type, params, seedseq):

    random_state = np.random.default_rng(seedseq)
    # Experiments with a single dataset
    if data_type == DataType.CONTINUOUS:
        data, truths = gen_continuous_data(params, random_state)
    elif data_type == DataType.TIMESERIES:
        data_summary, truths, valid = gen_time_data(params, seedseq)
        assert valid
        data = data_summary.datasets
    elif data_type == DataType.CONT_MCONTEXT:
        data_summary, truths = gen_context_data(params, random_state)
        data = data_summary.datasets
    else:
        raise ValueError(f"{data_type}")

    return data, truths

