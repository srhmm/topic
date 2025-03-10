from types import SimpleNamespace

from exp.util.ty import ExpType
from exp.methods.methods import ids_continuous_methods, ids_time_continuous_methods, ids_change_methods, ids_time_change_methods


def exp_defaults(exp, competitors, base=True):
    """ Exp Configuration in the paper figures"""
    defaults_continuous_paper = dict(
        exp=exp,
        methods=ids_continuous_methods(competitors),
        S=[1000, 750, 500, 250],
        N=[8, 5, 2, 3, 10, 15],
        C=[1],
        R=[1],
        P=[0.5, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1],
        IV=1,
        score=0,  # GAM
        funs=[1],  # continuous nonlinear
        noises=[0],  # gauss noise
        reps=50,
    )
    defaults_continuous = dict(
        exp=exp,
        methods=ids_continuous_methods(competitors),
        S=[1000],
        N=[8, 5, 2],
        P=[0.5, 0.2,  0.4, 0.6, 0.8, 1],
        C=[1],
        R=[1],
        IV=1,
        score=0,  # GAM
        funs=[1],  # continuous nonlinear
        noises=[0],  # gauss noise
        reps=50,
    )

    defaults_changes = dict(
        exp=exp,
        methods=ids_continuous_methods(competitors),
        S=[250, 1000, 750, 500],
        N=[5],
        C=[3, 1, 5, 10, 15],
        dag_edge_p=[0.5],
        R=[1],
        score=0,  # GAM
        funs=[1],  # nonlinear
        noises=[0],  # gauss noise
        reps=50,
    )
    if exp == ExpType.CONTINUOUS.value:
        defaults = defaults_continuous
        defaults["methods"] = ids_continuous_methods(competitors)
    elif exp == ExpType.TIME.value:
        defaults = defaults_continuous
        defaults["methods"] = ids_time_continuous_methods(competitors)
        defaults["S"] = [500]
        defaults["funs"] = [2]  # ts nonlinear
    elif exp == ExpType.CHANGES.value:
        defaults = defaults_changes
        defaults["methods"] = ids_change_methods(competitors)
    elif exp == ExpType.TIMECHANGES.value:
        defaults = defaults_changes
        defaults["methods"] = ids_time_change_methods(competitors)
        defaults["funs"] = [2]  # ts nonlinear
    elif exp in [ExpType.TUEBINGEN.value, ExpType.REGED.value, ExpType.SACHS.value]:
        defaults = defaults_continuous
        defaults["methods"] = ids_continuous_methods(competitors)
    else:
        raise ValueError(ExpType)

    # secondary hyperparams
    defaults["dagtype"] = ["erdos_cd"]
    defaults["true_tau_max"] = [2]
    defaults["assumed_tau_max"] = [2]
    defaults["IV"] = 1
    defaults["iv_type"] = 0
    return defaults
