
from exp.util.ty import FunType, ExpType, NoiseType, InterventionType
from exp.experiment import Experiment
from exp.run import run
from exp.run_realworld import run_tuebingen, run_reged, run_sachs


if __name__ == "__main__":
    """
    >>> python run_exp.py -e 0 --quick
    """
    import sys
    import argparse
    import logging
    from pathlib import Path

    logging.basicConfig()
    log = logging.getLogger("TOP")
    log.setLevel("INFO")

    ap = argparse.ArgumentParser("TOP")

    def enum_help(enm) -> str:
        return ','.join([str(e.value) + '=' + str(e) for e in enm])

    # experiment
    ap.add_argument("-e", "--exp", default=0, help=f"{enum_help(ExpType)}", type=int)

    # run options
    ap.add_argument("--quick", action="store_true", help="run a shorter experiment for testing")
    ap.add_argument("--safe", action="store_true", help="catch exceptions and skip")
    ap.add_argument("--competitors", action="store_true", help="include competitors")
    ap.add_argument("--skip_sid", action="store_false", help="SID call")
    ap.add_argument("-sd", "--seed", default=42, type=int)
    ap.add_argument("-nj", "--n_jobs", default=1, type=int)
    ap.add_argument("-v", "--verbosity", default=1, type=int)

    # path
    ap.add_argument("-bd", "--base_dir", default="")
    ap.add_argument("-wd", "--out_dir", default="res/")
    ap.add_argument("-rd", "--read_dir", default="res/")

    argv = sys.argv[1:]
    nmsp = ap.parse_args(argv)

    experiment = Experiment(**nmsp.__dict__)

    # Logging
    Path(nmsp.out_dir).mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(f"{nmsp.out_dir}.log")
    fh.setLevel(logging.INFO)
    experiment.logger=log
    experiment.logger.addHandler(fh)
    import warnings

    warnings.filterwarnings("ignore")

    if experiment.exp_type.is_synthetic():
        run(experiment)
    else:
        experiment.reps = 1
        if experiment.exp_type == ExpType.REGED:
            run_reged(experiment)
        elif experiment.exp_type == ExpType.SACHS:
            run_sachs(experiment)
        elif experiment.exp_type == ExpType.TUEBINGEN:
            run_tuebingen(experiment)
        else:
            raise ValueError(ExpType)

