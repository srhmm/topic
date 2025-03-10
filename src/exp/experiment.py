import logging
from typing import List

from exp.defaults import exp_defaults
from exp.methods.methods import id_to_method
from exp.util.ty import FunType, ExpType, NoiseType, InterventionType, DagType
from topic.scoring.fitting import ScoreType, GPType
from topic.scoring.models.mdl_fourierf import FourierType


class Experiment:
    logger: logging
    seed: int
    rep: int
    reps: int = 50
    n_jobs: int = 1
    quick: bool
    fixed: dict = {}
    functions: List = []
    exps: List = []

    parameters_to_change = {
        "S",  "N",  "C",  "P"
    }
    def __init__(self, **kwargs):
        self.__dict__.update((k, v) for k, v in kwargs.items())

        # get parameters for specific experiment type
        self.exp_type = ExpType(self.exp)
        defaults = exp_defaults(self.exp, competitors=self.competitors)
        self.__dict__.update((k, v) for k, v in defaults.items())

        self.noises = [ns for ns in map(NoiseType, self.noises)]
        self.dagtype = [dg for dg in map(DagType, self.dagtype)]
        self.funs = [fun.to_fun() for fun in map(FunType, self.funs)]
        self.intervention_type=InterventionType(defaults["iv_type"])

        # method hyperparams
        methods = kwargs.get("methods", defaults["methods"])
        self.methods = [id_to_method(m) for m in methods]  # map MethodType
        self.score_type=ScoreType(self.score)
        self.gp_type=GPType.EXACT
        self.fourier_type=FourierType.QUADRATURE
        self.n_components=50

        self.attrs = self.get_attributes()
        self.exps = self.get_experiment_cases()

        self.reps = min(self.reps, 3) if self.quick else max(self.reps, 3) # for computing avgs etc

    def get_attributes(self):
        params = {nm: self.__dict__[nm] for nm in self.parameters_to_change}
        return params

    def get_cases(self):
        self.fixed = {attr: val[0] for (attr, val) in self.attrs.items()}
        # Keep one attribute fixed and get all combos of the others
        combos = [
            ({nm: (self.attrs[nm][i] if nm == fixed_nm else self.fixed[nm]) for nm in self.attrs})
            for fixed_nm in self.fixed
            for i in range(len(self.attrs[fixed_nm]))
        ]
        test_cases = {"_".join(f"{arg}_{val}" for arg, val in combo.items()): combo for combo in combos}
        # small runs first
        test_cases = dict(sorted(test_cases.items(), key=lambda dic: (dic[1]["N"], dic[1]["S"])))
        return test_cases

    def get_experiment_cases(self):
        attrs = {param: self.__dict__[nm]  for nm, param in  {('noises', 'NS'), ('funs', 'F'), ('dagtype', 'DG')}}
        combos = [{'F': fun, 'NS': ns, 'DG': dg} for fun in attrs['F'] for ns in attrs['NS'] for dg in attrs['DG']]
        return combos