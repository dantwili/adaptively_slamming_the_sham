import os
from abc import ABC, abstractmethod
import numpy as np
from cmdstanpy import CmdStanModel, CmdStanMCMC

from .results import ExperimentResult

MODEL_DIR = os.path.abspath(os.path.join(os.getcwd(), "..", "models"))
MODEL_FLAT_HYPERPRIORS = CmdStanModel(
    stan_file=os.path.join(MODEL_DIR, "flat_hyperpriors.stan")
)
MODEL_ORACLE_HYPERPARAMETERS = CmdStanModel(
    stan_file=os.path.join(MODEL_DIR, "oracle_hyperparameters.stan")
)
MODEL_CONST_B_FLAT_HYPERS_THETA = CmdStanModel(
    stan_file=os.path.join(MODEL_DIR, "const_b_flat_hypers_theta.stan")
)


class Fitter(ABC):
    def __init__(self, model: CmdStanModel):
        self.model = model
        self.name = model._name

    def fit(self, expt: ExperimentResult, **kwargs) -> CmdStanMCMC:
        return self.model.sample(data=self.to_dict(expt), show_progress=False)

    @abstractmethod
    def to_dict(self, expt: ExperimentResult) -> dict: ...


class FitterFlatHyperpriors(Fitter):
    def __init__(self, model: CmdStanModel = MODEL_FLAT_HYPERPRIORS):
        super().__init__(model)

    def to_dict(self, expt: ExperimentResult) -> dict:
        return {
            "J": len(expt.data),
            "j": list(range(1, len(expt.data) + 1)),
            "y1_bar": list(expt.data["y1_bar"]),
            "y0_bar": list(expt.data["y0_bar"]),
            "sigma_y1_bar": list(expt.data["sigma_y1_bar"]),
            "sigma_y0_bar": list(expt.data["sigma_y0_bar"]),
            "n": list(expt.data["n"]),
            "p": list(expt.data["p"]),
        }


class FitterOracleHyperparameters(Fitter):
    def __init__(self, model: CmdStanModel = MODEL_ORACLE_HYPERPARAMETERS):
        super().__init__(model)

    def to_dict(self, expt: ExperimentResult) -> dict:
        return {
            "J": len(expt.data),
            "j": list(range(1, len(expt.data) + 1)),
            "y1_bar": list(expt.data["y1_bar"]),
            "y0_bar": list(expt.data["y0_bar"]),
            "sigma_y1_bar": list(expt.data["sigma_y1_bar"]),
            "sigma_y0_bar": list(expt.data["sigma_y0_bar"]),
            "n": list(expt.data["n"]),
            "p": list(expt.data["p"]),
            "mu_b": expt.params["mu_b"],
            "mu_theta": expt.params["mu_theta"],
            "sigma_b": expt.params["sigma_b"],
            "sigma_theta": expt.params["sigma_theta"],
        }


class FitterEstimatedHyperparameters(Fitter):
    def __init__(self, model: CmdStanModel = MODEL_ORACLE_HYPERPARAMETERS):
        super().__init__(model)

    def to_dict(self, expt: ExperimentResult) -> dict:
        # given the experimental data, we need to estimate
        # the hyperparameters and use these estimates to form our new prior

        fitter = FitterFlatHyperpriors()
        sampler = fitter.fit(expt)

        mu_b_hat = np.mean(sampler.mu_b)
        sigma_b_hat = np.mean(sampler.sigma_b)
        mu_theta_hat = np.mean(sampler.mu_theta)
        sigma_theta_hat = np.mean(sampler.sigma_theta)

        return {
            "J": len(expt.data),
            "j": list(range(1, len(expt.data) + 1)),
            "y1_bar": list(expt.data["y1_bar"]),
            "y0_bar": list(expt.data["y0_bar"]),
            "sigma_y1_bar": list(expt.data["sigma_y1_bar"]),
            "sigma_y0_bar": list(expt.data["sigma_y0_bar"]),
            "n": list(expt.data["n"]),
            "p": list(expt.data["p"]),
            "mu_b": mu_b_hat,
            "mu_theta": mu_theta_hat,
            "sigma_b": sigma_b_hat,
            "sigma_theta": sigma_theta_hat,
        }


class FitterConstbFlatHypersTheta(FitterFlatHyperpriors):
    def __init__(self, model: CmdStanModel = MODEL_CONST_B_FLAT_HYPERS_THETA):
        super().__init__(model)
