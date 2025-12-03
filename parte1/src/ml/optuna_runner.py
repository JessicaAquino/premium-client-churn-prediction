from src.ml.optimization_config import OptimizationConfig

import optuna
from optuna.study import Study

import logging

import json

logger = logging.getLogger(__name__)


class OptunaRunner:
    def __init__(self, cfg: OptimizationConfig):
        self.cfg = cfg
        self.db_path = cfg.output_path + "db/"
        self.params_path = cfg.output_path + "best_params/"

    def run_study(self, objective_fn):
        storage = f"sqlite:///{self.db_path}optimization_lgbm.db"
        study_name = f"study_lgbm_binary{self.cfg.name}"

        study = optuna.create_study(
            direction="maximize",
            study_name=study_name,
            storage=storage,
            load_if_exists=True
        )

        logger.info(f"Starting study {study_name}")
        study.optimize(objective_fn, n_trials=self.cfg.n_trials)
        return study

    def save_best_params(self, study: Study):
        best_params = study.best_trial.params
        filename = self.params_path + f"best_params_binary{self.cfg.name}.json"
        with open(filename, "w") as f:
            json.dump(best_params, f, indent=4)
        logger.info(f"Best params saved to {filename}")
