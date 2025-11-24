import logging
import json
import optuna
import numpy as np
import matplotlib.pyplot as plt

from ml.optimization_config import OptimizationConfig
from ml import optimization as lo
from ml import lgbm_train_test as tt
from core import col_selection as cs
from core import feature_engineering as fe
from core import preprocessing as pp
from core import target_generation as tg
from infra import loader_utils as lu
from pipelines.base_pipeline import BasePipeline

logger = logging.getLogger(__name__)


class ChallengeOnePipeline(BasePipeline):
    """
    Implementation of the BasePipeline for the CHALLENGE_01 experiment.
    """

    # region --- Required abstract methods -------------------------------------

    def load_data(self):
        return lu.load_data(f"{self.ctx.path_data}{self.ctx.input_dataset}", "csv")

    def column_selection(self, df):
        self.cols_lag_delta, self.cols_ratios = cs.col_selection(df)

    def feature_engineering(self, df):
        df = fe.feature_engineering_pipeline(df, {
            "lag": {"columns": self.cols_lag_delta, "n": 2},
            "delta": {"columns": self.cols_lag_delta, "n": 2},
            "ratio": {"pairs": self.cols_ratios}
        })
        return df

    def preprocessing(self, df):
        return pp.preprocessing_pipeline(
            df,
            self.ctx.binary_positives,
            self.ctx.month_train,
            self.ctx.month_validation
        )

    def train(self, X_train, y_train, w_train):
        logger.info("Starting hyperparameter optimization...")

        opt_cfg = OptimizationConfig.from_context(self.ctx)

        # Run optimization
        study = lo.run_lgbm_optimization(X_train, y_train, w_train, opt_cfg)

        best_iter = study.best_trial.user_attrs["best_iter"]
        best_params = study.best_trial.params

        logger.info(f"Best iteration: {best_iter}")
        logger.info(f"Best params: {best_params}")

        # Save model
        tt_cfg = tt.TrainTestConfig(
            gain_amount=self.ctx.gain_amount,
            cost_amount=self.ctx.cost_amount,
            name=self.ctx.study_name,
            output_path=self.ctx.path_lgbm_model,
            seeds=self.ctx.seeds
        )

        self.model = tt.entrenamiento_lgbm(
            X_train, y_train, None, best_iter, best_params, tt_cfg
        )

        self.best_iter = best_iter
        self.best_params = best_params

    def evaluate(self, X_test, y_test):
        logger.info("Evaluating model performance...")
        y_pred = self.model.predict(X_test)

        # Simple gain curve visualization
        gain = np.where(y_test == 1, 780000, -20000)
        sorted_idx = np.argsort(y_pred)[::-1]
        gain_cum = np.cumsum(gain[sorted_idx])

        plt.figure(figsize=(10, 6))
        plt.plot(range(len(gain_cum)), gain_cum)
        plt.title("Cumulative Gain Curve")
        plt.xlabel("Clients")
        plt.ylabel("Gain")
        plt.savefig(f"{self.ctx.path_graphics}gain_curve_{self.ctx.study_name}.png")

        logger.info("Evaluation complete.")
        logger.info(f"Max gain: {gain_cum.max()}")

        return {"max_gain": gain_cum.max()}
    
    
    def run(self, until: str = "evaluate"):
        df = self.load_data()
        self.column_selection(df)
        df = self.feature_engineering(df)
        X_train, y_train_binary, w_train, X_test, y_test_binary, y_test_class, w_test = self.preprocessing(df)

        if until == "preprocessing":
            return X_train, y_train_binary, w_train, X_test, y_test_binary, y_test_class, w_test

        self.train(X_train, y_train_binary, w_train)
        if until == "train":
            return
        
        return self.evaluate(X_test, y_test_binary)

    # endregion ---------------------------------------------------------------

    # region --- Optional helper methods --------------------------------------

    def kaggle_prediction(self, top_n: int = 13500):
        """
        Generates Kaggle-ready prediction file using best study parameters.
        """
        logger.info("Generating Kaggle prediction...")

        # Reload data and re-train with best params
        df = self.load_data()
        self.column_selection(df)
        df = self.feature_engineering(df)
        X_train, y_train, _, X_test = self.preprocessing(df)

        name_best_params_file = f"best_params_binary{self.ctx.study_name}.json"
        storage_name = "sqlite:///" + self.ctx.path_lgbm_opt_db + "optimization_lgbm_best.db"

        study = optuna.load_study(
            study_name=f"study_lgbm_binary{self.ctx.study_name}", storage=storage_name
        )

        best_iter = study.best_trial.user_attrs["best_iter"]

        with open(self.ctx.path_lgbm_opt_best_params + name_best_params_file, "r") as f:
            best_params = json.load(f)

        tt_cfg = tt.TrainTestConfig(
            gain_amount=self.ctx.gain_amount,
            cost_amount=self.ctx.cost_amount,
            name=self.ctx.study_name,
            output_path=self.ctx.path_lgbm_model,
            seeds=self.ctx.seeds
        )

        model_lgbm = tt.entrenamiento_lgbm(
            X_train, y_train, None, best_iter, best_params, tt_cfg
        )

        # Predictions
        df_pred = X_test[["numero_de_cliente"]].copy()
        df_pred["PredictedProb"] = model_lgbm.predict(X_test)
        df_pred = df_pred.sort_values("PredictedProb", ascending=False)

        df_pred["Predicted"] = 0
        df_pred.iloc[:top_n, df_pred.columns.get_loc("Predicted")] = 1

        output = f"{self.ctx.path_prediction}prediccion{self.ctx.study_name}.csv"
        df_pred.set_index("numero_de_cliente")[["Predicted"]].to_csv(output)
        logger.info(f"Kaggle prediction saved: {output}")




    # endregion ---------------------------------------------------------------
