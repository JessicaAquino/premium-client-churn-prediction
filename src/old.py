import logging

import config.conf as cf
import infra.loader_utils as lu
import core.col_selection as cs
import core.feature_engineering as fe
import core.preprocessing as pp
import core.target_generation as tg
import config.logger_config as lc
import ml.lgbm_optimization as lo
import ml.lgbm_train_test as tt
from config.context import Context

from ml.optimization_config import OptimizationConfig

import optuna
import json
import pandas as pd
import polars as pl
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

def target_generation():
    tg.target_generation(name="competencia_02")
    

def hyperparams_opt():
    # 0. Load data
    df = lu.load_data(f"{PATH_DATA}competencia_01.csv", "csv")

    # 1. Columns selection
    cols_lag_delta_max_min_regl, cols_ratios = cs.col_selection(df)

    # 2. Feature Engineering
    df = fe.feature_engineering_pipeline(df, {
        "lag": {
            "columns": cols_lag_delta_max_min_regl,
            "n": 2
        },
        "delta": {
            "columns": cols_lag_delta_max_min_regl,
            "n": 2
        },
        # "minmax": {
        #     "columns": cols_lag_delta_max_min_regl
        # },
        "ratio": {
            "pairs": cols_ratios
        },
        # "linreg": {
        #     "columns": cols_lag_delta_max_min_regl,
        #     "window": 3
        # }
    })

    # 3. Preprocessing
    X_train, y_train_binary, w_train, X_test, y_test_binary, y_test_class, w_test = pp.preprocessing_pipeline(
        df,
        BINARY_POSITIVES,
        MONTH_TRAIN,
        MONTH_VALIDATION
    )

    # 4. Hyperparameters optimization

    opt_cfg = OptimizationConfig(
        n_trials=LGBM_N_TRIALS,
        name=STUDY_NAME,

        gain_amount=GAIN_AMOUNT,
        cost_amount=COST_AMOUNT,

        n_folds=LGBM_N_FOLDS,
        n_boosts=LGBM_N_BOOSTS,
        seeds=SEEDS,
        output_path=PATH_LGBM_OPT
    )
 
    study = lo.run_lgbm_optimization(X_train, y_train_binary, w_train, opt_cfg)

    # 5. Entrenamiento lgbm con la mejor iteración y mejores hiperparámetros

    best_iter = study.best_trial.user_attrs["best_iter"]
    best_params = study.best_trial.params

    tt_cfg = tt.TrainTestConfig(
        gain_amount=GAIN_AMOUNT,
        cost_amount=COST_AMOUNT,

        name=STUDY_NAME,

        output_path=PATH_LGBM_MODEL,
        seeds=SEEDS

    )
    model_lgbm = tt.entrenamiento_lgbm(X_train , y_train_binary, w_train ,best_iter,best_params , tt_cfg)
    y_pred=tt.evaluacion_lgbm(X_test , y_test_binary ,model_lgbm)


    logger.info("Pipeline ENDED!")

def kaggle_prediction():
    STUDY_NAME = "_20251003"
    
    NEW_STUDY = "_20251011_10"
    TOP_N = 13500

    logger.info("STARTING this wonderful pipeline!")

    # 0. Load data
    df = lu.load_data(f"{PATH_DATA}competencia_01.csv", "csv")

    # 1. Columns selection
    cols_lag_delta_max_min_regl, cols_ratios = cs.col_selection(df)

    # 2. Feature Engineering
    df = fe.feature_engineering_pipeline(df, {
        "lag": {
            "columns": cols_lag_delta_max_min_regl,
            "n": 2
        },
        "delta": {
            "columns": cols_lag_delta_max_min_regl,
            "n": 2
        },
        # "minmax": {
        #     "columns": cols_lag_delta_max_min_regl
        # },
        "ratio": {
            "pairs": cols_ratios
        },
        # "linreg": {
        #     "columns": cols_lag_delta_max_min_regl,
        #     "window": 3
        # }
    })

    # 3. Preprocessing
    MONTH_TRAIN.append(MONTH_VALIDATION)

    X_train, y_train_binary, w_train, X_test, y_test_binary, y_test_class, w_test = pp.preprocessing_pipeline(
        df,
        BINARY_POSITIVES,
        MONTH_TRAIN,
        MONTH_TEST
    )

    # 4. Best hyperparams loading
    name_best_params_file = f"best_params_binary{STUDY_NAME}.json"
    storage_name = "sqlite:///" + PATH_LGBM_OPT_DB + "optimization_lgbm_best.db"
    study = optuna.load_study(study_name='study_lgbm_binary'+STUDY_NAME, storage=storage_name)
    
    # 5. Training with best attempt and hyperparams
    best_iter = study.best_trial.user_attrs["best_iter"]
    
    with open(PATH_LGBM_OPT_BEST_PARAMS + name_best_params_file, "r") as f:
        best_params = json.load(f)
    logger.info(f"Hyperparams OK?: {study.best_trial.params == best_params}")
    
    tt_cfg = tt.TrainTestConfig(
        gain_amount=GAIN_AMOUNT,
        cost_amount=COST_AMOUNT,

        name=STUDY_NAME,

        output_path=PATH_LGBM_MODEL,
        seeds=SEEDS
    )
    
    model_lgbm = tt.entrenamiento_lgbm(X_train, y_train_binary, w_train ,best_iter,best_params , tt_cfg)

    # 6. Prediction!

    # y_test_binary=X_test[["numero_de_cliente"]]
    # y_pred=model_lgbm.predict(X_test)
    # y_test_binary["Predicted"] = y_pred
    # y_test_binary["Predicted"]=y_test_binary["Predicted"].apply(lambda x : 1 if x >=0.025 else 0)
    # logger.info(f"cantidad de bajas predichas : {(y_test_binary==1).sum()}")
    # y_test_binary=y_test_binary.set_index("numero_de_cliente")
    # y_test_binary.to_csv(f"output/prediction/prediccion{NEW_STUDY}.csv")

    # Number of clients you want to mark as positive

    # Keep predictions
    y_test_binary = X_test[["numero_de_cliente"]].copy()
    y_test_binary["PredictedProb"] = model_lgbm.predict(X_test)

    # Sort descending by predicted probability
    y_test_binary = y_test_binary.sort_values("PredictedProb", ascending=False)

    # Assign 1 to top N, 0 to the rest
    y_test_binary["Predicted"] = 0
    y_test_binary.iloc[:TOP_N, y_test_binary.columns.get_loc("Predicted")] = 1

    # Logging & save
    logger.info(f"cantidad de bajas predichas (top {TOP_N}): {y_test_binary['Predicted'].sum()}")

    y_test_binary_copy = y_test_binary.set_index("numero_de_cliente")
    y_test_binary_copy.to_csv(f"output/prediction/prediccion{STUDY_NAME}_prob.csv")

    y_test_binary = y_test_binary.set_index("numero_de_cliente")[["Predicted"]]
    y_test_binary.to_csv(f"output/prediction/prediccion{NEW_STUDY}.csv")


    logger.info("Pipeline ENDED!")

def evaluate_threshold():

    logger.info("STARTING this wonderful pipeline!")

    # 0. Load data
    df = lu.load_data(f"{PATH_DATA}competencia_01.csv", "csv")

    # 1. Columns selection
    cols_lag_delta_max_min_regl, cols_ratios = cs.col_selection(df)

    # 2. Feature Engineering
    df = fe.feature_engineering_pipeline(df, {
        "lag": {
            "columns": cols_lag_delta_max_min_regl,
            "n": 2
        },
        "delta": {
            "columns": cols_lag_delta_max_min_regl,
            "n": 2
        },
        # "minmax": {
        #     "columns": cols_lag_delta_max_min_regl
        # },
        "ratio": {
            "pairs": cols_ratios
        },
        # "linreg": {
        #     "columns": cols_lag_delta_max_min_regl,
        #     "window": 3
        # }
    })

    # 3. Preprocessing
    X_train, y_train_binary, w_train, X_test, y_test_binary, y_test_class, w_test = pp.preprocessing_pipeline(
        df,
        BINARY_POSITIVES,
        MONTH_TRAIN,
        MONTH_VALIDATION
    )

    # 4. Best hyperparams loading
    name_best_params_file = f"best_params_binary{STUDY_NAME}.json"
    storage_name = "sqlite:///" + PATH_LGBM_OPT_DB + "optimization_lgbm_vm.db"
    study = optuna.load_study(study_name='study_lgbm_binary'+STUDY_NAME, storage=storage_name)
    
    # 5. Training with best attempt and hyperparams
    best_iter = study.best_trial.user_attrs["best_iter"]
    
    with open(PATH_LGBM_OPT_BEST_PARAMS + name_best_params_file, "r") as f:
        best_params = json.load(f)
    logger.info(f"Hyperparams OK?: {study.best_trial.params == best_params}")
    
    tt_cfg = tt.TrainTestConfig(
        gain_amount=GAIN_AMOUNT,
        cost_amount=COST_AMOUNT,

        name=STUDY_NAME,

        output_path=PATH_LGBM_MODEL,
        seeds=SEEDS
    )
    
    model_lgbm = tt.entrenamiento_lgbm(X_train, y_train_binary, w_train ,best_iter,best_params , tt_cfg)

    # 6. Prediction

    y_pred = model_lgbm.predict(X_test)

    # 7. Ganancia

    ganancia = np.where(y_test_binary == 1, 780000, 0) - np.where(y_test_binary == 0, 20000, 0)

    idx = np.argsort(y_pred)[::-1]

    ganancia = ganancia[idx]
    y_pred = y_pred[idx]

    ganancia_cum = np.cumsum(ganancia)

    piso_envios = 4000
    techo_envios = 20000

    plt.figure(figsize=(10, 6))
    plt.plot(y_pred[piso_envios:techo_envios], ganancia_cum[piso_envios:techo_envios], label='Ganancia LGBM')
    plt.title('Curva de Ganancia')
    plt.xlabel('Predicción de probabilidad')
    plt.ylabel('Ganancia')
    plt.axvline(x=0.025, color='g', linestyle='--', label='Punto de corte a 0.025')
    plt.legend()

    plt.savefig(f"output/graphics/ganancia_prob{STUDY_NAME}.png", bbox_inches='tight')

    # Buscando mejor ganancia

    piso_envios = 4000
    techo_envios = 20000

    ganancia_max = ganancia_cum.max()
    gan_max_idx = np.where(ganancia_cum == ganancia_max)[0][0]

    plt.figure(figsize=(10, 6))
    plt.plot(range(piso_envios, len(ganancia_cum[piso_envios:techo_envios]) + piso_envios), ganancia_cum[piso_envios:techo_envios], label='Ganancia LGBM')
    plt.axvline(x=gan_max_idx, color='g', linestyle='--', label=f'Punto de corte a la ganancia máxima {gan_max_idx}')
    plt.axhline(y=ganancia_max, color='r', linestyle='--', label=f'Ganancia máxima {ganancia_max}')
    plt.title('Curva de Ganancia')
    plt.xlabel('Clientes')
    plt.ylabel('Ganancia')
    plt.legend()
    plt.savefig(f"{PATH_GRAPHICS}ganancia_prob_client{STUDY_NAME}.png", bbox_inches='tight')

    logger.info("Pipeline ENDED!")

def get_top_n_predictions(csv_path: str, n: int) -> pl.DataFrame:
    """
    Reads a CSV with columns ['numero_de_cliente', 'PredictedProb', 'Predicted']
    and returns the top N rows based on PredictedProb.
    The output contains only ['numero_de_cliente', 'Predicted'].
    """

    # 1. Read CSV
    df = pl.read_csv(csv_path)

    # 2. Sort descending by PredictedProb
    df = df.sort("PredictedProb", descending=True)

    # 3. Assign 1 to top N, 0 to rest
    df = df.with_columns(
        pl.when(pl.arange(0, df.height) < n)
        .then(1)
        .otherwise(0)
        .alias("Predicted")
    )

    # 4. Keep only the two columns
    return df.select(["numero_de_cliente", "Predicted"])

def compare():
    pred1 = pd.read_csv("output/prediction/prediccion_patito.csv", sep=',')
    pred2 = pd.read_csv("output/prediction/prediccion_20251003.csv", sep=',')

    merged = pred1.merge(pred2, on="numero_de_cliente", suffixes=("", "_patito"))
    diffs = merged[merged["Predicted"] != merged["Predicted_patito"]]

    print(f"Differences found: {len(diffs)}")
    if len(diffs) > 0:
        diffs.to_csv("output/diffs.csv", index=False)

if __name__ == "__main__":
    
    logger.info("STARTING this wonderful pipeline!")

    cfg = cf.load_config("CHALLENGE_01")

    ctx = Context(cfg)

    lu.ensure_dirs(
        ctx.path_logs,
        # PATH_DATA,
        ctx.path_lgbm_opt,
        ctx.path_lgbm_opt_best_params,
        ctx.path_lgbm_opt_db,
        ctx.path_lgbm_model,
        ctx.path_prediction,
        ctx.path_graphics,
    )
    lc.setup_logging(ctx.path_logs)

    # hyperparams_opt()
    # kaggle_prediction()
    # compare()
    # evaluate_threshold()
    # top_clients = get_top_n_predictions("output/prediction/prediccion_20251012_01_prob.csv", n=11000)
    # top_clients.write_csv("output/prediction/prediccion_20251012_28.csv")

    target_generation()
