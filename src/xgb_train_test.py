#lgbm_train_test.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit
import xgboost as xgb

import logging
from time import time
import datetime

import pickle
import json
import os
from typing import Any

from src.config import GANANCIA,ESTIMULO

ganancia_acierto = GANANCIA
costo_estimulo=ESTIMULO

logger = logging.getLogger(__name__)

def entrenamiento_xgb(X_train: pd.DataFrame,y_train_binaria: pd.Series,w_train: pd.Series,best_iter: int,best_parameters: dict[str, object],name: str,output_path: str,semilla: int) -> xgb.Booster:
    name = f"{name}_model_XGB"
    logger.info(f"Comienzo del entrenamiento del XGB : {name} en el mes train : {X_train['foto_mes'].unique()}")
    logger.info(f"Mejor cantidad de árboles para el mejor model {best_iter}")
    
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "seed": int(semilla),
        "verbosity": 0,
        **best_parameters
    }

    # 2) Pasá feature_names al DMatrix
    dtrain = xgb.DMatrix(
        data=X_train,
        label=y_train_binaria,
        weight=w_train)

    model_xgb = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=int(best_iter)
    )

    logger.info(f"comienzo del guardado en {output_path}")
    try:
        filename = output_path + f"{name}.txt"
        model_xgb.save_model(filename)
        logger.info(f"Modelo {name} guardado en {filename}")
        logger.info(f"Fin del entrenamiento del XGB en el mes train : {X_train['foto_mes'].unique()}")
    except Exception as e:
        logger.error(f"Error al intentar guardar el modelo {name}, por el error {e}")
        return
    return model_xgb


def grafico_feature_importance(model_xgb: Any, X_train: pd.DataFrame, name: str, output_path: str):
    logger.info("Comienzo del grafico de feature importance")
    os.makedirs(output_path, exist_ok=True)
    name = f"{name}_feature_importance"

    booster = model_xgb.get_booster() if hasattr(model_xgb, "get_booster") else model_xgb

    try:
        if isinstance(booster, xgb.Booster) and (booster.feature_names is None or any(f.startswith("f") for f in booster.feature_names)):
            booster.feature_names = X_train.columns.tolist()
    except Exception:
        pass  

    try:
        fig = plt.figure(figsize=(10, 20))
        ax = fig.add_subplot(111)
        xgb.plot_importance(
            booster,
            ax=ax,
            importance_type="gain",    
            show_values=False,
            max_num_features=50
        )
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f"{name}_grafico.png"), bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        logger.error(f"Error al intentar graficar los feat importances: {e}")

    try:
        scores = booster.get_score(importance_type="gain")
        if not scores:
            logger.warning("get_score() devolvió un dict vacío; puede que el modelo no tenga importancias disponibles.")
            importance_df = pd.DataFrame(columns=["feature", "importance", "importance_%"])
        else:
            mapping = {f"f{i}": col for i, col in enumerate(X_train.columns)}
            importance_df = (pd.Series(scores, name="importance")
                               .rename(index=lambda k: mapping.get(k, k))
                               .reset_index()
                               .rename(columns={"index": "feature"}))
            importance_df = importance_df.sort_values("importance", ascending=False)
            total = importance_df["importance"].sum()
            importance_df["importance_%"] = (importance_df["importance"] / total * 100.0) if total else 0.0

        importance_df.to_excel(os.path.join(output_path, f"{name}_data_frame.xlsx"), index=False)
        logger.info("Guardado feat imp en excel con ÉXITO")
    except Exception as e:
        logger.error(f"Error al intentar guardar los feat imp en excel por {e}")

    logger.info("Fin del grafico de feature importance")

def prediccion_test_xgb(X, model_xgb):
    mes=X["foto_mes"].unique()
    logger.info(f"comienzo prediccion del modelo en el mes {mes}")
    dtest = xgb.DMatrix(X) 
    y_pred = model_xgb.predict(dtest)
    logger.info("Fin de la prediccion del modelo")
    return y_pred

# Los calculos de curvas de ganancias e histograma estan en el lgbm
