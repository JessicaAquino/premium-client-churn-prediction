#lgbm_optimizacion.py
import pandas as pd
import numpy as np
import lightgbm as lgb
import polars as pl
import os

from joblib import Parallel, delayed
import optuna
from optuna.study import Study
from time import time

import json
import logging
from optuna.samplers import TPESampler # Para eliminar el componente estocastico de optuna
from optuna.visualization import plot_param_importances, plot_contour,  plot_slice, plot_optimization_history

from src.config import GANANCIA,ESTIMULO,SEMILLA ,N_BOOSTS ,N_FOLDS, MES_VAL_BAYESIANA, MES_TRAIN
from src.config import  path_output_bayesian_db,path_output_bayesian_bestparams ,path_output_bayesian_best_iter ,path_output_bayesian_graf

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
import json
import logging

# Reutilizo tus constantes/paths
# from src.config import SEMILLA, N_BOOSTS, N_FOLDS, db_path, best_iter_path, bestparams_path, GANANCIA, ESTIMULO
logger = logging.getLogger(__name__)

ganancia_acierto = GANANCIA
costo_estimulo   = ESTIMULO

# === Métrica custom de ganancia para XGBoost ===
def xgb_gan_eval_individual(preds: np.ndarray, dtrain: xgb.DMatrix):
    weight = dtrain.get_weight()
    ganancia = np.where(weight == 1.00002, ganancia_acierto, 0) - np.where(weight < 1.00002, costo_estimulo, 0)
    ganancia = ganancia[np.argsort(preds)[::-1]]
    ganancia = np.cumsum(ganancia)
    logger.info(f"ganancia max acumulada : {np.max(ganancia)}")
    logger.info(f"cliente optimo : {np.argmax(ganancia)}")
    return 'gan_eval', float(np.max(ganancia))

def xgb_gan_eval_ensamble(preds: np.ndarray, dtrain: xgb.DMatrix):
    logger.info("Calculo ganancia ENSAMBLE")
    weight = dtrain.get_weight()
    ganancia = np.where(weight == 1.00002, ganancia_acierto, 0) - np.where(weight < 1.00002, costo_estimulo, 0)
    ganancia_sorted = ganancia[np.argsort(preds)[::-1]]
    ganancia_acumulada = np.cumsum(ganancia_sorted)
    ganancia_maxima = np.max(ganancia_acumulada)
    idx_max_gan = np.argmax(ganancia_acumulada)
    logger.info(f"ganancia max acumulada : {ganancia_maxima}")
    logger.info(f"cliente optimo : {idx_max_gan}")
    ganancia_media_meseta = np.mean(ganancia_acumulada[idx_max_gan-500 : idx_max_gan+500])
    logger.info(f"ganancia media meseta : {ganancia_media_meseta}")
    return  float(ganancia_media_meseta),int(idx_max_gan),float(ganancia_maxima)


def optim_hiperp_binaria_xgb(X_train: pd.DataFrame| pl.DataFrame,y_train_binaria: pd.Series|pl.Series,w_train: pd.Series|pl.Series,n_trials: int, name:str,fecha,semillas:list)-> Study:
    logger.info(f"Comienzo optimizacion hiperp binario (XGBoost) : {name}")
    # DMatrix con pesos
    if isinstance(X_train, pl.DataFrame):
        X_train = X_train.to_pandas()
    if isinstance(y_train_binaria, pl.Series):
        y_train_binaria = y_train_binaria.to_pandas()
    if isinstance(w_train, pl.Series):
        w_train = w_train.to_pandas()
    num_meses = len(MES_TRAIN)
    f_val = X_train["foto_mes"] == MES_VAL_BAYESIANA

    X_val = X_train.loc[f_val]
    y_val_binaria = y_train_binaria[X_val.index]
    w_val = w_train[X_val.index]

    X_train = X_train.loc[~f_val]
    y_train_binaria = y_train_binaria[X_train.index]
    w_train = w_train[X_train.index]
    logger.info(f"Meses train en bayesiana : {X_train['foto_mes'].unique()}")
    logger.info(f"Meses validacion en bayesiana : {X_val['foto_mes'].unique()}")


    

    def objective(trial: optuna.trial.Trial) -> float:
        max_depth         = trial.suggest_int('max_depth', 3, 12)
        eta               = trial.suggest_float('eta', 0.01, 0.2)                
        min_child_weight  = trial.suggest_float('min_child_weight', 1.0, 20.0)
        subsample         = trial.suggest_float('subsample', 0.5, 1.0)
        colsample_bytree  = trial.suggest_float('colsample_bytree', 0.5, 1.0)
        gamma             = trial.suggest_float('gamma', 0.0, 10.0)
        reg_lambda        = trial.suggest_float('lambda', 0.0, 20.0)              
        reg_alpha         = trial.suggest_float('alpha', 0.0, 10.0)              

        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',             
            'tree_method': 'hist',                
            'max_depth': max_depth,
            'eta': eta,
            'min_child_weight': min_child_weight,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'gamma': gamma,
            'lambda': reg_lambda,
            'alpha': reg_alpha,
            # 'seed': SEMILLA,
            'verbosity': 0
        }
        es_rounds = int(50 + 5 / eta)
        dtrain = xgb.DMatrix(data=X_train,label=y_train_binaria.values,weight=w_train.values )
        dval= xgb.DMatrix(data=X_val,label=y_val_binaria.values,weight=w_val.values )
        y_preds=[]
        best_iters=[]
        for semilla in semillas:
            params['seed'] = semilla
            model_i = xgb.train(
                params = params,
                dtrain=dtrain,
                num_boost_round=N_BOOSTS,
                evals=[(dval, "valid")],
                custom_metric=xgb_gan_eval_individual,
                callbacks=[
                    xgb.callback.EarlyStopping(
                        rounds=es_rounds,
                        metric_name="gan_eval",  
                        save_best=True,
                        maximize=True
                    )
                ],
                verbose_eval=False
            )
            best_iter = model_i.best_iteration
            best_iters.append(best_iter)
            y_pred_i = model_i.predict(dval, iteration_range=(0, best_iter + 1))
            y_preds.append(y_pred_i)

        y_preds_matrix = np.vstack(y_preds)    
        y_pred_ensamble = np.mean(y_preds_matrix, axis=0)
        ganancia_media_meseta, cliente_optimo, ganancia_max = xgb_gan_eval_ensamble(y_pred_ensamble,dval)
        best_iter_promedio = float(np.mean(best_iters))
        guardar_iteracion(trial,ganancia_media_meseta,cliente_optimo,ganancia_max,best_iter_promedio,y_preds_matrix,best_iters,name,fecha,semillas)
        return float(ganancia_media_meseta) * num_meses
    

    storage_name = "sqlite:///" + path_output_bayesian_db + "optimization_xgb.db"
    study_name = f"study_{name}"

    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
    )

    study.optimize(objective, n_trials=n_trials)

    return study



def guardar_iteracion(trial,ganancia_media_meseta,cliente_optimo,ganancia_max,best_iter_medio,
                      y_pred_i_lista , best_iters ,
                       name,fecha,semillas ):
    logger.info(f"Comienzo del guardado de la iteracion : {trial.number}")
    
    archivo = path_output_bayesian_bestparams + f"best_params_{name}.json"

    iteracion_data = {
        'trial_number': trial.number,
        'params': trial.params,
        'ganancia_media_meseta': float(ganancia_media_meseta),
        'cliente_optimo':int(cliente_optimo),
        'ganancia_max':float(ganancia_max),
        'best_iter_trial':int(best_iter_medio),
        # 'y_pred_i_lista':y_pred_i_lista,
        'best_iters':best_iters,
        'datetime': fecha,
        'state': 'COMPLETE',  # Si llegamos aquí, el trial se completó exitosamente
        'configuracion': {
            'semilla': semillas,
            'mes_train': MES_TRAIN,
            'mes_validacion': MES_VAL_BAYESIANA}
            }
    # Cargar datos existentes si el archivo ya existe
    if os.path.exists(archivo):
        with open(archivo, 'r') as f:
            try:
                datos_existentes = json.load(f)
                if not isinstance(datos_existentes, list):
                    datos_existentes = []
            except json.JSONDecodeError:
                datos_existentes = []
    else:
        datos_existentes = []

    # Agregar nueva iteración
    datos_existentes.append(iteracion_data)

    # Guardar todas las iteraciones en el archivo
    with open(archivo, 'w') as f:
        json.dump(datos_existentes, f, indent=2)

    logger.info(f"Iteración {trial.number} guardada en {archivo}")
    logger.info(f"Ganancia media meseta: {ganancia_media_meseta:,.0f}" + "---" + "Parámetros: {params}")



def graficos_bayesiana(study:Study, fecha:str,name: str):
    logger.info(f"Comienzo de la creacion de graficos de {name}")
    try:
        fig1 = plot_optimization_history(study)
        fig1.write_image(path_output_bayesian_graf+f"{fecha}_{name}_graficos_opt_history.png")

        fig2 = plot_param_importances(study)
        fig2.write_image(path_output_bayesian_graf+f"{fecha}_{name}_graficos_param_importances.png")

        fig3 = plot_slice(study)
        fig3.write_image(path_output_bayesian_graf+f"{fecha}_{name}_graficos_slice.png")

        fig4 = plot_contour(study)
        fig4.write_image(path_output_bayesian_graf+f"{fecha}_{name}_graficos_contour_all.png")

        fig5 = plot_contour(study, params=["num_leaves", "learning_rate"])
        fig5.write_image(path_output_bayesian_graf+f"{fecha}_{name}_graficos_contour_specific.png")

        logger.info(f" Gráficos guardados en {path_output_bayesian_graf}")
    except Exception as e:
        logger.error(f"Error al generar las gráficas: {e}")
