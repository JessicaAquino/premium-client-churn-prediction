#lgbm_optimizacion.py
import pandas as pd
import polars as pl
import numpy as np
import lightgbm as lgb
import os

from joblib import Parallel, delayed
import optuna
from optuna.study import Study
from time import time
from datetime import datetime
import json
import logging
from optuna.samplers import TPESampler # Para eliminar el componente estocastico de optuna
from optuna.visualization import plot_param_importances, plot_contour,  plot_slice, plot_optimization_history

from src.config import GANANCIA,ESTIMULO,SEMILLA ,N_BOOSTS ,N_FOLDS, MES_VAL_BAYESIANA, MES_TRAIN
from src.config import  path_output_bayesian_db,path_output_bayesian_bestparams ,path_output_bayesian_best_iter ,path_output_bayesian_graf


logger = logging.getLogger(__name__)


def lgb_gan_eval_individual(y_pred, data):
    logger.info("Calculo ganancia INDIVIDUAL")
    weight = data.get_weight()
    ganancia = np.where(weight == 1.00002, GANANCIA, 0) - np.where(weight < 1.00002, ESTIMULO, 0)
    ganancia = ganancia[np.argsort(y_pred)[::-1]]
    ganancia = np.cumsum(ganancia)
    #con polars
    # df_eval = pl.DataFrame({"y_pred":y_pred , "weight":weight})
    # df_sorted = df_eval.sort("y_pred" , descending=True)
    # df_sorted = df_sorted.with_columns([pl.when(pl.col('weight') == 1.00002).then(GANANCIA).otherwise(-ESTIMULO).alias('ganancia_individual')])
    # df_sorted = df_sorted.with_columns([pl.col('ganancia_individual').cum_sum().alias('ganancia_acumulada')])
    # ganancia_maxima = df_sorted.select(pl.col('ganancia_acumulada').max()).item()
    # id_gan_max = df_sorted["ganancia_acumulada"].arg_max()
    # media_meseta = df_sorted.slice(id_gan_max-500, 1000)['ganancia_acumulada'].mean()
    logger.info(f"ganancia max acumulada : {np.max(ganancia)}")
    logger.info(f"cliente optimo : {np.argmax(ganancia)}")
    return 'gan_eval', np.max(ganancia) , True

def lgb_gan_eval_ensamble(y_pred , data):
    logger.info("Calculo ganancia ENSAMBLE")
    weight = data.get_weight()
    ganancia =np.where(weight == 1.00002 , GANANCIA, 0) - np.where(weight < 1.00002 , ESTIMULO ,0)
    ganancia_sorted = ganancia[np.argsort(y_pred)[::-1]]
    ganancia_acumulada = np.cumsum(ganancia_sorted)
    ganancia_max = np.max(ganancia_acumulada)
    idx_max_gan = np.argmax(ganancia_acumulada)
    logger.info(f"ganancia max acumulada : {ganancia_max}")
    logger.info(f"cliente optimo : {idx_max_gan}")
    ganancia_media_meseta = np.mean(ganancia_acumulada[idx_max_gan-500 : idx_max_gan+500])
    logger.info(f"ganancia media meseta : {ganancia_media_meseta}")
    return ganancia_media_meseta ,idx_max_gan ,ganancia_max

def optim_hiperp_binaria(X_train:pd.DataFrame | pl.DataFrame ,y_train_binaria:pd.Series|pl.Series|np.ndarray,w_train:pd.Series|pl.Series|np.ndarray, n_trials:int, name:str,fecha,semillas:list)-> Study:
    logger.info(f"Comienzo optimizacion hiperp binario: {name}")
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

    def objective(trial):
        num_leaves = trial.suggest_int('num_leaves', 8, 100)
        learning_rate = trial.suggest_float('learning_rate', 0.003, 0.1) 
        min_data_in_leaf = trial.suggest_int('min_data_in_leaf', 10, 1600)
        feature_fraction = trial.suggest_float('feature_fraction', 0.1, 1.0)
        bagging_fraction = trial.suggest_float('bagging_fraction', 0.1, 1.0)
        lambda_l1 = trial.suggest_float('lambda_l1', 1e-3, 10.0, log=True)
        lambda_l2 = trial.suggest_float('lambda_l2', 1e-3, 10.0, log=True)
        # Agregar maxdepth
        # Agregar lo de regularizacion

        params = {
            'objective': 'binary',
            'metric': 'custom',
            'boosting_type': 'gbdt',
            'first_metric_only': True,
            'boost_from_average': True,
            'feature_pre_filter': False,
            'max_bin': 31,
            'num_leaves': num_leaves,
            'learning_rate': learning_rate,
            'min_data_in_leaf': min_data_in_leaf,
            'feature_fraction': feature_fraction,
            'bagging_fraction': bagging_fraction,
            'bagging_freq': 1 , # 
            'lambda_l1':lambda_l1,
            'lambda_l2':lambda_l2,
            'extra_trees' : True,
            'verbose': -1
        }
        train_data = lgb.Dataset(X_train,label=y_train_binaria,weight=w_train)
        val_data = lgb.Dataset(X_val,label=y_val_binaria,weight=w_val)
        y_preds=[]
        best_iters=[]
        for semilla in semillas:
            params['seed'] = semilla
            model_i = lgb.train(
                    params=params,
                    train_set=train_data,
                    num_boost_round=N_BOOSTS,
                    valid_sets=[val_data],
                    valid_names=['valid'],
                    feval=lgb_gan_eval_individual,
                    callbacks=[
                        lgb.early_stopping(stopping_rounds=int(50 + 5/learning_rate), verbose=False),
                        lgb.log_evaluation(period=200),
                        ]
                    )
            y_pred_i = model_i.predict(X_val,num_iteration=model_i.best_iteration)
            y_preds.append(y_pred_i)
            best_iters.append(model_i.best_iteration)
        y_preds_matrix = np.vstack(y_preds)
        y_pred_ensamble = np.mean(y_preds_matrix , axis=0)
        ganancia_media_meseta , cliente_optimo,ganancia_max = lgb_gan_eval_ensamble(y_pred_ensamble , val_data)
        best_iter_promedio =  np.mean(best_iters)


        guardar_iteracion(trial,ganancia_media_meseta,cliente_optimo,ganancia_max,best_iter_promedio,y_preds_matrix,best_iters,name,fecha,semillas)

        return float(ganancia_media_meseta) * num_meses
        
    storage_name = "sqlite:///" + path_output_bayesian_db + "optimization_lgbm.db"
    study_name = f"study_{name}"    # VAria en numero de bayesiana y len(semillas)


    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True
        #sampler=TPESampler(seed=SEMILLA)
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




def _ganancia_prob(y_hat:pd.Series|np.ndarray , y:pd.Series|np.ndarray ,prop=1,class_index:int =1,threshold:int=0.025)->float:
    logger.info("comienzo funcion ganancia con threhold = 0.025")
    @np.vectorize
    def _ganancia_row(predicted , actual , threshold=0.025):
        return (predicted>=threshold) * (GANANCIA if actual=="BAJA+2" else -ESTIMULO)
    logger.info("Finalizacion funcion ganancia con threhold = 0.025")
    return _ganancia_row(y_hat[:,class_index] ,y).sum() /prop
