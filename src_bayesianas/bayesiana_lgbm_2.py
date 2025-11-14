#main.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import datetime 
import logging
import json
import lightgbm as lgb
from src.config import *
from src.preprocesamiento import split_train_test_apred
from src.lgbm_optimizacion import optim_hiperp_binaria , graficos_bayesiana
from src.optimization_ZS import optimizar_zero_shot
## ---------------------------------------------------------Configuraciones Iniciales -------------------------------


## Carga de variables
n_trials=N_TRIALS



logger = logging.getLogger(__name__)

## --------------------------------------------------------Funcion main ------------------------------------------

def lanzar_bayesiana_lgbm(fecha:str , semillas:list,n_experimento:str|int ,proceso_ppal:str):
    #"""---------------------- CAMBIAR INPUTS --------------------------------------------------------"""
    numero=n_experimento
    #"""----------------------------------------------------------------------------------------------"""
    # name=f"bayesiana_{numero}_lgbm_{fecha}"
    tipo_bayesiana = TIPO_BAYESIANA
    name=f"{proceso_ppal}_{numero}_LGBM_TIPO_{tipo_bayesiana}_{len(semillas)}_SEMILLAS_{N_TRIALS}_TRIALS_{N_BOOSTS}_BOOSTS"
    nombre_log=f"log_{name}_{fecha}"
    logger.info(f"Inicio de ejecucion del flujo : {name}")
    if tipo_bayesiana =="OB":
        logger.info(f"=== COMIENZO OPTIMIZACIÓN {tipo_bayesiana} ===")
        X_train, y_train_binaria,y_train_class, w_train, X_test, y_test_binaria, y_test_class, w_test,X_apred, y_apred = split_train_test_apred(numero,MES_TRAIN,MES_TEST,MES_A_PREDECIR,SEMILLA,SUBSAMPLEO)    
        study = optim_hiperp_binaria(X_train , y_train_binaria,w_train ,n_trials , name,fecha,semillas)
        graficos_bayesiana(study ,fecha, name)
        logger.info("=== ANÁLISIS DE RESULTADOS ===")
        trials_df = study.trials_dataframe()
        if len(trials_df) > 0:
            top_5 = trials_df.nlargest(5, 'value')
            logger.info("Top 5 mejores trials:")
            for idx, trial in top_5.iterrows():
                logger.info(f"  Trial {trial['number']}: {trial['value']:,.0f}")
    
        logger.info(f"=== OPTIMIZACIÓN {tipo_bayesiana} COMPLETADA ===")
    elif tipo_bayesiana =="ZS":
        resultado_zs = optimizar_zero_shot(archivo_base=name, feature_subset=None )
        # Desempacar resultados del diccionario
        ganancia_val = resultado_zs["ganancia_validacion"]
        umbral_sugerido = resultado_zs["umbral_sugerido"]
        params_lightgbm = resultado_zs["best_params_lightgbm"]
        hyperparams = resultado_zs["best_params_flaml"]
        paths = resultado_zs["paths"]

        logger.info("=== ANÁLISIS DE RESULTADOS ===")
        logger.info(f"✅ Ganancia en validación: {ganancia_val:,.0f}")
        logger.info(f"✅ Umbral sugerido: {umbral_sugerido:.4f}")
        logger.info(f"✅ Parámetros FLAML guardados: {len(hyperparams)} parámetros")
        logger.info(f"✅ Parámetros LightGBM guardados: {len(params_lightgbm)} parámetros")
        logger.info(f"✅ Archivos generados:")
        logger.info(f"   - Iteraciones: {paths['iteraciones']}")
        logger.info(f"   - Best params: {paths['best_params']}")

        logger.info(f"=== OPTIMIZACIÓN {tipo_bayesiana} COMPLETADA ===")


    
    logger.info(f">>> Ejecucion finalizada. Revisar logs para mas detalles. {nombre_log}")

