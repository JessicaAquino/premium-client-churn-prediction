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
from src.xgb_optimizacion import optim_hiperp_binaria_xgb,graficos_bayesiana
## ---------------------------------------------------------Configuraciones Iniciales -------------------------------


## Carga de variables
n_trials=N_TRIALS



logger = logging.getLogger(__name__)

## --------------------------------------------------------Funcion main ------------------------------------------

def lanzar_bayesiana_xgb(fecha:str , semillas:list,n_experimento:str|int ,proceso_ppal:str):
    #"""---------------------- CAMBIAR INPUTS --------------------------------------------------------"""
    numero=n_experimento 
    #"""----------------------------------------------------------------------------------------------"""
    name=f"{proceso_ppal}_{numero}_XGB_{len(semillas)}_SEMILLAS_{N_TRIALS}_TRIALS_{N_BOOSTS}_BOOSTS"
    nombre_log=f"log_{name}_{fecha}"
    logger.info(f"Inicio de ejecucion del flujo : {name}")
    
    X_train, y_train_binaria,y_train_class, w_train, X_test, y_test_binaria, y_test_class, w_test,X_apred, y_apred = split_train_test_apred(numero,MES_TRAIN,MES_TEST,MES_A_PREDECIR,SEMILLA,SUBSAMPLEO)

    
    ## 4.a. Optimizacion Hiperparametros

    study = optim_hiperp_binaria_xgb(X_train , y_train_binaria,w_train ,n_trials , name,fecha,semillas)
    graficos_bayesiana(study ,fecha, name)

    logger.info("=== ANÁLISIS DE RESULTADOS ===")
    trials_df = study.trials_dataframe()
    if len(trials_df) > 0:
        top_5 = trials_df.nlargest(5, 'value')
        logger.info("Top 5 mejores trials:")
        for idx, trial in top_5.iterrows():
            logger.info(f"  Trial {trial['number']}: {trial['value']:,.0f}")
  
    
    logger.info(f">>> Ejecucion finalizada. Revisar logs para mas detalles. {nombre_log}")


