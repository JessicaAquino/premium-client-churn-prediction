#lgbm_train_test.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.model_selection import StratifiedShuffleSplit

from typing import Tuple
import logging
from time import time
import datetime

import pickle
import json

from src.config import *

ganancia_acierto = GANANCIA
costo_estimulo=ESTIMULO

logger = logging.getLogger(__name__)



def entrenamiento_lgbm(X_train:pd.DataFrame ,y_train_binaria:pd.Series|np.ndarray,w_train:pd.Series|np.ndarray, best_iter:int, best_parameters:dict[str, object],name:str,output_path:str,semilla:int)->lgb.Booster:
    name=f"{name}_model_lgbm"
    logger.info(f"Comienzo del entrenamiento del lgbm : {name} en el mes train : {X_train['foto_mes'].unique()}")    
    print(f"Mejor cantidad de árboles para el mejor model best iter : {best_iter}")
    params = {
        'objective': 'binary',
        'metric': 'none',              
        'boosting_type': 'gbdt',
        'first_metric_only': True,
        'boost_from_average': True,
        'feature_pre_filter': False,
        'max_bin': 31,
        'num_leaves': best_parameters['num_leaves'],
        'learning_rate': best_parameters['learning_rate'],
        'min_data_in_leaf': best_parameters['min_data_in_leaf'],
        'feature_fraction': best_parameters['feature_fraction'],
        'bagging_fraction': best_parameters['bagging_fraction'],
        'bagging_freq': 1,
        'lambda_l1': best_parameters['lambda_l1'],
        'lambda_l2': best_parameters['lambda_l2'],
        'extra_trees': True,
        'seed': semilla,
        'verbose': -1,
    }

    train_data = lgb.Dataset(X_train,
                            label=y_train_binaria,
                            weight=w_train)

    model_lgbm = lgb.train(params,
                    train_data,
                    num_boost_round=int(best_iter))

    logger.info(f"comienzo del guardado en {output_path}")
    try:
        filename=output_path+f'{name}.txt'
        model_lgbm.save_model(filename )                         
        logger.info(f"Modelo {name} guardado en {filename}")
        logger.info(f"Fin del entrenamiento del LGBM en el mes train : {X_train['foto_mes'].unique()}")
    except Exception as e:
        logger.error(f"Error al intentar guardar el modelo {name}, por el error {e}")
        return
    return model_lgbm

def entrenamiento_zlgbm(X_train:pd.DataFrame ,y_train_binaria:pd.Series|np.ndarray,name:str,output_path:str,semilla:int)->lgb.Booster:
    lgbm_params = {
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'learning_rate': LEARNING_RATE,
        'num_leaves': NUM_LEAVES,
        'feature_fraction': FEATURE_FRACTION,
        'bagging_fraction': BAGGING_FRACTION,
        'bagging_freq': BAGGING_FREQ,
        'min_data_in_leaf': MIN_DATA_IN_LEAF,
        'max_bin': MAX_BIN,
        'verbose': -1,
        'seed': semilla,
        'force_row_wise': True,
    }
    
    if GRADIENT_BOUND is not None:
        lgbm_params['gradient_bound'] = GRADIENT_BOUND
    train_data = lgb.Dataset(X_train, label=y_train_binaria, free_raw_data=True)
    model_lgbm = lgb.train(
        lgbm_params,
        train_data,
        num_boost_round=NUM_BOOST_ROUND
    )   
    logger.info(f"comienzo del guardado en {output_path}")
    try:
        filename=output_path+f'{name}.txt'
        model_lgbm.save_model(filename)                         
        logger.info(f"Modelo {name} guardado en {filename}")
        logger.info(f"Fin del entrenamiento del ZLGBM en el mes train : {X_train['foto_mes'].unique()}")
    except Exception as e:
        logger.error(f"Error al intentar guardar el modelo {name}, por el error {e}")
        return 
    return model_lgbm
    
def entrenamiento_lgbm_zs(X_train:pd.DataFrame ,y_train_binaria:pd.Series|np.ndarray,w_train:pd.Series|np.ndarray,best_iter:int, best_parameters:dict[str, object],name:str,output_path:str,semilla:int)->lgb.Booster:
    name=f"{name}_model_lgbm"
    logger.info(f"Comienzo del entrenamiento del lgbm con ZS : {name} en el mes train : {X_train['foto_mes'].unique()}")    
    print(f"Mejor cantidad de árboles para el mejor model best iter : {best_iter}")
    params = best_parameters
    params["seed"]=semilla
    params["max_bin"]=31
    if params["feature_fraction"]==1.0:
        params["feature_fraction"]=0.8
    if params["bagging_fraction"]==1.0:
        params["bagging_fraction"]=0.8
    if best_iter>3000:
        best_iter= 3000
    
    params["bagging_freq"] = 1

    

    logger.info(f"semilla del entrenamiento :{params['seed']}")
    train_data = lgb.Dataset(X_train,
                            label=y_train_binaria,
                            weight=w_train)
    model_lgbm = lgb.train( params,
                      train_set=train_data,
                      num_boost_round=int(best_iter))   

    logger.info(f"comienzo del guardado en {output_path}")
    try:
        filename=output_path+f'{name}.txt'
        model_lgbm.save_model(filename )                         
        logger.info(f"Modelo {name} guardado en {filename}")
        logger.info(f"Fin del entrenamiento del LGBM en el mes train : {X_train['foto_mes'].unique()}")
    except Exception as e:
        logger.error(f"Error al intentar guardar el modelo {name}, por el error {e}")
        return
    return model_lgbm


def grafico_feature_importance(model_lgbm:lgb.Booster,X_train:pd.DataFrame,name:str,output_path:str):
    logger.info("Comienzo del grafico de feature importance")
    name=f"{name}_feature_importance"
    try:
        lgb.plot_importance(model_lgbm, figsize=(10, 20))
        plt.savefig(output_path+f"{name}_grafico.png", bbox_inches='tight')
        plt.close()
    except Exception as e:
        logger.error(f"Error al intentar graficar los feat importances: {e}")
    logger.info("Fin del grafico de feature importance")

    importances = model_lgbm.feature_importance()
    feature_names = X_train.columns.tolist()
    importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    importance_df = importance_df.sort_values('importance', ascending=False)
    importance_df["importance_%"] = (importance_df["importance"] /importance_df["importance"].sum())*100
    # importance_df[importance_df['importance'] > 0]
    logger.info("Guardado de feat import en excel")
    try :
        importance_df.to_excel(output_path+f"{name}_data_frame.xlsx" ,index=False)
        logger.info("Guardado feat imp en excel con EXITO")
    except Exception as e:
        logger.error(f"Error al intentar guardar los feat imp en excel por {e}")

def prediccion_test_lgbm(X:pd.DataFrame ,  model_lgbm:lgb.Booster)-> pd.Series:
    mes=X["foto_mes"].unique()
    logger.info(f"comienzo prediccion del modelo en el mes {mes}")
    y_pred_lgbm = model_lgbm.predict(X)
    logger.info("Fin de la prediccion del modelo")
    return y_pred_lgbm


def ganancia_umbral_prob(y_pred:pd.Series,y_test_class:pd.Series ,prop=1,threshold:float=0.025)->float:
    # logger.info(f"comienzo funcion ganancia con threshold = {threshold}")
    ganancia = np.where(y_test_class=="BAJA+2" , ganancia_acierto,0) - np.where(y_test_class!="BAJA+2" , costo_estimulo,0)
    # logger.info(f"fin evaluacion modelo.")
    return ganancia[y_pred >= threshold].sum() / prop

def ganancia_umbral_cliente(y_pred:pd.Series , y_test_class:pd.Series , prop =1 , n_clientes :int=10000)-> float:
    threshold_cliente = int(n_clientes * prop)
    ganancia = np.where(y_test_class=="BAJA+2" , ganancia_acierto,0) - np.where(y_test_class!="BAJA+2" , costo_estimulo,0)
    idx_sorted=np.argsort(y_pred)[::-1]
    ganancia_sorted=ganancia[idx_sorted]
    return ganancia_sorted[:threshold_cliente].sum()/ prop


def calc_estadisticas_ganancia(y_test_class:pd.Series ,y_pred_lgbm:pd.Series ,name:str,output_path:str , semilla:int , guardar:bool)-> Tuple[dict,pd.Series,np.ndarray]:
    name=f"{name}_umbral_optimo"
    logger.info(f"Comienzo del calculo de las estadisticas de la ganancia para la semilla {semilla}")

    ganancia = np.where(y_test_class=="BAJA+2" , ganancia_acierto,0) - np.where(y_test_class!="BAJA+2" , costo_estimulo,0)
    try:
        idx_sorted = np.argsort(y_pred_lgbm)[::-1]
        y_pred_sorted = y_pred_lgbm[idx_sorted]

        ganancia_sorted = ganancia[idx_sorted]
        ganancia_acumulada=np.cumsum(ganancia_sorted)
        ganancia_max_acumulada = np.max(ganancia_acumulada)
        indx_ganancia_max_acumulada = np.argmax(ganancia_acumulada)
        umbral_optimo = y_pred_sorted[indx_ganancia_max_acumulada]
        ganancia_media_meseta = np.mean(ganancia_acumulada[indx_ganancia_max_acumulada-500:indx_ganancia_max_acumulada+500 ])

        
    except Exception as e:
        logger.error(f"Hubo un error por {e}")
        raise

    logger.info(f"Umbral_prob optimo = {umbral_optimo}")
    logger.info(f"Numero de cliente optimo : {indx_ganancia_max_acumulada}")
    logger.info(f"Ganancia maxima con el punto optimo : {ganancia_max_acumulada}")
    logger.info(f"Ganancia media meseta alrededor del punto optimo : {ganancia_media_meseta}")
    estadisticas_ganancia = {
    "umbral_optimo": float(umbral_optimo),
    "cliente": int(indx_ganancia_max_acumulada),
    "ganancia_max": float(ganancia_max_acumulada),
    "ganancia_media_meseta" : float(ganancia_media_meseta),
    "SEMILLA":semilla
    }
    if guardar :
        try:
            with open(output_path+f"{name}.json", "w") as f:
                json.dump(estadisticas_ganancia, f, indent=4)
        except Exception as e:
            logger.error(f"Error al intentar guardar el dict de estadisticas de la ganancia como json --> {e}")
        logger.info(f"Las estadisticas de la ganancia son : {estadisticas_ganancia}")
        logger.info("Fin de la prediccion de Las estadisticas de la ganancia ")
    else:
        logger.info("No se guarda porque se va a guardar todos los umbrales despues del for")
        logger.info("Fin del calculo de las estadisticas de la ganancia")
    # resultados_ganancia = {"estadistica_ganancias":estadisticas_ganancia,"y_pred_sorted":y_pred_sorted ,"ganancia_acumulada": ganancia_acumulada }
    return estadisticas_ganancia , y_pred_sorted , ganancia_acumulada


def grafico_curvas_ganancia(y_pred_sorted:pd.Series|dict[pd.Series] , ganancia_acumulada:pd.Series|dict[pd.Series], estadisticas_ganancia_dict:dict,name:str, output_path:str):
    name=f"{name}_curvas_ganancia"
    piso=4000
    techo=20000
    if not (isinstance(y_pred_sorted,dict)):
        logger.info(f"Comienzo de los graficos de curva de ganancia con una semilla : {estadisticas_ganancia_dict['SEMILLA']} ")
        umbral_optimo= estadisticas_ganancia_dict["umbral_optimo"]
        indx_ganancia_max_acumulada = estadisticas_ganancia_dict["cliente"]
        ganancia_max_acumulada= estadisticas_ganancia_dict["ganancia_max"]
        ganancia_media_meseta = estadisticas_ganancia_dict["ganancia_media_meseta"]

        try:
            plt.figure(figsize=(10, 6))
            plt.plot(y_pred_sorted[piso:techo] ,ganancia_acumulada[piso:techo] ,label=f" ganancia media meseta a {ganancia_media_meseta} /ganancia max a {ganancia_max_acumulada} / punto de corte a {umbral_optimo}")
            plt.xlabel('Predicción de probabilidad')
            plt.ylabel('Ganancia')
            plt.title(f"Curva Ganancia respecto a probabilidad: {name}")
            plt.axvline(x=0.025 , color="red" , linestyle="--" ,label="Punto de Corte a 0.025")
            plt.axvline(x=umbral_optimo , color="green" , linestyle="--")
            plt.axhline(y=ganancia_max_acumulada , color="green" , linestyle="--")
            plt.axhline(y=ganancia_media_meseta , color="orange" , linestyle="--")
            plt.legend()
            plt.savefig(output_path+f"{name}_probabilidad.png", bbox_inches='tight')
            logger.info("Creacion de los graficos Curva Ganancia respecto a probabilidad")
        except Exception as e:
            logger.error(f"Error al tratar de crear los graficos Curva Ganancia respecto a probabilidad --> {e}")

        try:
            plt.figure(figsize=(10, 6))
            plt.plot(range(piso,len(ganancia_acumulada[piso:techo])+piso) ,ganancia_acumulada[piso:techo] ,label=f"ganancia max a {ganancia_max_acumulada} / punto de corte a {indx_ganancia_max_acumulada} / ganancia media meseta {ganancia_media_meseta}")
            plt.xlabel('Clientes')
            plt.ylabel('Ganancia')
            plt.title(f"Curva Ganancia con numero de clientes: {name}")
            plt.axvline(x=indx_ganancia_max_acumulada , color="green" , linestyle="--" )
            plt.axhline(y=ganancia_max_acumulada , color="green",linestyle="--" )
            plt.axhline(y=ganancia_media_meseta , color="orange" , linestyle="--")
            plt.legend()
            plt.savefig(output_path+f"{name}_numero_cliente.png", bbox_inches='tight')
            logger.info("Creacion de los graficos Curva Ganancia respecto al cliente")
        except Exception as e:
            logger.error(f"Error al tratar de crear los graficos Curva Ganancia respecto a probabilidad --> {e}")

    else:
        semillas =estadisticas_ganancia_dict.keys()
        logger.info(f"Comienzo de los graficos de curva de ganancia con varias semillas = {semillas}")
        plt.figure(figsize=(10, 6))
        valores_ordenados = sorted([v["ganancia_media_meseta"] for v in estadisticas_ganancia_dict.values()],reverse=True)
        ganancia_top_n = valores_ordenados[min(5, len(valores_ordenados) - 1)]

        for i,s in enumerate(semillas) :
            umbral_optimo=estadisticas_ganancia_dict[s]["umbral_optimo"]
            indx_max_ganancia_acumulada = estadisticas_ganancia_dict[s]["cliente"]
            ganancia_max_acumulada= estadisticas_ganancia_dict[s]["ganancia_max"]
            ganancia_media_meseta= estadisticas_ganancia_dict[s]["ganancia_media_meseta"]
            y_pred_sorted_s=y_pred_sorted[s]
            ganancia_acumulada_s = ganancia_acumulada[s]
            if s == "ensamble_semillas":
                alpha = 1
            else:
                alpha=0.3
            if (ganancia_media_meseta>= ganancia_top_n) | (s == "ensamble_semillas"):
                linea,=plt.plot(y_pred_sorted_s[piso:techo] ,ganancia_acumulada_s[piso:techo],alpha=alpha ,label=f"SEMILLA {s}  ganancia media meseta a {ganancia_media_meseta} / ganancia max a {ganancia_max_acumulada} / punto de corte a {umbral_optimo}")
            else :
                linea,=plt.plot(y_pred_sorted_s[piso:techo] ,ganancia_acumulada_s[piso:techo],alpha=alpha)

            color=linea.get_color()
            if i==0:
                plt.axvline(x=0.025 , color="red" , linestyle="--" ,label="Punto de Corte a 0.025" , alpha=0.3)
            
            plt.axvline(x=umbral_optimo , color=color , linestyle="--")
            plt.axhline(y=ganancia_media_meseta , color=color , linestyle="--")
            plt.xlabel('Predicción de probabilidad')
            plt.ylabel('Ganancia')
            plt.title(f"Curva Ganancia respecto a probabilidad: {name}")
            plt.grid(True, linestyle="--", alpha=0.3)
            plt.legend()
        try:
            logger.info("Guardando grafico de curva de ganancia con prob")
            plt.tight_layout()
            plt.savefig(output_path+f"{name}_probabilidad.png", bbox_inches='tight')
            plt.close()
        except Exception as e:
            logger.error(f"Error al intentar guardar el grafico de curva de ganancia de prob por {e}")
            
        plt.figure(figsize=(10, 6))
        for s in semillas :

            umbral_optimo=estadisticas_ganancia_dict[s]["umbral_optimo"]
            indx_max_ganancia_acumulada = estadisticas_ganancia_dict[s]["cliente"]
            ganancia_max_acumulada= estadisticas_ganancia_dict[s]["ganancia_max"]
            ganancia_media_meseta= estadisticas_ganancia_dict[s]["ganancia_media_meseta"]
            y_pred_sorted_s=y_pred_sorted[s]
            ganancia_acumulada_s = ganancia_acumulada[s]
            if s == "ensamble_semillas":
                alpha = 1
            else:
                alpha=0.3
            
            if (ganancia_media_meseta>= ganancia_top_n) | (s == "ensamble_semillas"):
                linea,=plt.plot(range(piso,len(ganancia_acumulada_s[piso:techo])+piso) ,ganancia_acumulada_s[piso:techo] ,alpha=alpha,label=f"SEMILLA {s}  ganancia media meseta a {ganancia_media_meseta} / ganancia max a {ganancia_max_acumulada} / punto de corte a {indx_max_ganancia_acumulada}")
            else :
                linea,=plt.plot(range(piso,len(ganancia_acumulada_s[piso:techo])+piso) ,ganancia_acumulada_s[piso:techo] ,alpha=alpha)

            color=linea.get_color()

            plt.axvline(x=indx_max_ganancia_acumulada , color=color , linestyle="--" )
            plt.axhline(y=ganancia_media_meseta , color=color,linestyle="--" )
            plt.xlabel('Clientes')
            plt.ylabel('Ganancia')
            plt.title(f"Curva Ganancia con numero de clientes: {name}")
            plt.grid(True, linestyle="--", alpha=0.3)
            plt.legend()
            
        try:
            logger.info("Guardando grafico de curva de ganancia con num cliente")
            plt.tight_layout()
            plt.savefig(output_path+f"{name}_numero_cliente.png", bbox_inches='tight')
            plt.close()
        except Exception as e:
            logger.error(f"Error al intentar guardar el grafico de curva de ganancia con n_cliente por {e}")
    return
def grafico_hist_ganancia(estadisticas_ganancia_dict: dict, name: str, output_path: str):
    name = f"{name}_hist_ganancia"
    ganancias = [v["ganancia_media_meseta"] for k,v in estadisticas_ganancia_dict.items() if k !="ensamble_semillas"]
    logger.info(f"ganancias : {ganancias}")
    # Crear figura
    plt.figure(figsize=(8, 5))
    sns.histplot(ganancias, bins=20, kde=True, color="skyblue", edgecolor="black")
    
    plt.title(f"Histograma por semilla de: {name}", fontsize=12)
    plt.xlabel("Ganancia media meseta")
    plt.ylabel("Frecuencia")
    plt.grid(True, alpha=0.3)
    
    logger.info(f"Intento de guardado en {output_path}")
    try:
        plt.tight_layout()
        plt.savefig(output_path + f"{name}.png", bbox_inches='tight')
        plt.close()
        logger.info(f"Guardado con éxito en {output_path+f'{name}.png'}")
    except Exception as e:
        logger.error(f"Error al intentar guardar por {e}")






# Funciones para tener en cuenta el private y el public
def evaluacion_public_private(X_test:pd.DataFrame , y_test_class:pd.Series , y_pred_model:pd.Series,umbral_mode:str,umbral:float|int,semilla:int,n_splits)->pd.DataFrame:
    logger.info(f"Comienzo de Calculo de las ganancias de public and private con {n_splits} splits")
    sss=StratifiedShuffleSplit(n_splits=n_splits,test_size=0.3,random_state=semilla)
    modelos={"lgbm":y_pred_model}
    rows=[]
    for private_index , public_index in sss.split(X_test , y_test_class):
        row={}
        for model_id , y_pred in modelos.items():
            y_true_private = y_test_class.iloc[private_index]
            y_pred_private = y_pred[private_index]
            y_true_public = y_test_class.iloc[public_index]
            y_pred_public = y_pred[public_index]

            if umbral_mode == "prob":
                row[model_id+"_public"] = ganancia_umbral_prob(y_pred_public, y_true_public, 0.3,umbral)
                row[model_id+"_private"] =ganancia_umbral_prob(y_pred_private, y_true_private, 0.7,umbral)
            elif umbral_mode == "n_cliente":
                row[model_id+"_public"] = ganancia_umbral_cliente(y_pred_public, y_true_public, 0.3,umbral)
                row[model_id+"_private"] =ganancia_umbral_cliente(y_pred_private, y_true_private, 0.7,umbral)

        rows.append(row)

    df_lb = pd.DataFrame(rows)
    df_lb_long = df_lb.reset_index()
    df_lb_long = df_lb_long.melt(id_vars=['index'], var_name='model_type', value_name='ganancia')
    df_lb_long[['modelo', 'tipo']] = df_lb_long['model_type'].str.split('_', expand=True)
    df_lb_long = df_lb_long[['ganancia', 'tipo', 'modelo']]
    logger.info(f"Calculo de las ganancias en pub y priv realizado con {n_splits} splits")
    return df_lb_long
    

def graf_hist_ganancias_public_private(df_lb_long:pd.DataFrame|list[pd.DataFrame] ,name:str ,output_path : str ,semillas:list[str]):
    logger.info("Comienzo del grafico de los histogramas")
    name=f"{name}_graf_ganancia_histograma"
    if isinstance(df_lb_long,pd.DataFrame):
        logger.info("graficando unico histograma")
        logger.info(f"cantidad de valores de ganancia : {df_lb_long.shape}")
        try:
            g = sns.FacetGrid(df_lb_long, col="tipo", row="modelo", aspect=2)
            g.map(sns.histplot, "ganancia", kde=True)
            plt.title(name)
            plt.savefig(output_path+f"{name}.png", bbox_inches='tight')
        except Exception as e:
            logger.error(f"Error al intentar hacer el grafico de los histogramas {e}")
    elif isinstance(df_lb_long,list):
# Unir todos los dataframes y anotar su semilla
        dfs=[]
        for df, s in zip(df_lb_long, semillas):
            tmp = df.copy()
            tmp["semilla"] = s
            dfs.append(tmp)
        big = pd.concat(dfs, ignore_index=True)

        # Parámetros de layout
        n_modelos = big["modelo"].nunique()
        n_semillas = len(semillas)

        # Un único FacetGrid: filas=modelo, columnas=semilla, hue=tipo
        g = sns.displot(
            data=big,
            x="ganancia",
            row="modelo",
            col="semilla",
            hue="tipo",
            kind="hist",          # o kind="kde" si preferís solo KDE
            kde=True,             # deja True si querés hist + KDE
            element="step",       # bordes claros
            stat="density",       # comparable entre tipos
            common_norm=False,    # no normaliza entre 'tipo'
            facet_kws=dict(margin_titles=True, sharex=False, sharey=False),
            height=2.8,           # ajustá a gusto
            aspect=1.6            # ancho relativo de cada subgraf
        )

        # Etiquetas y estética
        g.set_axis_labels("Ganancia", "Densidad")
        for ax in g.axes.flat:
            ax.grid(True, alpha=0.2)

        # Título general y guardado
        g.figure.suptitle(f"Distribución de ganancias - {name}", y=1.02, fontsize=14)
        g.figure.tight_layout()
        g.figure.savefig(output_path + f"{name}.png", bbox_inches="tight")
        plt.close(g.figure)

        logger.info("Fin de los histogramas (facet por semilla x modelo, hue=tipo)")

    # elif isinstance(df_lb_long,list):
    #     logger.info("Grafico en grilla")
    #     n_graficos = len(df_lb_long)
    #     filas = int(np.ceil(np.sqrt(n_graficos)))
    #     columnas = int(np.ceil(np.sqrt(n_graficos)))

    #     fig,axes=plt.subplots(filas,columnas , figsize=(columnas*4, filas*3))
    #     if isinstance(axes, np.ndarray):
    #         axes = axes.flatten()
    #     else:
    #         axes = np.array([axes])

    #     for i,(df,semilla) in enumerate(zip(df_lb_long,semillas)) :
    #         ax=axes[i]
    #         sns.histplot(df, x="ganancia", kde=True, ax=ax)
    #         ax.set_title(f"SEED_{semilla}")
    #     for j in range(n_graficos, filas*columnas):
    #         fig.delaxes(axes[j])
    #     try:
    #         fig.tight_layout()
    #         fig.suptitle(f"Distribución de ganancias - {name}", fontsize=14, y=1.02)
    #         fig.savefig(output_path+f"{name}.png", bbox_inches="tight")
    #         plt.close(fig) 
    #     except Exception as e:
    #         logger.error(f"Error al guardar el grafico de grillas de hist {e}")

    # logger.info("Fin de los histogramas de public and private")


## PREDICCION FINAL-------------------------------------------------------------------


def preparacion_ypred_kaggle( y_apred:pd.DataFrame, y_pred:pd.Series ,umbral_cliente:int , name:str ,output_path:str) -> pd.DataFrame:
    logger.info("Comienzo de la preparacion de las predicciones finales")
    name = name+"_predicciones_finales"
    y_apred = y_apred.copy()
    y_apred["prediction"] = y_pred 
    y_apred= y_apred.sort_values(by="prediction" , ascending=False)
    k = int(np.floor(umbral_cliente))
    y_apred["prediction"] = 0

    y_apred.iloc[:k , y_apred.columns.get_loc("prediction")] = 1
    logger.info(f"cantidad de bajas predichas : {int((y_apred['prediction']==1).sum())}")
    y_apred = y_apred.set_index("numero_de_cliente")
    file_name=output_path+name+".csv"
    try:
        y_apred.to_csv(file_name)
        logger.info(f"predicciones guardadas en {file_name}")
    except Exception as e:
        logger.error(f"Error al intentar guardar las predicciones --> {e}")
        raise
    return y_apred


