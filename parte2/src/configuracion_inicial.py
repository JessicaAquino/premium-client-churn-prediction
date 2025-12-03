import logging
from src.config import *
import json
import pandas as pd
import duckdb


def creacion_directorios():
    ## Creacion de las carpetas
    if in_gcp :
        os.makedirs("datasets/",exist_ok=True)
    os.makedirs(PATH_INPUT_DATA,exist_ok=True)
            #LOGS PATHS
    os.makedirs(PATH_LOGS,exist_ok=True)
    os.makedirs(PATH_LOG_GLOBAL,exist_ok=True)
            #OUTPUT PATHS
    os.makedirs(PATH_OUTPUT_DATA,exist_ok=True)
    os.makedirs(PATH_OUTPUT_BAYESIAN,exist_ok=True)
    os.makedirs(PATH_OUTPUT_FINALES,exist_ok=True)
    os.makedirs(PATH_OUTPUT_EXPERIMENTOS,exist_ok=True)
    os.makedirs(PATH_OUTPUT_EDA , exist_ok=True)
            #BAYESIANA
    os.makedirs(path_output_bayesian_db,exist_ok=True)
    os.makedirs(path_output_bayesian_bestparams,exist_ok=True)
    os.makedirs(path_output_bayesian_best_iter,exist_ok=True)
    os.makedirs(path_output_bayesian_graf,exist_ok=True)
            #FINALES
    os.makedirs(path_output_finales_model,exist_ok=True)
    os.makedirs(path_output_finales_feat_imp,exist_ok=True)
    os.makedirs(path_output_prediccion_final,exist_ok=True)
            #EXPERIMENTOS
    os.makedirs(path_output_exp_model,exist_ok=True)
    os.makedirs(path_output_exp_feat_imp,exist_ok=True)
    os.makedirs(path_output_exp_graf_gan_hist_total,exist_ok=True)
    os.makedirs(path_output_exp_graf_curva_ganancia,exist_ok=True)
    os.makedirs(path_output_exp_umbral,exist_ok=True)
    os.makedirs(path_output_exp_prediction,exist_ok=True)


def creacion_logg_local(nombre_log:str):
    logging.basicConfig(
        level=logging.INFO, #Puede ser INFO o ERROR
        format='%(asctime)s - %(levelname)s - %(name)s  - %(funcName)s -  %(lineno)d - %(message)s',
        handlers=[
            logging.FileHandler(f"{PATH_LOGS}/{nombre_log}", mode="w", encoding="utf-8"),
            logging.StreamHandler()
        ]
    )

def creacion_logg_global(fecha:str, competencia:str, proceso_ppal:str, n_experimento:str,n_semillas:int,in_gcp:bool=in_gcp):
    if proceso_ppal =="analisis_exploratorio":
        registro = {
            "fecha": fecha,
            "competencia": competencia,
            "proceso_ppal": proceso_ppal,
            "n_experimento": n_experimento,
            "in_gcp":in_gcp
        }
    else:
        registro = {
            "fecha": fecha,
            "competencia": competencia,
            "proceso_ppal": proceso_ppal,
            "n_experimento": n_experimento,
            "n_semillas": n_semillas,
            "in_gcp":in_gcp
        }

    # Append en formato JSON para facilitar parsing después
    file_path = PATH_LOG_GLOBAL + "registro_global.txt"
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(registro) + "\n")

def creacion_df_small(tabla:str="df_completo")->pd.DataFrame:
    sql=f"SELECT * FROM {tabla} LIMIT 1"
    conn = duckdb.connect(PATH_DATA_BASE_DB)
    df=conn.execute(sql).df()
    conn.close()
    return df