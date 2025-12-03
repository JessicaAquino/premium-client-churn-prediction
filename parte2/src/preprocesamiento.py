#preprocesamiento.py
import pandas as pd
import polars as pl
import numpy as np
from sklearn.impute import SimpleImputer
from typing import Tuple
from src.config import SEMILLA, PATH_DATA_BASE_DB , FILE_INPUT_DATA_PARQUET, GCP_PATH
import logging
import duckdb
import os
from src.config import SUBSAMPLEO
logger = logging.getLogger(__name__)
# columnas problemáticas (las que te tiró LightGBM)
ERR_COLS = (
    ["tmobile_app", "cmobile_app_trx",
     "cmobile_app_trx_max", "cmobile_app_trx_min",
     "tmobile_app_max", "tmobile_app_min","ctrx_quarter_normalizado"]
    + [f"cmobile_app_trx_lag_{i}" for i in range(1, 5)]
    + [f"tmobile_app_lag_{i}" for i in range(1, 5)]
)

def _existencia_tabla_duckdb()->bool:
    logger.info("Comienzo de Comprobando la existencia de la tabla df_init")
    sql = """
            SELECT EXISTS(
            SELECT 1 
            FROM information_schema.tables 
            WHERE table_name = 'df_init')"""
    conn=duckdb.connect(PATH_DATA_BASE_DB)
    existe=conn.execute(sql).fetchone()[0]
    conn.close()
    logger.info(f"La tabla existe ? : {existe}")
    logger.info("FIN de Comprobando la existencia de la tabla df_init")
    return existe
def _existencia_parquet()->bool:
    logger.info(f"Comienzo de la verificacion de la existencia del parquet en {FILE_INPUT_DATA_PARQUET}")
    existe=os.path.exists(FILE_INPUT_DATA_PARQUET)
    logger.info(f"El parquet existe ?: {existe} ")
    logger.info(f"Fin de la verificacion de la existencia del parquet en {FILE_INPUT_DATA_PARQUET}")
    return existe


def _create_table_de_parquet():
    logger.info("Comienzo de la creacion de la tabla a partir del parquet")
    sql = f"""create or replace table df_init as 
                select * from read_parquet('{FILE_INPUT_DATA_PARQUET}')"""
    conn=duckdb.connect(PATH_DATA_BASE_DB, read_only=False)
    conn.execute(sql)
    conn.close()
    logger.info("Fin de la creacion de la tabla a partir del parquet")

def verificacion_o_creacion_tabla():
    logger.info("Comienzo del proceso inicial de verif o creacion de df_init")
    if _existencia_tabla_duckdb():
        return
    elif _existencia_parquet():
        _create_table_de_parquet()
    else:
        logger.info(f"parquet NO existe en {FILE_INPUT_DATA_PARQUET} pero si en el bucket")
        logger.info(f"Ejecutar a mano : gsutil cp gs://{GCP_PATH}/datasets/competencia_02_final.parquet /datasets/ y volver el experimento")
        raise


# def split_train_test_apred(n_exp:int|str,mes_train:list[int],mes_test:int|list[int]
#                            ,mes_apred:int,semilla:int=SEMILLA,
#                            subsampleo:float=SUBSAMPLEO , feature_subset= None,n_canaritos:int=None)->Tuple[pd.DataFrame,
#                                                                np.ndarray,np.ndarray,np.ndarray, 
#                                                                np.ndarray, pd.DataFrame, 
#                                                                np.ndarray,np.ndarray,np.ndarray, 
#                                                                np.ndarray,pd.DataFrame,
#                                                                pd.DataFrame]:
#     logger.info("Comienzo del slpiteo de TRAIN - TEST - APRED")

        
#     sql_canaritos =''
#     if n_canaritos is not None and n_canaritos>0 :
#         for c in range(1,n_canaritos+1):
#             sql_canaritos += f'RANDOM() as canarito_{c}, '

#     exclude=''
#     if feature_subset is not None:
#         for i,f in enumerate(feature_subset):
#             if i ==0:
#                 exclude+=f'EXCLUDE({f}'
#             else:
#                 exclude+=f',{f}'
#         exclude+=')'

#     mes_train_sql = f"{mes_train[0]}"
#     for m in mes_train[1:]:    
#         mes_train_sql += f",{m}"
#     sql_train=f"""select {sql_canaritos} * {exclude} 
#                 from df_init
#                 where foto_mes IN ({mes_train_sql})"""
#     logger.info(f"sql train query : {sql_train}")
#     if isinstance(mes_test,list):
#         mes_test_sql = f"{mes_test[0]}"
#         for m in mes_test[1:]:    
#             mes_test_sql += f",{m}"
#         sql_test=f"""select {sql_canaritos} * {exclude}
#                     from df_init
#                     where foto_mes IN ({mes_test_sql})"""
#     elif isinstance(mes_test,int):
#         mes_test_sql = f"{mes_test}"
#         sql_test=f"""select {sql_canaritos} * {exclude}
#                     from df_init
#                     where foto_mes = {mes_test_sql}"""
#     logger.info(f"sql test query : {sql_test}")
        
#     mes_apred_sql = f"{mes_apred}"
#     sql_apred=f"""select {sql_canaritos} * {exclude}
#                 from df_init
#                 where foto_mes = {mes_apred_sql}"""
#     logger.info(f"sql apred query : {sql_apred}")
    
#     conn=duckdb.connect(PATH_DATA_BASE_DB)
#     seed_float = (semilla % 10000) / 10000.0
#     conn.execute("SELECT setseed(?)", [seed_float])

#     logger.info("Comienzo de la transformacion a polars de train data")
#     train_data = conn.execute(sql_train).pl()
#     logger.info("Comienzo de la transformacion a polars de test data")
#     test_data = conn.execute(sql_test).pl()
#     logger.info("Comienzo de la transformacion a polars de apred data")
#     apred_data = conn.execute(sql_apred).pl()
#     conn.close()

#     logger.info("Conversion polars a pandas de train")
#     train_data = train_data.to_pandas()
#     logger.info("Conversion polars a pandas de test")

#     test_data = test_data.to_pandas()
#     logger.info("Conversion polars a pandas de apred")

#     apred_data = apred_data.to_pandas()


#     if subsampleo is not None:
#         train_data=undersampling(train_data , subsampleo,semilla)
#     logger.info(f"Terminada la carga de df con columnas: {train_data.columns}")
#     # TRAIN
#     X_train = train_data.drop(['clase_ternaria', 'clase_weight', 'clase_binary','clase_binary2'], axis=1)
#     y_train_binaria = train_data['clase_binary'].to_numpy()
#     y_train_binaria_2 = train_data['clase_binary2'].to_numpy()
#     y_train_class=train_data["clase_ternaria"].to_numpy()
#     w_train = train_data['clase_weight'].to_numpy()

#     # TEST
#     X_test = test_data.drop(['clase_ternaria', 'clase_weight','clase_binary','clase_binary2'], axis=1)
#     y_test_binaria = test_data['clase_binary'].to_numpy()
#     y_test_binaria_2 = test_data['clase_binary2'].to_numpy()
#     y_test_class = test_data['clase_ternaria'].to_numpy()
#     w_test = test_data['clase_weight'].to_numpy()


#     # A PREDECIR
#     X_apred = apred_data.drop(['clase_ternaria', 'clase_weight','clase_binary','clase_binary2'], axis=1)
#     y_apred=X_apred[["numero_de_cliente"]] # DF
  

#     logger.info(f"X_train shape : {X_train.shape} / y_train shape : {y_train_binaria.shape} de los meses : {X_train['foto_mes'].unique()}")
#     logger.info(f"X_test shape : {X_test.shape} / y_test shape : {y_test_binaria.shape}  del mes : {X_test['foto_mes'].unique()}")
#     logger.info(f"X_apred shape : {X_apred.shape} / y_apred shape : {y_apred.shape}  del mes : {X_apred['foto_mes'].unique()}")

#     logger.info(f"cantidad de baja y continua en train:{np.unique(y_train_binaria,return_counts=True)}")
#     logger.info(f"cantidad de baja y continua en test:{np.unique(y_test_binaria,return_counts=True)}")
#     logger.info("Finalizacion label binario")
#     # ÚSALO justo antes de entrenar:
#     X_train = coerce_numeric_cols(X_train, ERR_COLS, fillna_val=0.0)
#     X_test  = coerce_numeric_cols(X_test,  ERR_COLS, fillna_val=0.0)
#     X_apred = coerce_numeric_cols(X_apred, ERR_COLS, fillna_val=0.0)
#     return X_train, y_train_binaria,y_train_binaria_2,y_train_class, w_train, X_test, y_test_binaria,y_test_binaria_2, y_test_class, w_test ,X_apred , y_apred 





def split_train_test_apred(n_exp:int|str,mes_train:list[int],mes_test:int|list[int]
                           ,mes_apred:int,semilla:int=SEMILLA,
                           subsampleo:float=SUBSAMPLEO , feature_subset= None,n_canaritos:int=None)->Tuple[pd.DataFrame,
                                                               np.ndarray,np.ndarray,np.ndarray, 
                                                               np.ndarray, pd.DataFrame, 
                                                               np.ndarray,np.ndarray,np.ndarray, 
                                                               np.ndarray,pd.DataFrame,
                                                               pd.DataFrame]:
    logger.info(f"Comienzo del slpiteo de TRAIN - TEST - APRED con subsampleo : {subsampleo}")

        
    sql_canaritos =''
    if n_canaritos is not None and n_canaritos>0 :
        for c in range(1,n_canaritos+1):
            sql_canaritos += f'RANDOM() as canarito_{c}, '

    exclude=''
    if feature_subset is not None:
        for i,f in enumerate(feature_subset):
            if i ==0:
                exclude+=f'EXCLUDE({f}'
            else:
                exclude+=f',{f}'
        exclude+=')'

    conn=duckdb.connect(PATH_DATA_BASE_DB)
    seed_float = (semilla % 10000) / 10000.0
    conn.execute("SELECT setseed(?)", [seed_float])
    

    mes_train_sql = f"{mes_train[0]}"
    for m in mes_train[1:]:    
        mes_train_sql += f",{m}"


    sql_train = f"""
        WITH continuas AS (
            SELECT DISTINCT numero_de_cliente
            FROM df_init
            WHERE foto_mes IN ({mes_train_sql})
            AND clase_ternaria = 'Continua'
        ),
        continuas_sample AS (
            SELECT numero_de_cliente
            FROM continuas
            ORDER BY RANDOM()
            LIMIT (
                SELECT CAST(COUNT(*) * {subsampleo} AS INTEGER)
                FROM continuas
            )
        )
    """


    sql_train += f"""SELECT {sql_canaritos} * {exclude} 
                    FROM df_init 
                    WHERE (foto_mes IN ({mes_train_sql}) AND clase_ternaria != 'Continua' ) OR 
                    (foto_mes IN ({mes_train_sql}) AND clase_ternaria = 'Continua' AND numero_de_cliente IN
                                        (SELECT numero_de_cliente 
                                            FROM continuas_sample )
                                        )"""
    logger.info(f"sql train query : {sql_train}")

    if isinstance(mes_test,list):
        mes_test_sql = f"{mes_test[0]}"
        for m in mes_test[1:]:    
            mes_test_sql += f",{m}"
        sql_test=f"""select {sql_canaritos} * {exclude}
                    from df_init
                    where foto_mes IN ({mes_test_sql})"""
    elif isinstance(mes_test,int):
        mes_test_sql = f"{mes_test}"
        sql_test=f"""select {sql_canaritos} * {exclude}
                    from df_init
                    where foto_mes = {mes_test_sql}"""
    logger.info(f"sql test query : {sql_test}")
        
    mes_apred_sql = f"{mes_apred}"
    sql_apred=f"""select {sql_canaritos} * {exclude}
                from df_init
                where foto_mes = {mes_apred_sql}"""
    logger.info(f"sql apred query : {sql_apred}")

   
    logger.info("Comienzo de la transformacion a polars de train data")
    train_data = conn.execute(sql_train).pl()
    logger.info("Comienzo de la transformacion a polars de test data")
    test_data = conn.execute(sql_test).pl()
    logger.info("Comienzo de la transformacion a polars de apred data")
    apred_data = conn.execute(sql_apred).pl()
    conn.close()

    logger.info("Conversion polars a pandas de train")
    train_data = train_data.to_pandas()
    logger.info("Conversion polars a pandas de test")

    test_data = test_data.to_pandas()
    logger.info("Conversion polars a pandas de apred")

    apred_data = apred_data.to_pandas()

   
    logger.info(f"Terminada la carga de df con columnas: {train_data.columns}")
    # TRAIN
    X_train = train_data.drop(['clase_ternaria', 'clase_weight', 'clase_binary','clase_binary2'], axis=1)
    y_train_binaria = train_data['clase_binary'].to_numpy()
    y_train_binaria_2 = train_data['clase_binary2'].to_numpy()
    y_train_class=train_data["clase_ternaria"].to_numpy()
    w_train = train_data['clase_weight'].to_numpy()

    # TEST
    X_test = test_data.drop(['clase_ternaria', 'clase_weight','clase_binary','clase_binary2'], axis=1)
    y_test_binaria = test_data['clase_binary'].to_numpy()
    y_test_binaria_2 = test_data['clase_binary2'].to_numpy()
    y_test_class = test_data['clase_ternaria'].to_numpy()
    w_test = test_data['clase_weight'].to_numpy()


    # A PREDECIR
    X_apred = apred_data.drop(['clase_ternaria', 'clase_weight','clase_binary','clase_binary2'], axis=1)
    y_apred=X_apred[["numero_de_cliente"]] # DF
  

    logger.info(f"X_train shape : {X_train.shape} / y_train shape : {y_train_binaria.shape} de los meses : {X_train['foto_mes'].unique()}")
    logger.info(f"X_test shape : {X_test.shape} / y_test shape : {y_test_binaria.shape}  del mes : {X_test['foto_mes'].unique()}")
    logger.info(f"X_apred shape : {X_apred.shape} / y_apred shape : {y_apred.shape}  del mes : {X_apred['foto_mes'].unique()}")

    logger.info(f"cantidad de baja y continua en train:{np.unique(y_train_binaria,return_counts=True)}")
    logger.info(f"cantidad de baja y continua en test:{np.unique(y_test_binaria,return_counts=True)}")
    logger.info("Finalizacion label binario")
    # ÚSALO justo antes de entrenar:
    X_train = coerce_numeric_cols(X_train, ERR_COLS, fillna_val=0.0)
    X_test  = coerce_numeric_cols(X_test,  ERR_COLS, fillna_val=0.0)
    X_apred = coerce_numeric_cols(X_apred, ERR_COLS, fillna_val=0.0)
    return X_train, y_train_binaria,y_train_binaria_2,y_train_class, w_train, X_test, y_test_binaria,y_test_binaria_2, y_test_class, w_test ,X_apred , y_apred 







# def split_train_test_apred(n_exp:int|str,mes_train:list[int],mes_test:int|list[int]
#                            ,mes_apred:int,semilla:int=SEMILLA,
#                            subsampleo:float=SUBSAMPLEO , feature_subset= None,n_canaritos:int=None)->Tuple[pd.DataFrame,
#                                                                np.ndarray,np.ndarray,np.ndarray, 
#                                                                np.ndarray, pd.DataFrame, 
#                                                                np.ndarray,np.ndarray,np.ndarray, 
#                                                                np.ndarray,pd.DataFrame,
#                                                                pd.DataFrame]:
#     logger.info("Comienzo del slpiteo de TRAIN - TEST - APRED")

        
#     sql_canaritos =''
#     if n_canaritos is not None and n_canaritos>0 :
#         for c in range(1,n_canaritos+1):
#             sql_canaritos += f'RANDOM() as canarito_{c}, '

#     exclude=''
#     if feature_subset is not None:
#         for i,f in enumerate(feature_subset):
#             if i ==0:
#                 exclude+=f'EXCLUDE({f}'
#             else:
#                 exclude+=f',{f}'
#         exclude+=')'

#     conn=duckdb.connect(PATH_DATA_BASE_DB)
#     seed_float = (semilla % 10000) / 10000.0
#     conn.execute("SELECT setseed(?)", [seed_float])
    

#     mes_train_sql = f"{mes_train[0]}"
#     for m in mes_train[1:]:    
#         mes_train_sql += f",{m}"

#     sql_continuas_sample = f"""
#     CREATE TEMP TABLE continuas_sample AS
#     SELECT DISTINCT numero_de_cliente
#     FROM df_init
#     WHERE foto_mes IN ({mes_train_sql})
#     AND clase_ternaria = 'Continua'
#     AND RANDOM() < {subsampleo}
#     """
#     logger.info("Creando tabla temporal de subsampleo...")
#     conn.execute(sql_continuas_sample)
#     logger.info("Fin de la creaion de la tabla temporal de subsampleo")


#     sql_train = f"""SELECT {sql_canaritos} * {exclude} 
#                     FROM df_init 
#                     WHERE (foto_mes IN ({mes_train_sql}) AND clase_ternaria != 'Continua' ) OR 
#                     (foto_mes IN ({mes_train_sql}) AND clase_ternaria = 'Continua' AND numero_de_cliente IN
#                                         (SELECT numero_de_cliente 
#                                             FROM continuas_sample )
#                                         )"""
#     logger.info(f"sql train query : {sql_train}")

#     if isinstance(mes_test,list):
#         mes_test_sql = f"{mes_test[0]}"
#         for m in mes_test[1:]:    
#             mes_test_sql += f",{m}"
#         sql_test=f"""select {sql_canaritos} * {exclude}
#                     from df_init
#                     where foto_mes IN ({mes_test_sql})"""
#     elif isinstance(mes_test,int):
#         mes_test_sql = f"{mes_test}"
#         sql_test=f"""select {sql_canaritos} * {exclude}
#                     from df_init
#                     where foto_mes = {mes_test_sql}"""
#     logger.info(f"sql test query : {sql_test}")
        
#     mes_apred_sql = f"{mes_apred}"
#     sql_apred=f"""select {sql_canaritos} * {exclude}
#                 from df_init
#                 where foto_mes = {mes_apred_sql}"""
#     logger.info(f"sql apred query : {sql_apred}")

   
#     logger.info("Comienzo de la transformacion a polars de train data")
#     train_data = conn.execute(sql_train).pl()
#     logger.info("Comienzo de la transformacion a polars de test data")
#     test_data = conn.execute(sql_test).pl()
#     logger.info("Comienzo de la transformacion a polars de apred data")
#     apred_data = conn.execute(sql_apred).pl()
#     conn.close()

#     logger.info("Conversion polars a pandas de train")
#     train_data = train_data.to_pandas()
#     logger.info("Conversion polars a pandas de test")

#     test_data = test_data.to_pandas()
#     logger.info("Conversion polars a pandas de apred")

#     apred_data = apred_data.to_pandas()

   
#     logger.info(f"Terminada la carga de df con columnas: {train_data.columns}")
#     # TRAIN
#     X_train = train_data.drop(['clase_ternaria', 'clase_weight', 'clase_binary','clase_binary2'], axis=1)
#     y_train_binaria = train_data['clase_binary'].to_numpy()
#     y_train_binaria_2 = train_data['clase_binary2'].to_numpy()
#     y_train_class=train_data["clase_ternaria"].to_numpy()
#     w_train = train_data['clase_weight'].to_numpy()

#     # TEST
#     X_test = test_data.drop(['clase_ternaria', 'clase_weight','clase_binary','clase_binary2'], axis=1)
#     y_test_binaria = test_data['clase_binary'].to_numpy()
#     y_test_binaria_2 = test_data['clase_binary2'].to_numpy()
#     y_test_class = test_data['clase_ternaria'].to_numpy()
#     w_test = test_data['clase_weight'].to_numpy()


#     # A PREDECIR
#     X_apred = apred_data.drop(['clase_ternaria', 'clase_weight','clase_binary','clase_binary2'], axis=1)
#     y_apred=X_apred[["numero_de_cliente"]] # DF
  

#     logger.info(f"X_train shape : {X_train.shape} / y_train shape : {y_train_binaria.shape} de los meses : {X_train['foto_mes'].unique()}")
#     logger.info(f"X_test shape : {X_test.shape} / y_test shape : {y_test_binaria.shape}  del mes : {X_test['foto_mes'].unique()}")
#     logger.info(f"X_apred shape : {X_apred.shape} / y_apred shape : {y_apred.shape}  del mes : {X_apred['foto_mes'].unique()}")

#     logger.info(f"cantidad de baja y continua en train:{np.unique(y_train_binaria,return_counts=True)}")
#     logger.info(f"cantidad de baja y continua en test:{np.unique(y_test_binaria,return_counts=True)}")
#     logger.info("Finalizacion label binario")
#     # ÚSALO justo antes de entrenar:
#     X_train = coerce_numeric_cols(X_train, ERR_COLS, fillna_val=0.0)
#     X_test  = coerce_numeric_cols(X_test,  ERR_COLS, fillna_val=0.0)
#     X_apred = coerce_numeric_cols(X_apred, ERR_COLS, fillna_val=0.0)
#     return X_train, y_train_binaria,y_train_binaria_2,y_train_class, w_train, X_test, y_test_binaria,y_test_binaria_2, y_test_class, w_test ,X_apred , y_apred 




# def split_train_test_apred(n_exp:int|str,mes_train:list[int],mes_test:int|list[int]
#                            ,mes_apred:int,semilla:int=SEMILLA,
#                            subsampleo:float=SUBSAMPLEO , feature_subset= None,n_canaritos:int=None)->Tuple[pd.DataFrame,
#                                                                np.ndarray,np.ndarray,np.ndarray, 
#                                                                np.ndarray, pd.DataFrame, 
#                                                                np.ndarray,np.ndarray,np.ndarray, 
#                                                                np.ndarray,pd.DataFrame,
#                                                                pd.DataFrame]:
#     logger.info("Comienzo del slpiteo de TRAIN - TEST - APRED")

        
#     sql_canaritos =''
#     if n_canaritos is not None and n_canaritos>0 :
#         for c in range(1,n_canaritos+1):
#             sql_canaritos += f'RANDOM() as canarito_{c}, '

#     exclude=''
#     if feature_subset is not None:
#         for i,f in enumerate(feature_subset):
#             if i ==0:
#                 exclude+=f'EXCLUDE({f}'
#             else:
#                 exclude+=f',{f}'
#         exclude+=')'

#     mes_train_sql = f"{mes_train[0]}"
#     for m in mes_train[1:]:    
#         mes_train_sql += f",{m}"


#     if isinstance(mes_test,list):
#         mes_test_sql = f"{mes_test[0]}"
#         for m in mes_test[1:]:    
#             mes_test_sql += f",{m}"
#     elif isinstance(mes_test,int):
#         mes_test_sql = f"{mes_test}"
        
#     mes_apred_sql = f"{mes_apred}"




#     sql_continuas = f"""
#     CREATE TEMP TABLE continuas_sample AS
#     SELECT numeros_unicos AS numero_de_cliente
#     FROM (
#         SELECT DISTINCT numero_de_cliente AS numeros_unicos
#         FROM df_init
#         WHERE foto_mes IN ({mes_train_sql})
#         AND clase_ternaria = 'Continua'
#     )
#     WHERE RANDOM() < {subsampleo}
#     """
#     logger.info("Comienzo de la ejecucion de numeros_unicos")

#     conn = duckdb.connect(PATH_DATA_BASE_DB)
#     seed_duck = (semilla % 2_000_000) / 1_000_000.0 - 1.0
#     logger.info(f"Semilla entera: {semilla} -> semilla duckdb: {seed_duck}")
#     conn.execute("SELECT setseed(?);", [seed_duck])
#     conn.execute(sql_continuas)
#     logger.info("Fin de la ejecucion de numeros_unicos")



#     sql_completo = f"""
#     SELECT {sql_canaritos} * {exclude},
#         CASE
#             WHEN (
#                     foto_mes IN ({mes_train_sql})
#                     AND clase_ternaria != 'Continua'
#             )
#             OR (
#                     foto_mes IN ({mes_train_sql})
#                     AND clase_ternaria = 'Continua'
#                     AND numero_de_cliente IN (SELECT numero_de_cliente FROM continuas_sample)
#             )
#             THEN 'train'
#             WHEN foto_mes IN ({mes_test_sql}) THEN 'test'
#             WHEN foto_mes = {mes_apred_sql} THEN 'apred'
#         END AS spliteo
#     FROM df_init
#     WHERE foto_mes IN ({mes_train_sql}, {mes_test_sql}, {mes_apred_sql})
#     """


#     logger.info(f"sql completo query : {sql_completo}")
#     logger.info("Comienzo de la transfor a pds")
#     data_completa = conn.execute(sql_completo).df()
#     conn.close()
#     logger.info("Fin de la transfor a pds")

#     train_data = data_completa[data_completa['spliteo'] == 'train'].drop(columns=['spliteo'])
#     test_data = data_completa[data_completa['spliteo'] == 'test'].drop(columns=['spliteo'])
#     apred_data = data_completa[data_completa['spliteo'] == 'apred'].drop(columns=['spliteo'])
    
#     # TRAIN
#     X_train = train_data.drop(['clase_ternaria', 'clase_weight', 'clase_binary','clase_binary2'], axis=1)
#     y_train_binaria = train_data['clase_binary'].to_numpy()
#     y_train_binaria_2 = train_data['clase_binary2'].to_numpy()
#     y_train_class=train_data["clase_ternaria"].to_numpy()
#     w_train = train_data['clase_weight'].to_numpy()

#     # TEST
#     X_test = test_data.drop(['clase_ternaria', 'clase_weight','clase_binary','clase_binary2'], axis=1)
#     y_test_binaria = test_data['clase_binary'].to_numpy()
#     y_test_binaria_2 = test_data['clase_binary2'].to_numpy()
#     y_test_class = test_data['clase_ternaria'].to_numpy()
#     w_test = test_data['clase_weight'].to_numpy()


#     # A PREDECIR
#     X_apred = apred_data.drop(['clase_ternaria', 'clase_weight','clase_binary','clase_binary2'], axis=1)
#     y_apred=X_apred[["numero_de_cliente"]] # DF
  

#     logger.info(f"X_train shape : {X_train.shape} / y_train shape : {y_train_binaria.shape} de los meses : {X_train['foto_mes'].unique()}")
#     logger.info(f"X_test shape : {X_test.shape} / y_test shape : {y_test_binaria.shape}  del mes : {X_test['foto_mes'].unique()}")
#     logger.info(f"X_apred shape : {X_apred.shape} / y_apred shape : {y_apred.shape}  del mes : {X_apred['foto_mes'].unique()}")

#     logger.info(f"cantidad de baja y continua en train:{np.unique(y_train_binaria,return_counts=True)}")
#     logger.info(f"cantidad de baja y continua en test:{np.unique(y_test_binaria,return_counts=True)}")
#     logger.info("Finalizacion label binario")
#     # ÚSALO justo antes de entrenar:
#     X_train = coerce_numeric_cols(X_train, ERR_COLS, fillna_val=0.0)
#     X_test  = coerce_numeric_cols(X_test,  ERR_COLS, fillna_val=0.0)
#     X_apred = coerce_numeric_cols(X_apred, ERR_COLS, fillna_val=0.0)
#     return X_train, y_train_binaria,y_train_binaria_2,y_train_class, w_train, X_test, y_test_binaria,y_test_binaria_2, y_test_class, w_test ,X_apred , y_apred 


# def split_train_test_apred(n_exp:int|str,mes_train:list[int],mes_test:int|list[int]
#                            ,mes_apred:int,semilla:int=SEMILLA,
#                            subsampleo:float=SUBSAMPLEO , feature_subset= None,n_canaritos:int=None)->Tuple[pd.DataFrame,
#                                                                np.ndarray,np.ndarray,np.ndarray, 
#                                                                np.ndarray, pd.DataFrame, 
#                                                                np.ndarray,np.ndarray,np.ndarray, 
#                                                                np.ndarray,pd.DataFrame,
#                                                                pd.DataFrame]:
#     logger.info("Comienzo del slpiteo de TRAIN - TEST - APRED")

        
#     sql_canaritos =''
#     if n_canaritos is not None and n_canaritos>0 :
#         for c in range(1,n_canaritos+1):
#             sql_canaritos += f'RANDOM() as canarito_{c}, '

#     exclude=''
#     if feature_subset is not None:
#         for i,f in enumerate(feature_subset):
#             if i ==0:
#                 exclude+=f'EXCLUDE({f}'
#             else:
#                 exclude+=f',{f}'
#         exclude+=')'

#     mes_train_sql = f"{mes_train[0]}"
#     for m in mes_train[1:]:    
#         mes_train_sql += f",{m}"


#     if isinstance(mes_test,list):
#         mes_test_sql = f"{mes_test[0]}"
#         for m in mes_test[1:]:    
#             mes_test_sql += f",{m}"
#     elif isinstance(mes_test,int):
#         mes_test_sql = f"{mes_test}"
        
#     mes_apred_sql = f"{mes_apred}"

        
#     sql_completo = f"""with continuas_train as 
#     (SELECT DISTINCT(numero_de_cliente) as numeros_unicos,
#     CASE WHEN RANDOM() < {subsampleo} THEN 1 ELSE 0 END AS flag_subsampleo
#     FROM df_init 
#     WHERE foto_mes in ({mes_train_sql}) AND clase_ternaria = 'Continua')
#     """


#     sql_completo += f"""SELECT {sql_canaritos} * {exclude} ,
#                     CASE
#                         WHEN( 
#                             (foto_mes IN ({mes_train_sql}) AND clase_ternaria != 'Continua' )
#                             OR
#                             (foto_mes IN ({mes_train_sql}) AND clase_ternaria = 'Continua' AND numero_de_cliente IN
#                                         (SELECT numeros_unicos 
#                                             FROM continuas_train 
#                                             WHERE flag_subsampleo =1)
#                                         )
#                             )THEN 'train' 
#                         WHEN foto_mes IN ({mes_test_sql}) THEN 'test' 
#                         WHEN foto_mes = {mes_apred_sql} THEN 'apred'
#                     END AS spliteo
#                     FROM df_init 
#                     WHERE foto_mes IN ({mes_train_sql}, {mes_test_sql},{mes_apred_sql})"""

#     logger.info(f"sql completo query : {sql_completo}")
#     logger.info("Comienzo de la transfor a pds")
#     conn=duckdb.connect(PATH_DATA_BASE_DB)
#     conn.execute(f"SET seed = {semilla};")
#     data_completa = conn.execute(sql_completo).df()
#     conn.close()
#     logger.info("Fin de la transfor a pds")

#     train_data = data_completa[data_completa['spliteo'] == 'train'].drop(columns=['spliteo'])
#     test_data = data_completa[data_completa['spliteo'] == 'test'].drop(columns=['spliteo'])
#     apred_data = data_completa[data_completa['spliteo'] == 'apred'].drop(columns=['spliteo'])
    
#     # TRAIN
#     X_train = train_data.drop(['clase_ternaria', 'clase_weight', 'clase_binary','clase_binary2'], axis=1)
#     y_train_binaria = train_data['clase_binary'].to_numpy()
#     y_train_binaria_2 = train_data['clase_binary2'].to_numpy()
#     y_train_class=train_data["clase_ternaria"].to_numpy()
#     w_train = train_data['clase_weight'].to_numpy()

#     # TEST
#     X_test = test_data.drop(['clase_ternaria', 'clase_weight','clase_binary','clase_binary2'], axis=1)
#     y_test_binaria = test_data['clase_binary'].to_numpy()
#     y_test_binaria_2 = test_data['clase_binary2'].to_numpy()
#     y_test_class = test_data['clase_ternaria'].to_numpy()
#     w_test = test_data['clase_weight'].to_numpy()


#     # A PREDECIR
#     X_apred = apred_data.drop(['clase_ternaria', 'clase_weight','clase_binary','clase_binary2'], axis=1)
#     y_apred=X_apred[["numero_de_cliente"]] # DF
  

#     logger.info(f"X_train shape : {X_train.shape} / y_train shape : {y_train_binaria.shape} de los meses : {X_train['foto_mes'].unique()}")
#     logger.info(f"X_test shape : {X_test.shape} / y_test shape : {y_test_binaria.shape}  del mes : {X_test['foto_mes'].unique()}")
#     logger.info(f"X_apred shape : {X_apred.shape} / y_apred shape : {y_apred.shape}  del mes : {X_apred['foto_mes'].unique()}")

#     logger.info(f"cantidad de baja y continua en train:{np.unique(y_train_binaria,return_counts=True)}")
#     logger.info(f"cantidad de baja y continua en test:{np.unique(y_test_binaria,return_counts=True)}")
#     logger.info("Finalizacion label binario")
#     # ÚSALO justo antes de entrenar:
#     X_train = coerce_numeric_cols(X_train, ERR_COLS, fillna_val=0.0)
#     X_test  = coerce_numeric_cols(X_test,  ERR_COLS, fillna_val=0.0)
#     X_apred = coerce_numeric_cols(X_apred, ERR_COLS, fillna_val=0.0)
#     return X_train, y_train_binaria,y_train_binaria_2,y_train_class, w_train, X_test, y_test_binaria,y_test_binaria_2, y_test_class, w_test ,X_apred , y_apred 



# def split_train_test_apred(n_exp:int|str,mes_train:list[int],mes_test:int|list[int]
#                            ,mes_apred:int,semilla:int=SEMILLA,
#                            subsampleo:float=SUBSAMPLEO , feature_subset= None,n_canaritos:int=None)->Tuple[pd.DataFrame,
#                                                                np.ndarray,np.ndarray,np.ndarray, 
#                                                                np.ndarray, pd.DataFrame, 
#                                                                np.ndarray,np.ndarray,np.ndarray, 
#                                                                np.ndarray,pd.DataFrame,
#                                                                pd.DataFrame]:
#     logger.info("Comienzo del slpiteo de TRAIN - TEST - APRED")

        
#     sql_canaritos =''
#     if n_canaritos is not None and n_canaritos>0 :
#         for c in range(1,n_canaritos+1):
#             sql_canaritos += f'RANDOM() as canarito_{c}, '

#     exclude=''
#     if feature_subset is not None:
#         for i,f in enumerate(feature_subset):
#             if i ==0:
#                 exclude+=f'EXCLUDE({f}'
#             else:
#                 exclude+=f',{f}'
#         exclude+=')'

#     mes_train_sql = f"{mes_train[0]}"
#     for m in mes_train[1:]:    
#         mes_train_sql += f",{m}"


#     if isinstance(mes_test,list):
#         mes_test_sql = f"{mes_test[0]}"
#         for m in mes_test[1:]:    
#             mes_test_sql += f",{m}"
#     elif isinstance(mes_test,int):
#         mes_test_sql = f"{mes_test}"
        
#     mes_apred_sql = f"{mes_apred}"
    
#     sql_completo = f"""SELECT {sql_canaritos} * {exclude} ,
#                         CASE
#                             WHEN foto_mes IN ({mes_train_sql}) THEN 'train' 
#                             WHEN foto_mes IN ({mes_test_sql}) THEN 'test' 
#                             WHEN foto_mes = {mes_apred_sql} THEN 'apred'
#                         END AS spliteo
#                         FROM df_init
#                         WHERE foto_mes IN ({mes_train_sql}, {mes_test_sql},{mes_apred_sql})"""
#     logger.info(f"sql apred query : {sql_completo}")

#     logger.info("Comienzo de la transfor a pds")
#     conn=duckdb.connect(PATH_DATA_BASE_DB)
#     conn.execute(f"SET seed = {semilla};")
#     data_completa = conn.execute(sql_completo).df()
#     conn.close()
#     logger.info("Fin de la transfor a pds")

#     train_data = data_completa[data_completa['spliteo'] == 'train'].drop(columns=['spliteo'])
#     test_data = data_completa[data_completa['spliteo'] == 'test'].drop(columns=['spliteo'])
#     apred_data = data_completa[data_completa['spliteo'] == 'apred'].drop(columns=['spliteo'])
    
#     if subsampleo is not None:
#         train_data=undersampling(train_data , subsampleo,semilla)
#     logger.info(f"Terminada la carga de df con columnas: {train_data.columns}")
#     # TRAIN
#     X_train = train_data.drop(['clase_ternaria', 'clase_weight', 'clase_binary','clase_binary2'], axis=1)
#     y_train_binaria = train_data['clase_binary'].to_numpy()
#     y_train_binaria_2 = train_data['clase_binary2'].to_numpy()
#     y_train_class=train_data["clase_ternaria"].to_numpy()
#     w_train = train_data['clase_weight'].to_numpy()

#     # TEST
#     X_test = test_data.drop(['clase_ternaria', 'clase_weight','clase_binary','clase_binary2'], axis=1)
#     y_test_binaria = test_data['clase_binary'].to_numpy()
#     y_test_binaria_2 = test_data['clase_binary2'].to_numpy()
#     y_test_class = test_data['clase_ternaria'].to_numpy()
#     w_test = test_data['clase_weight'].to_numpy()


#     # A PREDECIR
#     X_apred = apred_data.drop(['clase_ternaria', 'clase_weight','clase_binary','clase_binary2'], axis=1)
#     y_apred=X_apred[["numero_de_cliente"]] # DF
  

#     logger.info(f"X_train shape : {X_train.shape} / y_train shape : {y_train_binaria.shape} de los meses : {X_train['foto_mes'].unique()}")
#     logger.info(f"X_test shape : {X_test.shape} / y_test shape : {y_test_binaria.shape}  del mes : {X_test['foto_mes'].unique()}")
#     logger.info(f"X_apred shape : {X_apred.shape} / y_apred shape : {y_apred.shape}  del mes : {X_apred['foto_mes'].unique()}")

#     logger.info(f"cantidad de baja y continua en train:{np.unique(y_train_binaria,return_counts=True)}")
#     logger.info(f"cantidad de baja y continua en test:{np.unique(y_test_binaria,return_counts=True)}")
#     logger.info("Finalizacion label binario")
#     # ÚSALO justo antes de entrenar:
#     X_train = coerce_numeric_cols(X_train, ERR_COLS, fillna_val=0.0)
#     X_test  = coerce_numeric_cols(X_test,  ERR_COLS, fillna_val=0.0)
#     X_apred = coerce_numeric_cols(X_apred, ERR_COLS, fillna_val=0.0)
#     return X_train, y_train_binaria,y_train_binaria_2,y_train_class, w_train, X_test, y_test_binaria,y_test_binaria_2, y_test_class, w_test ,X_apred , y_apred 







# def split_train_test_apred(n_exp:int|str,mes_train:list[int],mes_test:int|list[int]
#                            ,mes_apred:int,semilla:int=SEMILLA,
#                            subsampleo:float=SUBSAMPLEO , feature_subset= None,n_canaritos:int=None)->Tuple[pd.DataFrame,
#                                                                np.ndarray,np.ndarray,np.ndarray, 
#                                                                np.ndarray, pd.DataFrame, 
#                                                                np.ndarray,np.ndarray,np.ndarray, 
#                                                                np.ndarray,pd.DataFrame,
#                                                                pd.DataFrame]:
#     logger.info("Comienzo del slpiteo de TRAIN - TEST - APRED")

        
#     sql_canaritos =''
#     if n_canaritos is not None and n_canaritos>0 :
#         for c in range(1,n_canaritos+1):
#             sql_canaritos += f'RANDOM() as canarito_{c}, '

#     exclude=''
#     if feature_subset is not None:
#         for i,f in enumerate(feature_subset):
#             if i ==0:
#                 exclude+=f'EXCLUDE({f}'
#             else:
#                 exclude+=f',{f}'
#         exclude+=')'

#     mes_train_sql = f"{mes_train[0]}"
#     for m in mes_train[1:]:    
#         mes_train_sql += f",{m}"
#     sql_train=f"""select {sql_canaritos} * {exclude} 
#                 from df_init
#                 where foto_mes IN ({mes_train_sql})"""
#     logger.info(f"sql train query : {sql_train}")
#     if isinstance(mes_test,list):
#         mes_test_sql = f"{mes_test[0]}"
#         for m in mes_test[1:]:    
#             mes_test_sql += f",{m}"
#         sql_test=f"""select {sql_canaritos} * {exclude}
#                     from df_init
#                     where foto_mes IN ({mes_test_sql})"""
#     elif isinstance(mes_test,int):
#         mes_test_sql = f"{mes_test}"
#         sql_test=f"""select {sql_canaritos} * {exclude}
#                     from df_init
#                     where foto_mes = {mes_test_sql}"""
#     logger.info(f"sql test query : {sql_test}")
        
#     mes_apred_sql = f"{mes_apred}"
#     sql_apred=f"""select {sql_canaritos} * {exclude}
#                 from df_init
#                 where foto_mes = {mes_apred_sql}"""
#     logger.info(f"sql apred query : {sql_apred}")
    
#     conn=duckdb.connect(PATH_DATA_BASE_DB)
#     seed_float = (semilla % 10000) / 10000.0
#     conn.execute("SELECT setseed(?)", [seed_float])
#     train_data = conn.execute(sql_train).df()
#     test_data = conn.execute(sql_test).df()
#     apred_data = conn.execute(sql_apred).df()
#     conn.close()
#     if subsampleo is not None:
#         train_data=undersampling(train_data , subsampleo,semilla)
#     logger.info(f"Terminada la carga de df con columnas: {train_data.columns}")
#     # TRAIN
#     X_train = train_data.drop(['clase_ternaria', 'clase_weight', 'clase_binary','clase_binary2'], axis=1)
#     y_train_binaria = train_data['clase_binary'].to_numpy()
#     y_train_binaria_2 = train_data['clase_binary2'].to_numpy()
#     y_train_class=train_data["clase_ternaria"].to_numpy()
#     w_train = train_data['clase_weight'].to_numpy()

#     # TEST
#     X_test = test_data.drop(['clase_ternaria', 'clase_weight','clase_binary','clase_binary2'], axis=1)
#     y_test_binaria = test_data['clase_binary'].to_numpy()
#     y_test_binaria_2 = test_data['clase_binary2'].to_numpy()
#     y_test_class = test_data['clase_ternaria'].to_numpy()
#     w_test = test_data['clase_weight'].to_numpy()


#     # A PREDECIR
#     X_apred = apred_data.drop(['clase_ternaria', 'clase_weight','clase_binary','clase_binary2'], axis=1)
#     y_apred=X_apred[["numero_de_cliente"]] # DF
  

#     logger.info(f"X_train shape : {X_train.shape} / y_train shape : {y_train_binaria.shape} de los meses : {X_train['foto_mes'].unique()}")
#     logger.info(f"X_test shape : {X_test.shape} / y_test shape : {y_test_binaria.shape}  del mes : {X_test['foto_mes'].unique()}")
#     logger.info(f"X_apred shape : {X_apred.shape} / y_apred shape : {y_apred.shape}  del mes : {X_apred['foto_mes'].unique()}")

#     logger.info(f"cantidad de baja y continua en train:{np.unique(y_train_binaria,return_counts=True)}")
#     logger.info(f"cantidad de baja y continua en test:{np.unique(y_test_binaria,return_counts=True)}")
#     logger.info("Finalizacion label binario")
#     # ÚSALO justo antes de entrenar:
#     X_train = coerce_numeric_cols(X_train, ERR_COLS, fillna_val=0.0)
#     X_test  = coerce_numeric_cols(X_test,  ERR_COLS, fillna_val=0.0)
#     X_apred = coerce_numeric_cols(X_apred, ERR_COLS, fillna_val=0.0)
#     return X_train, y_train_binaria,y_train_binaria_2,y_train_class, w_train, X_test, y_test_binaria,y_test_binaria_2, y_test_class, w_test ,X_apred , y_apred 









def undersampling(df:pd.DataFrame ,undersampling_rate:float , semilla:int) -> pd.DataFrame:
    logger.info("Comienzo del subsampleo")
    np.random.seed(semilla)
    clientes_minoritaria = df.loc[df["clase_ternaria"] != "Continua", "numero_de_cliente"].unique()
    clientes_mayoritaria = df.loc[df["clase_ternaria"] == "Continua", "numero_de_cliente"].unique()

    logger.info(f"Clientes minoritarios: {len(clientes_minoritaria)}")
    logger.info(f"Clientes mayoritarios: {len(clientes_mayoritaria)}")

    n_sample = int(len(clientes_mayoritaria) * undersampling_rate)
    clientes_mayoritaria_sample = np.random.choice(clientes_mayoritaria, n_sample, replace=False)

    # Unimos los IDs seleccionados
    clientes_finales = np.concatenate([clientes_minoritaria, clientes_mayoritaria_sample])

    df_train_undersampled = df[df["numero_de_cliente"].isin(clientes_finales)].copy()

    logger.info(f"Shape original: {df.shape}")
    logger.info(f"Shape undersampled: {df_train_undersampled.shape}")

    df_train_undersampled = df_train_undersampled.sample(frac=1, random_state=semilla).reset_index(drop=True)
    return df_train_undersampled



def coerce_numeric_cols(df: pd.DataFrame, cols: list[str], fillna_val: float = 0.0) -> pd.DataFrame:
    df = df.copy()
    # solo las que EXISTEN en el df (para evitar KeyError)
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return df
    # normaliza espacios, coma->punto y fuerza numérico
    for c in cols:
        s = df[c].astype(str).str.strip().replace({"": np.nan})
        s = s.str.replace(",", ".", regex=False)
        df[c] = pd.to_numeric(s, errors="coerce").fillna(fillna_val)
    return df

















# def split_train_test_apred(n_exp:int|str,mes_train:list[int],mes_test:int|list[int],mes_apred:int,semilla:int=SEMILLA,subsampleo:float=SUBSAMPLEO)->Tuple[pd.DataFrame,pd.Series, pd.Series, pd.Series, pd.DataFrame, pd.Series, pd.Series, pd.Series,pd.DataFrame,pd.DataFrame]:
#     logger.info("Comienzo del slpiteo de TRAIN - TEST - APRED")
#     mes_train_sql = f"{mes_train[0]}"
#     for m in mes_train[1:]:    
#         mes_train_sql += f",{m}"
#     sql_train=f"""select *
#                 from df_init
#                 where foto_mes IN ({mes_train_sql})"""
#     if isinstance(mes_test,list):
#         mes_test_sql = f"{mes_test[0]}"
#         for m in mes_test[1:]:    
#             mes_test_sql += f",{m}"
#         sql_test=f"""select *
#                     from df_init
#                     where foto_mes IN ({mes_test_sql})"""
#     elif isinstance(mes_test,int):
#         mes_test_sql = f"{mes_test}"
#         sql_test=f"""select *
#                     from df_init
#                     where foto_mes = {mes_test_sql}"""
        
#     mes_apred_sql = f"{mes_apred}"
#     sql_apred=f"""select *
#                 from df_init
#                 where foto_mes = {mes_apred_sql}"""
    
#     conn=duckdb.connect(PATH_DATA_BASE_DB)
#     train_data = conn.execute(sql_train).pl()
#     test_data = conn.execute(sql_test).pl()
#     apred_data = conn.execute(sql_apred).pl()
#     conn.close()
#     if subsampleo is not None:
#         train_data=undersampling(train_data , subsampleo,semilla)

#     # --- TRAIN ---
#     X_train = train_data.drop(["clase_ternaria", "clase_weight", "clase_binary"])
#     y_train_binaria = train_data.get_column("clase_binary")     # pl.Series
#     y_train_class  = train_data.get_column("clase_ternaria")    # pl.Series (str/categorical)
#     w_train  = train_data.get_column("clase_weight")        # pl.Series (float)

#     # --- TEST ---
#     X_test = test_data.drop(["clase_ternaria", "clase_weight", "clase_binary"])
#     y_test_binaria = test_data.get_column("clase_binary")
#     y_test_class = test_data.get_column("clase_ternaria")
#     w_test = test_data.get_column("clase_weight")


#     # --- A PREDECIR ---
#     X_apred = apred_data.drop(["clase_ternaria", "clase_weight", "clase_binary"])
#     # y_apred como DF con la columna numero_de_cliente
#     y_apred = X_apred.select(["numero_de_cliente"])  # pl.DataFrame

#     logger.info(f"X_train shape : {X_train.shape} / y_train shape : {y_train_binaria.len()} de los meses : {X_train.select(pl.col('foto_mes').unique().sort()).to_series().to_list()}")
#     logger.info(f"X_test shape : {X_test.shape} / y_test shape : {y_test_binaria.len()}  del mes : {    X_test.select(pl.col('foto_mes').unique().sort()).to_series().to_list()}")
#     logger.info(f"X_apred shape : {X_apred.shape} / y_apred shape : {y_apred.shape}  del mes : {X_apred.select(pl.col('foto_mes').unique().sort()).to_series().to_list()}")

#     vc_train = y_train_binaria.value_counts().sort(by="clase_binary")
#     vc_test  = y_test_binaria.value_counts().sort(by="clase_binary")
#     logger.info(
#         f"cantidad de baja y continua en train: "
#         f"{dict(zip(vc_train['clase_binary'].to_list(), vc_train['count'].to_list()))}")
#     logger.info(
#         f"cantidad de baja y continua en test: "
#         f"{dict(zip(vc_test['clase_binary'].to_list(), vc_test['count'].to_list()))}")
#     logger.info("Finalizacion label binario")
#     logger.info("Transformacion a pandas ")
#     # Conversión a pandas
#     X_train       = X_train.to_pandas()
#     y_train_binaria= y_train_binaria.to_pandas()
#     y_train_class  = y_train_class.to_pandas()
#     w_train     = w_train.to_pandas()

#     X_test= X_test.to_pandas()
#     y_test_binaria= y_test_binaria.to_pandas()
#     y_test_class= y_test_class.to_pandas()
#     w_test= w_test.to_pandas()

#     X_apred= X_apred.to_pandas()
#     y_apred= y_apred.to_pandas()


#     return X_train, y_train_binaria,y_train_class, w_train, X_test, y_test_binaria, y_test_class, w_test ,X_apred , y_apred 



# def undersampling(df: pl.DataFrame, undersampling_rate: float, semilla: int) -> pl.DataFrame:
#     logger.info("Comienzo del subsampleo")
#     np.random.seed(semilla)

#     clientes_minoritaria = (
#         df.filter(pl.col("clase_ternaria") != "Continua")
#         .select("numero_de_cliente")
#         .unique()
#         .to_series()
#         .to_numpy()
#     )

#     clientes_mayoritaria = (
#         df.filter(pl.col("clase_ternaria") == "Continua")
#         .select("numero_de_cliente")
#         .unique()
#         .to_series()
#         .to_numpy()
#     )

#     logger.info(f"Clientes minoritarios: {len(clientes_minoritaria)}")
#     logger.info(f"Clientes mayoritarios: {len(clientes_mayoritaria)}")

#     n_sample = int(len(clientes_mayoritaria) * undersampling_rate)
#     clientes_mayoritaria_sample = np.random.choice(clientes_mayoritaria, n_sample, replace=False)

#     clientes_finales = np.concatenate([clientes_minoritaria, clientes_mayoritaria_sample])

#     df_train_undersampled = df.filter(pl.col("numero_de_cliente").is_in(clientes_finales))

#     logger.info(f"Shape original: {df.shape}")
#     logger.info(f"Shape undersampled: {df_train_undersampled.shape}")
#     df_train_undersampled = df_train_undersampled.sample(fraction=1.0, shuffle=True, seed=semilla)

#     return df_train_undersampled


# def split_train_test_apred_python(df:pl.DataFrame|np.ndarray , mes_train:list[int],mes_test:list[int],mes_apred:int,semilla:int=SEMILLA,subsampleo:float=None) ->Tuple[pd.DataFrame,pd.Series, pd.Series, pd.Series, pd.DataFrame, pd.Series, pd.Series, pd.Series,pd.DataFrame,pd.DataFrame]:
#     logger.info(f"mes train={mes_train}  -  mes test={mes_test} - mes apred={mes_apred} ")

#     train_data = df[df['foto_mes'].isin(mes_train)]
#     test_data = df[df['foto_mes'].isin(mes_test)]
#     apred_data = df[df['foto_mes'] == mes_apred]

#     if subsampleo is not None:
#         train_data=undersampling(train_data , subsampleo,semilla)

#     # TRAIN
#     X_train = train_data.drop(['clase_ternaria', 'clase_weight', 'clase_binary'], axis=1)
#     y_train_binaria = train_data['clase_binary']
#     y_train_class=train_data["clase_ternaria"]
#     w_train = train_data['clase_weight']

#     # TEST
#     X_test = test_data.drop(['clase_ternaria', 'clase_weight','clase_binary'], axis=1)
#     y_test_binaria = test_data['clase_binary']
#     y_test_class = test_data['clase_ternaria']
#     w_test = test_data['clase_weight']


#     # A PREDECIR
#     X_apred = apred_data.drop(['clase_ternaria', 'clase_weight','clase_binary'], axis=1)
#     y_apred=X_apred[["numero_de_cliente"]] # DF
  

#     logger.info(f"X_train shape : {X_train.shape} / y_train shape : {y_train_binaria.shape} de los meses : {X_train['foto_mes'].unique()}")
#     logger.info(f"X_test shape : {X_test.shape} / y_test shape : {y_test_binaria.shape}  del mes : {X_test['foto_mes'].unique()}")
#     logger.info(f"X_apred shape : {X_apred.shape} / y_apred shape : {y_apred.shape}  del mes : {X_apred['foto_mes'].unique()}")

#     logger.info(f"cantidad de baja y continua en train:{np.unique(y_train_binaria,return_counts=True)}")
#     logger.info(f"cantidad de baja y continua en test:{np.unique(y_test_binaria,return_counts=True)}")
#     logger.info("Finalizacion label binario")
#     return X_train, y_train_binaria,y_train_class, w_train, X_test, y_test_binaria, y_test_class, w_test ,X_apred , y_apred 

