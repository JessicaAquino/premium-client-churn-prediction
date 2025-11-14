#preprocesamiento.py
import pandas as pd
import polars as pl
import numpy as np
from sklearn.impute import SimpleImputer
from typing import Tuple
from src.config import SEMILLA, PATH_DATA_BASE_DB
import logging
import duckdb
from src.config import SUBSAMPLEO
logger = logging.getLogger(__name__)
# columnas problemáticas (las que te tiró LightGBM)
ERR_COLS = (
    ["tmobile_app", "cmobile_app_trx",
     "cmobile_app_trx_max", "cmobile_app_trx_min",
     "tmobile_app_max", "tmobile_app_min"]
    + [f"cmobile_app_trx_lag_{i}" for i in range(1, 5)]
    + [f"tmobile_app_lag_{i}" for i in range(1, 5)]
)


def split_train_test_apred(n_exp:int|str,mes_train:list[int],mes_test:int|list[int]
                           ,mes_apred:int,semilla:int=SEMILLA,
                           subsampleo:float=SUBSAMPLEO , feature_subset= None,n_canaritos:int=None)->Tuple[pd.DataFrame,
                                                               np.ndarray,np.ndarray, 
                                                               np.ndarray, pd.DataFrame, 
                                                               np.ndarray, np.ndarray, 
                                                               np.ndarray,pd.DataFrame,
                                                               pd.DataFrame]:
    logger.info("Comienzo del slpiteo de TRAIN - TEST - APRED")

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

    mes_train_sql = f"{mes_train[0]}"
    for m in mes_train[1:]:    
        mes_train_sql += f",{m}"
    sql_train=f"""select {sql_canaritos} * {exclude} 
                from df_completo
                where foto_mes IN ({mes_train_sql})"""
    if isinstance(mes_test,list):
        mes_test_sql = f"{mes_test[0]}"
        for m in mes_test[1:]:    
            mes_test_sql += f",{m}"
        sql_test=f"""select {sql_canaritos} * {exclude}
                    from df_completo
                    where foto_mes IN ({mes_test_sql})"""
    elif isinstance(mes_test,int):
        mes_test_sql = f"{mes_test}"
        sql_test=f"""select {sql_canaritos} * {exclude}
                    from df_completo
                    where foto_mes = {mes_test_sql}"""
        
    mes_apred_sql = f"{mes_apred}"
    sql_apred=f"""select {sql_canaritos} * {exclude}
                from df_completo
                where foto_mes = {mes_apred_sql}"""
    
    conn=duckdb.connect(PATH_DATA_BASE_DB)
    seed_float = (semilla % 10000) / 10000.0
    conn.execute("SELECT setseed(?)", [seed_float])
    train_data = conn.execute(sql_train).df()
    test_data = conn.execute(sql_test).df()
    apred_data = conn.execute(sql_apred).df()
    conn.close()
    if subsampleo is not None:
        train_data=undersampling(train_data , subsampleo,semilla)
    logger.info(f"Terminada la carga de df con columnas: {train_data.columns}")
    # TRAIN
    X_train = train_data.drop(['clase_ternaria', 'clase_peso', 'clase_binaria'], axis=1)
    y_train_binaria = train_data['clase_binaria'].to_numpy()
    y_train_class=train_data["clase_ternaria"].to_numpy()
    w_train = train_data['clase_peso'].to_numpy()

    # TEST
    X_test = test_data.drop(['clase_ternaria', 'clase_peso','clase_binaria'], axis=1)
    y_test_binaria = test_data['clase_binaria'].to_numpy()
    y_test_class = test_data['clase_ternaria'].to_numpy()
    w_test = test_data['clase_peso'].to_numpy()


    # A PREDECIR
    X_apred = apred_data.drop(['clase_ternaria', 'clase_peso','clase_binaria'], axis=1)
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
    return X_train, y_train_binaria,y_train_class, w_train, X_test, y_test_binaria, y_test_class, w_test ,X_apred , y_apred 

# def split_train_test_apred(n_exp:int|str,mes_train:list[int],mes_test:int|list[int],mes_apred:int,semilla:int=SEMILLA,subsampleo:float=SUBSAMPLEO)->Tuple[pd.DataFrame,pd.Series, pd.Series, pd.Series, pd.DataFrame, pd.Series, pd.Series, pd.Series,pd.DataFrame,pd.DataFrame]:
#     logger.info("Comienzo del slpiteo de TRAIN - TEST - APRED")
#     mes_train_sql = f"{mes_train[0]}"
#     for m in mes_train[1:]:    
#         mes_train_sql += f",{m}"
#     sql_train=f"""select *
#                 from df_completo
#                 where foto_mes IN ({mes_train_sql})"""
#     if isinstance(mes_test,list):
#         mes_test_sql = f"{mes_test[0]}"
#         for m in mes_test[1:]:    
#             mes_test_sql += f",{m}"
#         sql_test=f"""select *
#                     from df_completo
#                     where foto_mes IN ({mes_test_sql})"""
#     elif isinstance(mes_test,int):
#         mes_test_sql = f"{mes_test}"
#         sql_test=f"""select *
#                     from df_completo
#                     where foto_mes = {mes_test_sql}"""
        
#     mes_apred_sql = f"{mes_apred}"
#     sql_apred=f"""select *
#                 from df_completo
#                 where foto_mes = {mes_apred_sql}"""
    
#     conn=duckdb.connect(PATH_DATA_BASE_DB)
#     train_data = conn.execute(sql_train).pl()
#     test_data = conn.execute(sql_test).pl()
#     apred_data = conn.execute(sql_apred).pl()
#     conn.close()
#     if subsampleo is not None:
#         train_data=undersampling(train_data , subsampleo,semilla)

#     # --- TRAIN ---
#     X_train = train_data.drop(["clase_ternaria", "clase_peso", "clase_binaria"])
#     y_train_binaria = train_data.get_column("clase_binaria")     # pl.Series
#     y_train_class  = train_data.get_column("clase_ternaria")    # pl.Series (str/categorical)
#     w_train  = train_data.get_column("clase_peso")        # pl.Series (float)

#     # --- TEST ---
#     X_test = test_data.drop(["clase_ternaria", "clase_peso", "clase_binaria"])
#     y_test_binaria = test_data.get_column("clase_binaria")
#     y_test_class = test_data.get_column("clase_ternaria")
#     w_test = test_data.get_column("clase_peso")


#     # --- A PREDECIR ---
#     X_apred = apred_data.drop(["clase_ternaria", "clase_peso", "clase_binaria"])
#     # y_apred como DF con la columna numero_de_cliente
#     y_apred = X_apred.select(["numero_de_cliente"])  # pl.DataFrame

#     logger.info(f"X_train shape : {X_train.shape} / y_train shape : {y_train_binaria.len()} de los meses : {X_train.select(pl.col('foto_mes').unique().sort()).to_series().to_list()}")
#     logger.info(f"X_test shape : {X_test.shape} / y_test shape : {y_test_binaria.len()}  del mes : {    X_test.select(pl.col('foto_mes').unique().sort()).to_series().to_list()}")
#     logger.info(f"X_apred shape : {X_apred.shape} / y_apred shape : {y_apred.shape}  del mes : {X_apred.select(pl.col('foto_mes').unique().sort()).to_series().to_list()}")

#     vc_train = y_train_binaria.value_counts().sort(by="clase_binaria")
#     vc_test  = y_test_binaria.value_counts().sort(by="clase_binaria")
#     logger.info(
#         f"cantidad de baja y continua en train: "
#         f"{dict(zip(vc_train['clase_binaria'].to_list(), vc_train['count'].to_list()))}")
#     logger.info(
#         f"cantidad de baja y continua en test: "
#         f"{dict(zip(vc_test['clase_binaria'].to_list(), vc_test['count'].to_list()))}")
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
#     X_train = train_data.drop(['clase_ternaria', 'clase_peso', 'clase_binaria'], axis=1)
#     y_train_binaria = train_data['clase_binaria']
#     y_train_class=train_data["clase_ternaria"]
#     w_train = train_data['clase_peso']

#     # TEST
#     X_test = test_data.drop(['clase_ternaria', 'clase_peso','clase_binaria'], axis=1)
#     y_test_binaria = test_data['clase_binaria']
#     y_test_class = test_data['clase_ternaria']
#     w_test = test_data['clase_peso']


#     # A PREDECIR
#     X_apred = apred_data.drop(['clase_ternaria', 'clase_peso','clase_binaria'], axis=1)
#     y_apred=X_apred[["numero_de_cliente"]] # DF
  

#     logger.info(f"X_train shape : {X_train.shape} / y_train shape : {y_train_binaria.shape} de los meses : {X_train['foto_mes'].unique()}")
#     logger.info(f"X_test shape : {X_test.shape} / y_test shape : {y_test_binaria.shape}  del mes : {X_test['foto_mes'].unique()}")
#     logger.info(f"X_apred shape : {X_apred.shape} / y_apred shape : {y_apred.shape}  del mes : {X_apred['foto_mes'].unique()}")

#     logger.info(f"cantidad de baja y continua en train:{np.unique(y_train_binaria,return_counts=True)}")
#     logger.info(f"cantidad de baja y continua en test:{np.unique(y_test_binaria,return_counts=True)}")
#     logger.info("Finalizacion label binario")
#     return X_train, y_train_binaria,y_train_class, w_train, X_test, y_test_binaria, y_test_class, w_test ,X_apred , y_apred 


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


