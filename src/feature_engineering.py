#feature_engineering.py
import pandas as pd
import numpy as np
import duckdb
import logging

#### FALTA AGREGAR LOS PUNTOS DE CONTROL PARA VISUALIZAR QUE ESTEN BIEN
from src.config import FILE_INPUT_DATA , PATH_DATA_BASE_DB,GCP_PATH
logger = logging.getLogger(__name__)

def copia_tabla_local_a_bucket():
    """
    Copia la tabla df_completo desde la base local (PATH_DATA_BASE_DB)
    hacia la base ubicada en el bucket.
    """
    logger.info("Inicio de la copia de df_completo desde local a bucket")

    conn = duckdb.connect(PATH_DATA_BASE_DB)
    try:
        # Adjuntamos la base del bucket
        conn.execute(f"""
            ATTACH '{GCP_PATH + PATH_DATA_BASE_DB}' AS db_destino;
        """)

        # Copiamos la tabla local hacia la base en el bucket
        conn.execute("""
            CREATE OR REPLACE TABLE db_destino.df_completo AS
            SELECT * FROM df_completo;
        """)

        conn.execute("DETACH db_destino;")
        logger.info("✅ Copia completada: df_completo (local) → df_completo (bucket)")
    except Exception as e:
        logger.error(f"❌ Error durante la copia local → bucket: {e}")
    finally:
        conn.close()

def copia_tabla(tabla_origen:str , tabla_copia:str):
    logger.info(f"Copia de la tabla {tabla_origen} a {tabla_copia}")
    sql = f"create or replace table {tabla_copia} as "
    sql+=f"""SELECT *
            from {tabla_origen}"""
    conn=duckdb.connect(PATH_DATA_BASE_DB)
    conn.execute(sql)
    conn.close()
    logger.info(f"Finalizada la Copia de la tabla {tabla_origen} a {tabla_copia}")
    

def feature_engineering_drop_cols(df:pd.DataFrame , columnas:list[str],tabla_origen:str="df",tabla_nueva:str="df") :
    logger.info(f"Comienzo del dropeo de las variables : {columnas}")

    sql = f"create or replace table {tabla_nueva} as "
    logger.info(f"Comienzo dropeo de {len(columnas)} columnas en la tabla {tabla_nueva}.")
    sql+= "SELECT * EXCLUDE("
    for i,c in enumerate(columnas):
        if c in df.columns:
            if i==0:
                sql+=f" {c}"
            else:
                sql+=f",{c}"
        else:
            logger.warning(f"No se encontro la columna {c} en la tabla")
    sql+= f") from {tabla_origen}"
    try:
        conn=duckdb.connect(PATH_DATA_BASE_DB)
        conn.execute(sql)
        conn.close()
        logger.info(f"Fin del dropeo de las variables : {columnas}.")
    except Exception as e:
        logger.error(f"Error al intentar crear en la base de datos --> {e}")
        raise
    return
def feature_engineering_drop_meses( meses_a_dropear:list ,tabla_origen:str="df",tabla_nueva:str="df"):
    logger.info(f"Comienzo del dropeo de los meses {meses_a_dropear}")
    query_meses = f"({meses_a_dropear[0]}"
    for m in meses_a_dropear[1:]:
        query_meses+= f", {m}"
    query_meses+= ")"

    sql = f"create or replace table {tabla_nueva} as "
    sql += f"""select *
                from {tabla_origen}
                where foto_mes not in {query_meses}"""
    conn=duckdb.connect(PATH_DATA_BASE_DB)
    conn.execute(sql)
    conn.close()
    logger.info(f"Fin del dropeo de los meses {meses_a_dropear}")
    return


def feature_engineering_correccion_variables_por_mes_por_media(df:pd.DataFrame , variable_meses_dict : dict):
    """ Funcion para corregir las variables en los meses que estan feos
    Hay que mandarle el diccionario de la siguiente forma :
    variable_meses_dict={
    "mrentabilidad":"(202106 , 202107)",
    "cpayroll":"(202107,202108)"} 

    Esto va a agregar una columna nueva llamada variable_corregida , en donde sea igual a la media de lso valores anterior y posterior, o si el mes esta bien, deja el valor por defecto
    """
    logger.info(f"Comienzo de la Correccion de las variables por media por mes de {variable_meses_dict}")
    variables_no_corregidas = {k:v for k,v in variable_meses_dict.items() if f"{k}_corregida" not in df.columns}
    variables_ya_corregidas = {k:v for k,v in variable_meses_dict.items() if f"{k}_corregida" in df.columns}
    if len(variables_no_corregidas)==0:
        logger.info(f"Ya se realizo la correcion de las variables y meses indicados : {variable_meses_dict}")
        return
    elif len(variables_no_corregidas)==len(variable_meses_dict):
        logger.info(f"Nunca se realizo la correcion de las variables y meses indicados : {variable_meses_dict}")
    elif len(variables_no_corregidas) < len(variable_meses_dict):
        logger.info(f"ya se realizo la correcion de las variables y meses indicados : {variables_ya_corregidas}")
        logger.info(f"Pero nunca se realizo la correcion de las variables y meses indicados : {variables_no_corregidas}")

        
    sql="create or replace table df_completo as "
    sql+="select *"
    for var,meses in variables_no_corregidas.items():
        sql+=f""", if(foto_mes IN {meses} , 
                (lag({var},1) over (partition by numero_de_cliente order by foto_mes) +  lead({var},1) over (partition by numero_de_cliente order by foto_mes))/2.0, 
                {var}) as {var}_corregida"""
    sql+=" from df_completo"
    
    conn=duckdb.connect(PATH_DATA_BASE_DB)
    conn.execute(sql)
    conn.close()
    logger.info(f"Fin de la correccion de las variables por la media por mes de {variable_meses_dict}")
    return

    


def feature_engineering_percentil(df:pd.DataFrame , columnas:list[str],bins:int=20):
    logger.info("Comienzo de la transformacion en percentil")

    if any( c.endswith("_percentil") for c in df.columns):
        logger.info("Ya se realizo percentil")
        return
    logger.info("Todavia no se realizo percentil")
    sql ="""CREATE or REPLACE table df_completo as """
    sql += f""" select *""" 
    for c in columnas:
        if c in df.columns:
            sql += f""", ntile({bins}) OVER (partition by foto_mes order by {c} ) as {c}_percentil"""
        else:
            logger.warning(f"No se encontro la columna {c} en el df")
    
    sql += " from df_completo"
    
    conn=duckdb.connect(PATH_DATA_BASE_DB)
    conn.execute(sql)
    conn.close()
    logger.info("Finalizacion del feature percentil")
    return

def suma_de_prod_servs( df:pd.DataFrame,columnas:list  ,prod_serv:str):
    logger.info(f"Comienzo de la suma de productos y servicios : {prod_serv}")

    nombre_columna = f"suma_de_{prod_serv}"

    if nombre_columna in df.columns:
        logger.info(f"Ya se realizo la suma para: {prod_serv}")
        return
    logger.info("Todavia no se realizo suma de productos y servicios")


    sql="create or replace table df_completo as "
    sql+="select * "
    for i,c in enumerate(columnas) :
        if i==0:
            sql+=f",if(try_cast({c} as double)>0, 1,0)"
        else:
            sql+=f"+ if(try_cast({c} as double)>0,1,0)"
    sql+=f" as {nombre_columna}"
    sql+=" from df_completo"
    conn=duckdb.connect(PATH_DATA_BASE_DB)
    conn.execute(sql)
    conn.close()
    logger.info(f"Fin de la suma de productos y servicios {prod_serv}")
    return
def suma_ganancias_gastos(df:pd.DataFrame,cols_ganancias:list ,cols_gastos:list):
    logger.info(f"Comienzo de las sumas y ratio ganancias y gastos")

    nombre_ganancia = "monto_ganancias"
    nombre_gasto = "monto_gastos"

    if nombre_ganancia in df.columns or nombre_gasto in df.columns:
        logger.info(f"Ya se realizo la suma de ganancias y gastos")
        return

    logger.info("Todavia no se realizo las sumas y ratios de ganancias y gastos")

    sql="create or replace table df_completo as "
    sql+= "select * "
    for i,c in enumerate(cols_ganancias) :
        if i==0:
            sql+=f",try_cast({c} as double)"
        else:
            sql+=f"+try_cast({c} as double)"
    sql+=f" as {nombre_ganancia}"

    for i,c in enumerate(cols_gastos) :
        if i==0:
            sql+=f",try_cast({c} as double)"
        else:
            sql+=f"+try_cast({c} as double)"
    sql+=f" as {nombre_gasto}"
    sql+=" from df_completo"
    conn=duckdb.connect(PATH_DATA_BASE_DB)
    conn.execute(sql)
    conn.close()
    logger.info(f"Fin de la suma de productos de ganancias y gastos")
    return

def ratios_ganancia_gastos(df:pd.DataFrame):

    logger.info("Comienzo del ratio ganancia_gasto")
    if "ganancia_gasto_ratio" in df.columns:
        logger.info("Ya se realizo el ratio ganancia gasto")
        return
    logger.info("Todavia no se realizo el ratio ganancia gasto")
    sql="create or replace table df_completo as "
    sql+=" select *"
    sql+=f", if(monto_gastos is NULL ,NULL,monto_ganancias/(monto_gastos+1)) as ganancia_gasto_ratio "
    sql+=" from df_completo"
    conn=duckdb.connect(PATH_DATA_BASE_DB)
    conn.execute(sql)
    conn.close()
    logger.info(f"Fin del ratio de productos de ganancias y gastos")
    return

def feature_engineering_lag(df:pd.DataFrame ,columnas:list[str],orden_lag:int=1 ):
    
    logger.info(f"Comienzo Feature de lag")

    orden_lag_ya_realizado=1
    marca_nueva_real=0
    while marca_nueva_real ==0 and orden_lag_ya_realizado <= orden_lag:
        if any(c.endswith(f"_lag_{orden_lag_ya_realizado}") for c in df.columns):
            logger.info(f"Ya se hizo lag_{orden_lag_ya_realizado}")
            orden_lag_ya_realizado+=1
        else:
            marca_nueva_real=1
    if orden_lag_ya_realizado > orden_lag:
        logger.info(f"Ya se hicieron todos los lags pedidos hasta orden {orden_lag_ya_realizado-1}")
        return
    
    logger.info(f"Ya se hicieron los lags hasta orden {orden_lag_ya_realizado-1}. Falta hasta orden {orden_lag}")
    sql = "CREATE or REPLACE table df_completo as "
    sql +="(SELECT *"
    for attr in columnas:
        if attr in df.columns:
            for i in range(orden_lag_ya_realizado,orden_lag+1):
                sql+= f",lag(try_cast({attr} as double),{i}) OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes) AS {attr}_lag_{i}"
        else:
            logger.warning(f"No se encontro el atributo {attr} en df")
    sql+=" FROM df_completo)"

    # Ejecucion de la consulta SQL
    conn = duckdb.connect(PATH_DATA_BASE_DB)
    conn.execute(sql)
    conn.close()
    logger.info(f"ejecucion lag finalizada")
    return

def feature_engineering_delta(df:pd.DataFrame , columnas:list[str],orden_delta:int=1 ) :
    
    logger.info(f"Comienzo feature de delta")
    orden_delta_ya_realizado=1
    marca_nueva_real=0
    while marca_nueva_real ==0 and orden_delta_ya_realizado <= orden_delta:
        if any(c.endswith(f"_delta_{orden_delta_ya_realizado}")  for c in df.columns):
            logger.info(f"Ya se hizo delta_{orden_delta_ya_realizado}_")
            orden_delta_ya_realizado+=1
        else:
            marca_nueva_real=1
    if orden_delta_ya_realizado > orden_delta:
        logger.info(f"Ya se hicieron todos los deltas pedidos hasta orden {orden_delta_ya_realizado-1}")
        return
    logger.info(f"Ya se hicieron los deltas hasta orden {orden_delta_ya_realizado-1}. Falta hasta orden {orden_delta}")

    
    sql = "CREATE or REPLACE table df_completo as "
    sql+="(SELECT *"
    for attr in columnas:
        if attr in df.columns:
            for i in range(orden_delta_ya_realizado,orden_delta+1):
                sql += (
                f", TRY_CAST({attr} AS DOUBLE) "
                f"- TRY_CAST({attr}_lag_{i} AS DOUBLE) AS {attr}_delta_{i}")
                # sql+= f", {attr}-{attr}_lag_{i} as delta_{i}_{attr}"
        else:
            logger.warning(f"No se encontro el atributo {attr} en df")
    sql+=" FROM df_completo)"
    conn = duckdb.connect(PATH_DATA_BASE_DB)
    conn.execute(sql)
    conn.close()
    logger.info(f"ejecucion delta finalizada")
    return 

def feature_engineering_ratio(df:pd.DataFrame|pd.Series, columnas:list[list[str]] ):
    """
    Genera variables de ratio para los atributos especificados utilizando SQL.
  
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos
    columnas : list[list]
        Lista de pares de columnas de monto y cantidad relacionados para generar ratios. Si es None, no se generan lags.
    cant_lag : int, default=1
        Cantidad de delta a generar para cada atributo
  
    Returns:
    --------
    pd.DataFrame
        DataFrame con las variables de ratios agregadas"""
    logger.info(f"Comienzo feature ratio")

    if any(c.endswith(f"_ratio") for c in df.columns):
        logger.info("Ya se hizo ratios")
        return
    logger.info("Todavia no se hizo ratios")
    sql="CREATE or REPLACE table df_completo as "
    sql+="(SELECT *"
    for par in columnas:
        if par[0] in df.columns and par[1] in df.columns:
            sql+=f", if({par[1]}=0 ,0,{par[0]}/{par[1]}) as {par[0]}_{par[1]}_ratio"
        else:
            print(f"no se encontro el par de atributos {par}")

    sql+=" FROM df_completo)"

    conn = duckdb.connect(PATH_DATA_BASE_DB)
    conn.execute(sql)
    conn.close()

    logger.info(f"ejecucion ratio finalizada.")
    return 

def feature_engineering_linreg(df : pd.DataFrame|np.ndarray , columnas:list[str],ventana:int=3) :
    logger.info(f"Comienzo feature reg lineal")

    if any(c.endswith("_slope") for c in df.columns):
        logger.info("Ya se hizo slope")
        return
    logger.info("Todavia no se hizo slope")
    sql = "Create or replace table df_completo as "
    sql+="SELECT *"
    try:
        for attr in columnas:
            if attr in df.columns:
                sql+=f", regr_slope(try_cast({attr} as double) , try_cast(cliente_antiguedad as double) ) over ventana_{ventana} as {attr}_slope"
            else :
                print(f"no se encontro el atributo {attr}")
        sql+=f" FROM df_completo window ventana_{ventana} as (partition by numero_de_cliente order by foto_mes rows between {ventana} preceding and current row)"
    except Exception as e:
        logger.error(f"Error en la regresion lineal : {e}")
        raise
    conn = duckdb.connect(PATH_DATA_BASE_DB)
    conn.execute(sql)
    conn.close()
    logger.info(f"ejecucion reg lineal finalizada")
    return 

def feature_engineering_max_min(df : pd.DataFrame|np.ndarray , columnas:list[str],ventana:int=3) :
    logger.info(f"Comienzo feature max min. df shape: {df.shape}")
    palabras_max_min=["_max","_min"]
    if any(any(c.endswith(p) for p in palabras_max_min) for c in df.columns):
        logger.info("Ya se hizo max min")
        return
    
    sql="CREATE or REPLACE table df_completo as "
    sql+="(SELECT *"
    try:

        for attr in columnas:
            if attr in df.columns:
                sql+=f", max(try_cast({attr} as double)  ) over ventana_{ventana} as {attr}_max ,min(try_cast({attr} as double)) over ventana_{ventana} as {attr}_min"
            else :
                print(f"no se encontro el atributo {attr}")
        sql+=f" FROM df_completo window ventana_{ventana} as (partition by numero_de_cliente order by foto_mes rows between {ventana} preceding and current row))"
    except Exception as e:
        logger.error(f"Error en la max min : {e}")
        raise
    conn=duckdb.connect(PATH_DATA_BASE_DB)
    conn.execute(sql)
    conn.close()
    logger.info(f"ejecucion max min finalizada. ")
    return 


# def feature_engineering_rank(df: pd.DataFrame, columnas: list[str]) -> pd.DataFrame:
#     """
#     Genera variables de ranking para los atributos especificados utilizando SQL.

#     Parameters:
#     -----------
#     df : pd.DataFrame
#         DataFrame con los datos
#     columnas : list[str]
#         Lista de atributos para los cuales generar rankings.

#     Returns:
#     --------
#     pd.DataFrame
#         DataFrame con las variables de ranking agregadas
#     """

#     if not columnas:
#         raise ValueError("La lista de columnas no puede estar vacía")

#     columnas_validas = [col for col in columnas if col in df.columns]
#     if not columnas_validas:
#         raise ValueError("Ninguna de las columnas especificadas existe en el DataFrame")

#     logger.info(f"Realizando feature engineering RANK para {len(columnas_validas)} columnas: {columnas_validas}")

#     logger.info(f"Antes del ranking : la media de la columna {columnas_validas[0]} en 04 es de {df.loc[df['foto_mes']==202104,columnas_validas[0]].mean()}")

#     rank_expressions = [
#         f"PERCENT_RANK() OVER (PARTITION BY foto_mes ORDER BY {col}) AS {col}_rank"
#         for col in columnas_validas
#     ]

#     sql = f"""
#     SELECT *,
#            {', '.join(rank_expressions)}
#     FROM df_completo
#     """

#     con = duckdb.connect(PATH_DATA_BASE_DB)
#     con.execute(sql)
#     con.close()
#     logger.info(f"Despues del ranking : la media de la columna {columnas_validas[0]} ")
#     logger.info(f"Feature engineering completado")
#     return 
    


# def feature_engineering_max_min_2(df:pd.DataFrame|np.ndarray , columnas:list[str]) -> pd.DataFrame|np.ndarray:
#     """
#     Genera variables de max y min para los atributos especificados por numero de cliente  utilizando SQL.
  
#     Parameters:
#     -----------
#     df : pd.DataFrame
#         DataFrame con los datos
#     columnas : list
#         Lista de atributos para los cuales generar min y max. Si es None, no se generan lags.
#     cant_lag : int, default=1
#         Cantidad de delta a generar para cada atributo
  
#     Returns:
#     --------
#     pd.DataFrame
#         DataFrame con las variables de lag agregadas
#     """
#     logger.info(f"Comienzo feature max min. df shape: {df.shape}")
      
#     sql="SELECT *"
#     for attr in columnas:
#         if attr in df.columns:
#             sql+=f", MAX({attr}) OVER (PARTITION BY numero_de_cliente) as MAX_{attr}, MIN({attr}) OVER (PARTITION BY numero_de_cliente) as MIN_{attr}"
#         else:
#             print(f"El atributo {attr} no se encuentra en el df")
    
#     sql+=" FROM df"

#     con = duckdb.connect(database=":memory:")
#     con.register("df", df)
#     df=con.execute(sql).df()
#     con.close()
#     logger.info(f"ejecucion max min finalizada. df shape: {df.shape}")
#     return df

