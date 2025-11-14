import duckdb
import polars as pl
import pandas as pd
from src.config import *
import logging

logger =logging.getLogger(__name__)

def create_data_base():
    logger.info(f"Creacion de la base de datos en : {PATH_DATA_BASE_DB}")
    sql = f"""
    create or replace table df_inicial as 
    select *
    from read_csv_auto('{FILE_INPUT_DATA_CRUDO}')"""

    conn=duckdb.connect(PATH_DATA_BASE_DB)
    conn.execute(sql)
    conn.close()

def contador_targets():
    logger.info("Inicio control de la cantidad de los targets")
    sql="""
        select foto_mes , 
        COUNT(*) FILTER(where clase_ternaria = 'BAJA+1') as "BAJA+1",
        COUNT(*) FILTER(where clase_ternaria = 'BAJA+2') as "BAJA+2",
        COUNT(*) FILTER(where clase_ternaria = 'Continua') as "Continua"
        from df_inicial
        group by foto_mes"""
    conn = duckdb.connect(PATH_DATA_BASE_DB)
    contador=conn.execute(sql).df()
    conn.close()
    logger.info(f"Contador clase ternaria : {contador}")
    logger.info("Fin control de la cantidad de los targets")


def creacion_clase_ternaria() :
    logger.info("Inicio de la creacion del target")

    sql= f"""CREATE or REPLACE table df_inicial as
    (with df2 as (
    SELECT foto_mes , numero_de_cliente,
    lead(foto_mes  , 1 ) OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes) as foto_mes_1,
    lead(foto_mes  , 2 ) OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes) as foto_mes_2
    FROM df_inicial)
    SELECT * EXCLUDE (foto_mes_1,foto_mes_2),
    if (foto_mes < 202108 , if(foto_mes <202107 ,
    if(df2.foto_mes_1 IS NULL,'BAJA+1', 
    if(df2.foto_mes_2 IS NULL,'BAJA+2','Continua')) ,
    if(df2.foto_mes_1 IS NULL,'BAJA+1',NULL)) ,NULL) as clase_ternaria
    from df_inicial
    LEFT JOIN df2 USING (numero_de_cliente,foto_mes))
    """
    conn = duckdb.connect(PATH_DATA_BASE_DB)
    conn.execute(sql)
    conn.close()
    logger.info(f"Fin de la creacion de target")
    return 

def conversion_binario():
    conn=duckdb.connect(PATH_DATA_BASE_DB)
    logger.info("Comienzo de la creacion clase binaria")
    sql_creacion="create or replace table df_inicial as "
    sql_creacion += """ SELECT *, 
                if(clase_ternaria= 'BAJA+2' ,1.00002 ,
                    if(clase_ternaria = 'BAJA+1' , 1.00001,1.0)) as clase_peso ,
                if(clase_ternaria = 'Continua' ,0 ,1) as clase_binaria
                from df_inicial"""
    conn.execute(sql_creacion)
    
    sql_contador="""SELECT foto_mes,
            COUNT(*) FILTER( where clase_peso = 1.00002) as peso_baja_2,
            COUNT(*) FILTER(where clase_peso = 1.00001) as peso_baja_1,
            COUNT(*) FILTER(where clase_peso = 1.0) as peso_continua,
            COUNT(*) FILTER(where clase_binaria =1) as binaria_bajas,
            COUNT(*) FILTER(where clase_binaria =0) as binaria_continua
            from df_inicial
            group by foto_mes"""
    contador_clase_peso =conn.execute(sql_contador).df()
    conn.close()
    logger.info(f"contador clase binaria y peso : \n{contador_clase_peso}")
    logger.info("Finalizacion de la creacion clase binaria")
    return


def lanzar_creacion_clase_ternaria_binaria_peso():
    logger.info("Lanzamiento de la creacion de la clase ternaria, binaria y clase peso target")
    create_data_base()
    creacion_clase_ternaria()
    contador_targets()
    conversion_binario()
    return 


