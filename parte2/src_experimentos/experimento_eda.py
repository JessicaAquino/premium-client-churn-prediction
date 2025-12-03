import duckdb 
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import *
from src.configuracion_inicial import creacion_df_small
from src.constr_lista_cols import contrs_cols_dropear_por_features_sufijos ,cols_a_dropear_variable_originales_o_percentiles
from src.eda import mean_por_mes , crear_reporte_pdf , nunique_por_mes



def lanzar_eda(competencia:str|int):
    name_eda= f"eda_comp_{competencia}"
    logger.info(f"Comienzo del eda {name_eda}")

    df_completo_chiquito=creacion_df_small("df_completo")

    sufijos=[f"lag_{i}" for i in range(1,4)]+[f"delta_{i}" for i in range(1,4)]+["_ratio","_slope","_max","_min","suma_de_",
             "monto_ganancias","monto_gastos","ganancia_gasto_dif"]
    cols_drops_1=contrs_cols_dropear_por_features_sufijos(df_completo_chiquito,sufijos)
    
    df_completo_chiquito=creacion_df_small("df_completo")
    cols_drops_2=cols_a_dropear_variable_originales_o_percentiles(df_completo_chiquito,a_eliminar="percentiles")

    cols_drops=list(set(cols_drops_1 + cols_drops_2 ))

    exclude=''
    for i,f in enumerate(cols_drops):
        if i ==0:
            exclude+=f'EXCLUDE({f}'
        else:
            exclude+=f',{f}'
    exclude+=')'
    sql=f""" SELECT * {exclude} from df_completo """
    logger.info(f"Query con las columnas a eliminar : {sql}")
    conn=duckdb.connect(PATH_DATA_BASE_DB)
    df = conn.execute(sql).df()
    conn.close()
    logger.info(f"df shape despues del exclude : {df.shape}")

    # df= pl.read_csv(FILE_INPUT_DATA, infer_schema_length=10000)
    # df=pl.read_parquet(FILE_INPUT_DATA_PARQUET)
    print(df.head(10))
    logger.info(df["foto_mes"].unique())
    # filtros_target=("BAJA+1","BAJA+2")
    media_por_mes = mean_por_mes(df=df , name=name_eda, filtros_target=None)

    crear_reporte_pdf(media_por_mes, xcol='foto_mes', columnas_y=media_por_mes.columns,
                  name_eda=name_eda,
                  motivo="media_por_mes")
    
    # variacion_por_mes = std_por_mes(df, filtros_target=filtros_target)
    # crear_reporte_pdf(variacion_por_mes, xcol='foto_mes', columnas_y=variacion_por_mes.columns,
    #               name_eda=name_eda,
    #               motivo="std_por_mes")
    
    num_uniques_por_mes = nunique_por_mes(df=df , name=name_eda, filtros_target=None)
    crear_reporte_pdf(num_uniques_por_mes, xcol='foto_mes', columnas_y=num_uniques_por_mes.columns,
                  name_eda=name_eda,
                  motivo="nunique_por_mes")
