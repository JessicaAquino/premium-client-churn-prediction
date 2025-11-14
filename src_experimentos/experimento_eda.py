import duckdb 
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import *
from src.eda import mean_por_mes , crear_reporte_pdf,std_por_mes , nunique_por_mes



def lanzar_eda(competencia:str|int):
    name_eda= f"eda_comp_{competencia}_bajas"
    df= pl.read_csv(FILE_INPUT_DATA, infer_schema_length=10000)
    print(df.head(10))
    logger.info(df["foto_mes"].unique())
    filtros_target=("BAJA+1","BAJA+2")
    media_por_mes = mean_por_mes(df=df , name=name_eda, filtros_target=filtros_target)

    crear_reporte_pdf(media_por_mes, xcol='foto_mes', columnas_y=media_por_mes.columns,
                  name_eda=name_eda,
                  motivo="media_por_mes")
    
    variacion_por_mes = std_por_mes(df, filtros_target=filtros_target)
    crear_reporte_pdf(variacion_por_mes, xcol='foto_mes', columnas_y=variacion_por_mes.columns,
                  name_eda=name_eda,
                  motivo="std_por_mes")
    
    num_uniques_por_mes = nunique_por_mes(df=df , name=name_eda, filtros_target=filtros_target)
    crear_reporte_pdf(num_uniques_por_mes, xcol='foto_mes', columnas_y=num_uniques_por_mes.columns,
                  name_eda=name_eda,
                  motivo="nunique_por_mes")
