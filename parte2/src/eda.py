from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import polars as pl
import duckdb 
import logging

from src.config import *

logger = logging.getLogger(__name__)
def nunique_por_mes(df:pd.DataFrame|pl.DataFrame ,name:str , filtros_target:int|tuple=None) ->pl.DataFrame|pd.DataFrame:
    logger.info("Comienzo del eda de nunique por foto_mes")
    name_file = name + "_nuniques_por_mes.csv"

    drop_cols = ["foto_mes" ]

    cols = [ c for c in df.columns if c not in drop_cols]
    
    sql='select foto_mes'

    for c in cols:
        sql+=f', count(distinct({c})) as {c}_nunique'

    sql+=' from df_completo'

    # if filtros_target is not None:
    #     if isinstance(filtros_target , tuple):
    #         sql+=f' where clase_ternaria in {filtros_target}'
    #     elif isinstance(filtros_target , int):
    #         sql+=f' where clase_ternaria = {filtros_target}'
    sql+=' group by foto_mes'

    con = duckdb.connect(PATH_DATA_BASE_DB)
    nuniques_por_mes = con.execute(sql).df()
    con.close()
    logger.info("Intento de guardado")
    try:
        nuniques_por_mes.to_csv(PATH_OUTPUT_EDA+ name_file)
        logger.info("Fin del eda de media por foto_mes")
    except Exception as e:
        logger.error(f"No se pudo guardar por : {e}")
    
    return nuniques_por_mes


def mean_por_mes(df:pd.DataFrame|pl.DataFrame ,name:str, filtros_target:int|tuple=None) ->pl.DataFrame|pd.DataFrame:
    logger.info("Comienzo del eda de media por foto_mes")
    
    name_file = name + "_medias_por_mes.csv"
    
    
    if isinstance(df , pl.DataFrame):
        num_cols=df.select(pl.selectors.numeric()).columns
    elif isinstance(df , pd.DataFrame):
        num_cols = df.select_dtypes(include="number").columns

    drop_cols = ["foto_mes"]

    num_cols = [ c for c in num_cols if c not in drop_cols]
    
    # Veo primero cuales son los uniques de foto_mes
    # logger.info("Vemos los uniques de los meses")
    # sql = 'select distinct(foto_mes) from df order by foto_mes'
    # con = duckdb.connect(database=":memory:")
    # con.register("df",df)
    # mes_unique = con.execute(sql).df()
    # con.close()
    # logger.info(f"Los unicos meses: {mes_unique}")

    # logger.info("Seguimos con el eda de media por foto_mes")
    
    sql='select foto_mes'

    for c in num_cols:
        sql+=f', AVG({c}) as {c}_mean'
    sql+=' from df_completo '

    # if filtros_target is not None:
    #     if isinstance(filtros_target , tuple):
    #         sql+=f' where clase_ternaria in {filtros_target}'
    #     elif isinstance(filtros_target , int):
    #         sql+=f' where clase_ternaria = {filtros_target}'

    sql += ' group by foto_mes'

    con = duckdb.connect(PATH_DATA_BASE_DB)
    medias_por_mes = con.execute(sql).df()
    con.close()

    logger.info("Intento de guardado")
    try:
        medias_por_mes.to_csv(PATH_OUTPUT_EDA+name_file)
        logger.info("Fin del eda de media por foto_mes")

    except Exception as e:
        logger.error(f"No se pudo guardar por : {e}")
    
    return medias_por_mes

# def std_por_mes(df:pd.DataFrame|pl.DataFrame , filtros_target:int|tuple=None) ->pl.DataFrame|pd.DataFrame:
#     logger.info("Comienzo del eda de std por foto_mes")
#     if isinstance(df , pl.DataFrame):
#         num_cols=df.select(pl.selectors.numeric()).columns
#     elif isinstance(df , pd.DataFrame):
#         num_cols = df.select_dtypes(include="number").columns

#     drop_cols = ["foto_mes" ]

#     num_cols = [ c for c in num_cols if c not in drop_cols]
    
#     # Veo primero cuales son los uniques de foto_mes

    
#     sql='select foto_mes'

#     for c in num_cols:
#         sql+=f', STDDEV_SAMP({c}) as {c}_STD'
#     sql+=' from df'
    
#     if filtros_target is not None:
#         if isinstance(filtros_target , tuple):
#             sql+=f' where clase_ternaria in {filtros_target}'
#         elif isinstance(filtros_target , int):
#             sql+=f' where clase_ternaria = {filtros_target}'
#     sql +=' group by foto_mes'

#     con = duckdb.connect(database=":memory:")
#     con.register("df",df)
#     variacion_por_mes = con.execute(sql).df()
#     con.close()
#     logger.info("Fin del eda de std por foto_mes")
#     return variacion_por_mes

# def crear_reporte_pdf(df, xcol, columnas_y, name_eda, motivo:str):
#     """
#     Genera un PDF con una página por gráfico 
#     """
#     name_pdf=name_eda +"_reporte_"+motivo+".pdf"
#     logger.info("Comienzo de la creacion del reporte")

#     drop_cols = ["foto_mes" ]

#     columnas_y = [ c for c in columnas_y if c not in drop_cols]

#     n_months = len(df["foto_mes"].unique())

#     salida_pdf = PATH_OUTPUT_EDA+name_pdf
#     df["_fecha"] = pd.to_datetime(df[xcol].astype(str), format="%Y%m")
#     with PdfPages(salida_pdf) as pdf:
#         for col in columnas_y:
#             fig, ax = plt.subplots(figsize=(11.69, 8.27))
#             sns.lineplot(data=df , x = "_fecha" , y =col,ax=ax, marker='o')
#             ax.set_title(f"{col} vs {xcol}")
#             ax.set_xlabel(xcol)
#             ax.set_ylabel(col)
#             ax.grid(True, alpha=0.3)
#             if n_months > 6:
#                 ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
#             else:
#                 ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
#             ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
#             plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

#             fig.tight_layout()
#             pdf.savefig(fig)
#             plt.close(fig)

#         # d = pdf.infodict()
#         # d['Title'] = titulo
#         # d['Author'] = "Tu nombre"
#         # d['Subject'] = "Reporte automático de gráficos"
#         # d['Keywords'] = "matplotlib, reporte, gráficos"
#         # d['Creator'] = "Python + Matplotlib"
#         logger.info("Fin de la creacion del reporte")


def crear_reporte_pdf(df, xcol, columnas_y, name_eda, motivo: str):
    """
    Genera un PDF con una página por gráfico 
    """
    name_pdf = name_eda + "_reporte_" + motivo + ".pdf"
    logger.info("Comienzo de la creacion del reporte")

    drop_cols = ["foto_mes"]
    columnas_y = [c for c in columnas_y if c not in drop_cols]

    salida_pdf = PATH_OUTPUT_EDA + name_pdf

    # Paso foto_mes (o la que venga en xcol) a datetime
    df = df.copy()
    df["_fecha"] = pd.to_datetime(df[xcol].astype(str), format="%Y%m")

    # Ticks solo donde hay datos
    fechas = df["_fecha"].sort_values().unique()

    with PdfPages(salida_pdf) as pdf:
        for col in columnas_y:
            fig, ax = plt.subplots(figsize=(11.69, 8.27))

            sns.lineplot(data=df, x="_fecha", y=col, ax=ax, marker='o')

            ax.set_title(f"{col} vs {xcol}")
            ax.set_xlabel(xcol)
            ax.set_ylabel(col)
            ax.grid(True, alpha=0.3)

            # Usamos solo las fechas reales del dataset
            ax.set_xticks(fechas)
            ax.set_xticklabels(
                [d.strftime("%Y-%m") for d in fechas],
                rotation=45,
                ha="right"
            )

            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        logger.info("Fin de la creacion del reporte")
