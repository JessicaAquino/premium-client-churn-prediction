import numpy as np
import pandas as pd
import logging
import json
from src.config import *
from src.configuracion_inicial import creacion_df_small
from src.constr_lista_cols import contruccion_cols,cols_a_dropear_variable_entera,cols_a_dropear_variable_por_feat ,cols_a_dropear_variable_originales_o_percentiles,cols_a_dropear_variable_originales_o_corregidas ,cols_conteo_servicios_productos,cols_beneficios_presion_economica
from src.feature_engineering import conversion_parquet,feature_engineering_mes,feature_engineering_ctrx_norm,feature_engineering_mpayroll_sobre_edad,copia_tabla,feature_engineering_correccion_variables_por_mes_por_media,suma_de_prod_servs,suma_ganancias_gastos,ratios_ganancia_gastos,feature_engineering_percentil,feature_engineering_lag,feature_engineering_delta,feature_engineering_max_min,feature_engineering_ratio,feature_engineering_linreg,feature_engineering_drop_cols,feature_engineering_drop_meses
## ---------------------------------------------------------Configuraciones Iniciales -------------------------------
## Carga de variables

logger=logging.getLogger(__name__)

def lanzar_feat_eng(fecha:str ,n_fe:int , proceso_ppal:str):
    numero=n_fe
    #"""----------------------------------------------------------------------------------------------"""
    name=f"FEAT_ENG_{numero}_{proceso_ppal}_VENTANA_{VENTANA}"
    logger.info(f"PROCESO PRINCIPAL ---> {proceso_ppal}")
    logger.info(f"Comienzo del experimento : {name}")

    # =========================================
    # AGREGAR :  MES - CTRX NORM -  MPAYROLL/EDAD
    df_completo_chiquito=creacion_df_small("df_completo")
    feature_engineering_mes(df_completo_chiquito)
    feature_engineering_ctrx_norm(df_completo_chiquito)
    feature_engineering_mpayroll_sobre_edad(df_completo_chiquito)


    # PERCENTIL
    df_completo_chiquito=creacion_df_small("df_completo") # Para agregar las columnas de las corregidas
    cols_percentil = ["mpayroll_sobre_edad"]
    feature_engineering_percentil(df_completo_chiquito[["mpayroll_sobre_edad"]] ,cols_percentil,bins=20)

 
     
    df_completo_chiquito=creacion_df_small("df_completo")
    lista_featues_adicionales = ["ctrx_quarter_normalizado","mpayroll_sobre_edad","mpayroll_sobre_edad_percentil"]
    df_completo_chiquito = df_completo_chiquito[lista_featues_adicionales]

    feature_engineering_lag(df_completo_chiquito,lista_featues_adicionales,ORDEN_LAGS)
    feature_engineering_delta(df_completo_chiquito,lista_featues_adicionales,ORDEN_LAGS)
    feature_engineering_linreg(df_completo_chiquito , lista_featues_adicionales,VENTANA)
    feature_engineering_max_min(df_completo_chiquito,lista_featues_adicionales ,VENTANA)
    
    # ------------- a partir de aca se trabaja con df------------------------#

    #DROPEO DE COLUMNAS ORIGINALES O CORREGIDAS
    # df_completo_chiquito=creacion_df_small("df")
    # cols_a_dropear_corr =cols_a_dropear_variable_originales_o_corregidas(df_completo_chiquito ,a_eliminar="originales")
    # feature_engineering_drop_cols(df_completo_chiquito, columnas=cols_a_dropear_corr)

    #DROPEO DE COLUMNAS ORIGINALES/CORREGIDAS O PERCENTILES
    # df_completo_chiquito=creacion_df_small("df")
    # cols_a_dropear_perc =cols_a_dropear_variable_originales_o_percentiles(df_completo_chiquito ,a_eliminar="originales")
    # feature_engineering_drop_cols(df_completo_chiquito, columnas=cols_a_dropear_perc )


    #DROPEO DE VARIABLE EN PARTICULAR Y TODAS SUS VARIANTES
    #df_completo_chiquito=creacion_df_small("df")
    # cols_a_dropear = cols_a_dropear_variable_entera(df_chiquito , ["mcuentas_saldo"])
    # feature_engineering_drop_cols(df_completo_chiquito, columnas=cols_a_dropear )

    #DROPEO DE VARIABLE+FEATURE EN PARTICULAR
    # df_completo_chiquito=creacion_df_small("df")
    # cols_a_dropear = cols_a_dropear_variable_por_feat(df_chiquito , ["mcuentas_saldo"],["_lag_1"])
    # feature_engineering_drop_cols(df_completo_chiquito, columnas=cols_a_dropear)

    #DROPEO DE MESES
    # df_completo_chiquito=creacion_df_small("df")
    # meses_a_dropear=[202106]
    # feature_engineering_drop_meses(meses_a_dropear)



    logger.info("================ FIN DEL PROCESO DE FEAT ENG =============================")





    

    
