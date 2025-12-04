#constr_lista_cols.py
import pandas as pd
import numpy as np
import logging
import duckdb
from typing import Tuple

from src.config import  PATH_DATA_BASE_DB
logger=logging.getLogger(__name__)

def contruccion_cols(df:pd.DataFrame)->Tuple[list,list,list]:
    logger.info("Comienzo de la extraccion de la seleccion de las columnas")
    palabras_features_excluir=["_lag","_delta","_slope","_max","_min","_ratio","_mean"]


    columnas_cleaned=[c for c in df.columns if not any(p in c for p in palabras_features_excluir)]
    
    col_drops=["numero_de_cliente","foto_mes","mes","active_quarter","clase_ternaria","clase_binary","clase_binary2","clase_weight","cliente_edad","cliente_antiguedad",
        "Visa_fultimo_cierre","Visa_fultimo_cierre","Master_fultimo_cierre","Visa_Fvencimiento",
        "Master_Fvencimiento"]
 
    lista_t=[c for c in list(map(lambda x : x if x[0]=='t' and x not in col_drops else np.nan ,columnas_cleaned )) if pd.notna(c)]
    lista_c=[c for c in list(map(lambda x : x if x[0]=='c' and x not in col_drops else np.nan ,columnas_cleaned )) if pd.notna(c)]
    lista_m=[c for c in list(map(lambda x : x if x[0]=='m' and x not in col_drops else np.nan ,columnas_cleaned )) if pd.notna(c)]
    lista_r=[c for c in columnas_cleaned if c not in (lista_t + lista_c + lista_m +col_drops )]


    # # Columnas lags y delta
    cols_lag_delta_max_min_regl=lista_m + lista_c+ lista_r +lista_t
    cols_percentil =  lista_m + [c for c in lista_r if "_m" in c]


    # # Columnas para regresion lineal y max-min
    # lista_regl_max_min = lista_m + lista_c+ lista_r+lista_r

    # # Columnas para los ratios
    cols_ratios=[]
    for c in lista_c:
        i=0
        while i < len(lista_m) and c[1:] != lista_m[i][1:]:
            i+=1
        if i < len(lista_m):
            cols_ratios.append([lista_m[i],c ])




    logger.info(f"columnas para lags y deltas ")
    logger.info(f"columnas para ratios :")
    logger.info("Finalizacion de la construccion de las columnas")

    return cols_percentil,cols_lag_delta_max_min_regl ,cols_ratios 

def cols_conteo_servicios_productos(df:pd.DataFrame)->Tuple[list,list,list,list,list,list,list,list]:
    dict_prod_serv={
    "master_visa_productos" : ["Master_msaldototal","Master_mconsumototal","Master_mpagado","Master_mlimitecompra",
    "Visa_msaldototal","Visa_mconsumototal","Visa_mpagado","Visa_mlimitecompra"],

    "cuentas_productos" : ['mcuenta_corriente','mcaja_ahorro','mcaja_ahorro_dolares','mcuentas_saldo',
                        'mcuenta_corriente_adicional','mcaja_ahorro_adicional'],
    "tarjetas_productos" :[c for c in df.columns if 'tarjeta' in c and c[0]=="m"],
    "prestamos_productos" : [c for c in df.columns if 'prest' in c and c[0]=="m"],
    "inversiones_productos" : ['mplazo_fijo_pesos','mplazo_fijo_dolares',
                            'minversion1_pesos','minversion1_dolares','minversion2'],
    "digitales_productos" : [c for c in df.columns if c[0]=="t"],
    "servicios_productos" : ['mpagodeservicios','mpagomiscuentas','mcuenta_debitos_automaticos'
                          ,'mforex_buy','mforex_sell','mtransferencias_recibidas','mtransferencias_emitidas',
                          'mextraccion_autoservicio','mcheques_depositados','mcheques_emitidos','mcajeros_propios_descuentos'],
    "seguros_productos" : [c for c in df.columns if 'segur' in c]}
    return dict_prod_serv

def cols_beneficios_presion_economica(df:pd.DataFrame):
    ganancias_gastos={"ganancias" : [
    # Ingresos por sueldo
    "mpayroll","mpayroll2",
    # Ahorro e inversiones (indican colchón financiero)
    "mplazo_fijo_pesos","mplazo_fijo_dolares","minversion1_pesos","minversion1_dolares","minversion2",
    # Plata que entra por transferencias
    "mtransferencias_recibidas",
    # Beneficios/descuentos (mejoran la situación neta)
    "mcajeros_propios_descuentos","mtarjeta_visa_descuentos","mtarjeta_master_descuentos"],
    
    "gastos": [# Comisiones y costos directos
    "mcomisiones","mcomisiones_mantenimiento","mcomisiones_otras",
    # Débitos y pagos de servicios (egresos automáticos)
    "mcuenta_debitos_automaticos","mpagodeservicios","mpagomiscuentas",
    # Deuda / préstamos (presión financiera)"mprestamos_personales",
    "mprestamos_prendarios","mprestamos_hipotecarios","mpasivos_margen",
    # Egresos por movimientos
    "mtransferencias_emitidas","mextraccion_autoservicio",
    "mcheques_emitidos","mcheques_depositados_rechazados","mcheques_emitidos_rechazados",
    # Uso de cajeros/ATM (generalmente salida de plata)
    "matm","matm_other"]
    }
    return ganancias_gastos


def cols_a_dropear_variable_originales_o_corregidas(df: pd.DataFrame, a_eliminar : str = "originales") -> list[str]:
    """
    La uso para eliminar las variables originales o las que corregi con la media
    Es mas para dejar las variables originales o las que corregi
    """
    logger.info("Comienzo de la seleccion de  originales o corregidas para eliminar")
    logger.info(f"Se eligio {a_eliminar} para eliminar")
    variables_corregidas = [c for c in df.columns if "_corregida" in c]
    variables_originales = list(map(lambda x : x.replace("_corregida",""), variables_corregidas))
    if a_eliminar =="originales":
        cols_a_dropear = variables_originales
    elif a_eliminar =="corregidas":
        cols_a_dropear = variables_corregidas
    logger.info(f"Fin de la seleccion de {a_eliminar} para eliminar")
    return cols_a_dropear

def cols_a_dropear_variable_originales_o_percentiles(df: pd.DataFrame, a_eliminar : str = "originales") -> list[str]:
    """
    La uso para eliminar las variables originales , las primeras 150 y pico, o las percentiles
    Es mas para dejar las variables originales o las de percentiles
    """
    logger.info("Comienzo de la seleccion de persentiles o originales para eliminar")
    logger.info(f"Se eligio {a_eliminar} para eliminar")
    variables_percentiles = [c for c in df.columns if "_percentil" in c]
    variables_originales = list(map(lambda x : x.replace("_percentil","") , variables_percentiles))
    if a_eliminar =="originales":
        cols_a_dropear = variables_originales
    elif a_eliminar =="percentiles":
        cols_a_dropear = variables_percentiles
    logger.info(f"Fin de la seleccion de {a_eliminar} para eliminar")
    return cols_a_dropear

def cols_a_dropear_variable_entera(df: pd.DataFrame, columnas_base: list[str]) -> list[str]:
    """
    A partir de columnas base (ej: 'mrentabilidad'), detecta también
    columnas derivadas como:
      - mrentabilidad_lag_1, mrentabilidad_lag_2, ...
      - delta_mrentabilidad
      - max_mrentabilidad
      - min_mrentabilidad
      - slope_mrentabilidad
    """
    logger.info(f"Comienzo de la seleccion de la variable {columnas_base} ENTERA y todas sus derivadas para ELIMINAR")
    
    cols_a_dropear = [c for c in df.columns for variable in columnas_base if variable in c]
    if len(cols_a_dropear)==0:
        logger.warning(f"No se encontro ninguna de las columnas en el df")
    logger.info(f"Fin del proceso de la seleccion de la variable entera y todas sus derivadas para ELIMINAR: {cols_a_dropear}")
    
    return list(set(cols_a_dropear))


def cols_a_dropear_variable_por_feat(df: pd.DataFrame, columnas_variables: list[str] , columnas_features:list[str]) -> list[str]:
    logger.info(f"Comienzo de la seleccion de las variables {columnas_features} a elimnar por features {columnas_features}")
    cols_a_dropear =[]
    for cv in columnas_variables:
        for cf in columnas_features:
            cols_a_dropear.append(cv+cf)
    cols_a_dropear = [c for c in cols_a_dropear if c in df.columns]
    logger.info(f"Fin de la seleccion de las variables a elimnar por feature {cols_a_dropear}")

    return cols_a_dropear

def contrs_cols_dropear_por_features_sufijos(df:pd.DataFrame,cols_features_sufijos:list[str]):
    logger.info(f"Comienzo de la seleccion de cols a dropear por features sufijos: {cols_features_sufijos}")
    cols_a_dropear=[c for c in df.columns for fs in cols_features_sufijos if fs in c]
    return cols_a_dropear


def contrs_cols_dropear_feat_imp(df:pd.DataFrame , file:str , threshold:float)->list[str]:
    logger.info(f"Comienzo de la seleccion de columnas a dropear")
    importance_df=pd.read_excel(file)
    f = importance_df["importance_%"]<=threshold
    cols_menos_importantes=list(importance_df.loc[f,'feature'].unique())
    cols_no_dropear=["foto_mes","numero_de_cliente"]
    cols_dropear=[c for c in  cols_menos_importantes if c not in cols_no_dropear]
    logger.info(f"Fin de la seleccion de columnas a dropear. Se eliminaran {len(cols_dropear)} columnas")

    return cols_dropear