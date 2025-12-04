#main.py
import matplotlib.pyplot as plt
import datetime 
import logging

from src.config import *
from src.configuracion_inicial import creacion_directorios,creacion_logg_local , creacion_logg_global
from src.generadora_semillas import create_semilla
from src.creacion_target import lanzar_creacion_clase_ternaria_binaria_peso
from src_feateng.feat_eng_armado_tablas import lanzar_feat_eng
from src_bayesianas.bayesiana_lgbm_2 import lanzar_bayesiana_lgbm
from src_bayesianas.bayesiana_xgb_2 import lanzar_bayesiana_xgb
from src_experimentos.experimento_eda import lanzar_eda
from src_experimentos.c_3_exp_ENSAMBLE_automatico import lanzar_experimento_lgbm
from src_experimentos.experimento_1_xgb_edited import lanzar_experimento_xgb




## ---------------------------------------------------------Configuraciones Iniciales -------------------------------


## Carga de variables
n_trials=N_TRIALS
creacion_directorios()
fecha = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
test= "TEST_TEST_TEST_TEST"
# ---------------------------------------------------------------------------------------------
competencia = COMPETENCIA
proceso_ppal = PROCESO_PPAL
# Conf fe
n_fe = N_FE
# Conf exp
n_experimento = N_EXPERIMENTO
n_semillas_exp = N_SEMILLAS_EXP
model_exp = MODEL_EXP
# Conf bay
n_bayesiana = N_BAYESIANA
n_semillas_bay=N_SEMILLAS_BAY
model_bay = MODEL_BAY
# ---------------------------------------------------------------------------------------------------------------------------
if proceso_ppal =="feat_eng":
    numero_proceso = n_fe
elif proceso_ppal in ["experimento", "test_exp","prediccion_final","test_prediccion_final"]:
    numero_proceso = n_experimento
    n_semillas = n_semillas_exp
elif proceso_ppal in ["bayesiana","test_baye"]:
    numero_proceso = n_bayesiana
    n_semillas = n_semillas_bay
try:
    study_name = f"_COMP_{competencia}_{proceso_ppal}_{numero_proceso}"
except Exception as e:
    study_name = f"_COMP_{competencia}_{proceso_ppal}"
nombre_log=fecha+study_name
# CONFIGURACION LOG LOCAL
creacion_logg_local(nombre_log=nombre_log)
logger = logging.getLogger(__name__)

## --------------------------------------------------------Funcion main ------------------------------------------

def main():
    
    logger.info(f"Inicio de ejecucion del flujo : {nombre_log}")
    try:
        semillas = create_semilla(n_semillas=n_semillas)
        logger.info(f"se crearon {len(semillas)} semillas")
    except Exception as e:
        logger.error(f"No es un error, pero no  se crearon semillas porque estamos en el proceso {proceso_ppal}. : {e}")

    # CONFIGURACION LOG GLOBAL
    try:
        creacion_logg_global(fecha=fecha , competencia=competencia ,proceso_ppal=proceso_ppal,n_experimento=n_experimento,n_semillas=n_semillas)
    except:
        creacion_logg_global(fecha=fecha , competencia=competencia ,proceso_ppal=proceso_ppal,n_experimento=n_experimento,n_semillas=[])

    if proceso_ppal =="creacion_target_clase_ternaria":
        lanzar_creacion_clase_ternaria_binaria_peso()
    elif proceso_ppal=="analisis_exploratorio":
        lanzar_eda(competencia=competencia)
    elif proceso_ppal =="feat_eng":
        lanzar_feat_eng(fecha,n_fe ,proceso_ppal)
    elif proceso_ppal =="bayesiana":
        if model_bay =="lgbm":
            lanzar_bayesiana_lgbm(fecha,semillas,n_bayesiana,proceso_ppal)
        elif model_bay =="xgb":
            lanzar_bayesiana_xgb(fecha,semillas,n_bayesiana,proceso_ppal)
    elif proceso_ppal =="test_baye":
        if model_bay =="lgbm":
            lanzar_bayesiana_lgbm(test,semillas,n_bayesiana,proceso_ppal)
        elif model_bay =="xgb":
            lanzar_bayesiana_xgb(test,semillas,n_bayesiana,proceso_ppal)
    elif (proceso_ppal =="test_exp") or (proceso_ppal=="test_prediccion_final"):
        if model_exp =="lgbm":
            lanzar_experimento_lgbm(test,semillas , n_experimento , proceso_ppal)
        elif model_exp =="xgb":
            lanzar_experimento_xgb(test,semillas , n_experimento , proceso_ppal)
    
    elif (proceso_ppal =="experimento") or (proceso_ppal=="prediccion_final"):
        if model_exp =="lgbm":
            lanzar_experimento_lgbm(fecha,semillas , n_experimento , proceso_ppal)
        elif model_exp =="xgb":
            lanzar_experimento_xgb(fecha,semillas , n_experimento , proceso_ppal)
    logger.info(f"Fin de la ejecucion del log: {nombre_log}")

    return

if __name__ =="__main__":
    main()