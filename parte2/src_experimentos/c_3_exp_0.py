# EXPERIMENTO 0
#experimento_i.py
 
import numpy as np
import pandas as pd
import logging
import json

from src.config import *
from src.configuracion_inicial import creacion_df_small
from src.constr_lista_cols import cols_a_dropear_variable_entera,contrs_cols_dropear_por_features_sufijos ,cols_a_dropear_variable_originales_o_percentiles
from src.preprocesamiento import verificacion_o_creacion_tabla,split_train_test_apred
from src.lgbm_train_test import preparacion_nclientesbajas_zulip,entrenamiento_zlgbm,entrenamiento_lgbm,entrenamiento_lgbm_zs,grafico_feature_importance,prediccion_test_lgbm ,calc_estadisticas_ganancia,grafico_curvas_ganancia, grafico_hist_ganancia ,preparacion_ypred_kaggle,preparacion_ytest_proba_kaggle
## ---------------------------------------------------------Configuraciones Iniciales -------------------------------
## Carga de variables

logger=logging.getLogger(__name__)

def lanzar_experimento_lgbm(fecha:str ,semillas:list[int],n_experimento:int,proceso_ppal:str ="experimento"): 
    #"""---------------------- CAMBIAR INPUTS --------------------------------------------------------"""
    numero=n_experimento
    #"""----------------------------------------------------------------------------------------------"""
    n_semillas = len(semillas)
    name=f"{proceso_ppal}_{numero}_LGBM_{len(semillas)}_SEMILLAS"
    logger.info(f"PROCESO PRINCIPAL ---> {proceso_ppal}")
    logger.info(f"Comienzo del experimento : {name} con {n_semillas} semillas")
    # Verificacion existencia .duckcb , .parquet ------------------------------------
    verificacion_o_creacion_tabla()
    # ---------------------- CONSTRUCCION COLUMNAS A ELIMINAR------------------------
    df_completo_chiquito=creacion_df_small("df_init")
    cols_drops_1=cols_a_dropear_variable_entera(df_completo_chiquito, ['mprestamos_personales','cprestamos_personales','suma_de_prestamos_productos'])
    
    df_completo_chiquito=creacion_df_small("df_init")
    cols_drops_2=cols_a_dropear_variable_originales_o_percentiles(df_completo_chiquito,a_eliminar="originales")

    cols_drops=list(set(cols_drops_1 + cols_drops_2))

    # Defini el path de los outputs de los modelos, de los graf de hist gan, de graf curva ganan, de umbrales, de feat import
    if (proceso_ppal =="experimento") or (proceso_ppal =="test_exp") :
        logger.info(f"LANZAMIENTO PARA EXP  CON {n_semillas} SEMILLAS")
        output_path_models = path_output_exp_model
        output_path_feat_imp = path_output_exp_feat_imp
        output_path_graf_ganancia_hist_total =path_output_exp_graf_gan_hist_total
        output_path_graf_curva_ganancia = path_output_exp_graf_curva_ganancia
        output_path_umbrales=path_output_exp_umbral
        logger.info(f"LANZAMIENTO PARA EXPERIMENTO {numero} CON {n_semillas} SEMILLAS")
        
    elif (proceso_ppal == "prediccion_final") or (proceso_ppal =="test_prediccion_final"):
        logger.info(f"LANZAMIENTO PARA PREDICCION FINAL CON {n_semillas} SEMILLAS")
        output_path_models = path_output_finales_model
        output_path_feat_imp =path_output_finales_feat_imp
    
    # 4. Carga de mejores Hiperparametros
    logger.info("Ingreso de hiperparametros de una Bayesiana ya realizada")
    #"""---------------------- CAMBIAR INPUTS --------------------------------------------------------"""
    tipo_bayesiana=TIPO_BAYESIANA
    numero_bayesiana_lgbm =N_BAYESIANA
    modelo_etiqueta="LGBM"
    cantidad_semillas =N_SEMILLAS_BAY
    cantidad_trials= N_TRIALS
    cantidad_boosts = N_BOOSTS
    #"""-----------------------------------------------------------------------------------------------"""
    if tipo_bayesiana != "ZLGBM":
        n_canaritos=None
        if( (proceso_ppal =="test_exp") or (proceso_ppal =="test_prediccion_final")):
            proceso_bayesiana = "test_baye"
        elif((proceso_ppal == "experimento" )or (proceso_ppal =="prediccion_final")):
            proceso_bayesiana = "bayesiana"
        name_best_params_file_lgbm=f"best_params{proceso_bayesiana}_{numero_bayesiana_lgbm}_{modelo_etiqueta}_TIPO_{tipo_bayesiana}_{cantidad_semillas}_SEMILLAS_{cantidad_trials}_TRIALS_{cantidad_boosts}_BOOSTS.json"

        try:
            with open(path_output_bayesian_bestparams+name_best_params_file_lgbm, "r") as f:
                best_params_dict = json.load(f)
                logger.info(f"Correcta carga de los best params del {modelo_etiqueta} ")

        except Exception as e:
            logger.error(f"No se pudo encontrar los best params ni best iter del {modelo_etiqueta} por el error {e}")
            raise
    
        if tipo_bayesiana =="OB":
            best_params_dict_sorted = sorted(best_params_dict , key=lambda x : x["ganancia_media_meseta"] ,reverse=True)
            top_models_bayesiana = best_params_dict_sorted[0:TOP_MODELS]
            top_bp = {b["trial_number"]:{"params":b["params"],"best_iter_trial":b["best_iter_trial"]} for b in top_models_bayesiana}
            logger.info(f"Los mejores Trials de la Bayesiana {N_BAYESIANA} son : {top_bp.keys()}")
            logger.info(f"Los mejores parametros del mejor modelo de la Bayesiana {N_BAYESIANA} son : {top_bp[list(top_bp.keys())[0]]['params']}")
            logger.info(f"El mejor num de iteracion del mejor modelo de la Bayesiana {N_BAYESIANA} es : {top_bp[list(top_bp.keys())[0]]['best_iter_trial']}")   

        elif tipo_bayesiana =="ZS":
            top_bp = {
                b["trial_number"]: {
                    "params": {k: v for k, v in b["params"].items() if k not in ["num_iterations", "seed"] },
                    "best_iter_trial": b["params"].get("num_iterations", None)
                }
                for b in best_params_dict
            }
    elif tipo_bayesiana =="ZLGBM":
        n_canaritos=N_CANARITOS
        top_bp = {0:0}
## 5. Primer Entrenamiento lgbm con la mejor iteracion y los mejores hiperparametros en [01,02,03] y evaluamos en 04 

    #3. spliteo train - test - apred - Subsampleo
    if (proceso_ppal =="experimento" or proceso_ppal =="test_exp") :
        logger.info(f"Entro en el proceso experimento !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        if isinstance(MES_TEST,int):
            mes_test=list(MES_TEST)
        elif isinstance(MES_TEST,list):
            mes_test = MES_TEST
        for mt in mes_test:
            logger.info(f"========================Comienzo del analisis en el mes test :{mt} ==============================")
            if mt == 202104:
                mes_train = MES_TRAIN_04
            elif mt == 202105:
                mes_train = MES_TRAIN_05
            elif mt == 202106:
                mes_train = MES_TRAIN_06
            elif mt == 202107:
                mes_train = MES_TRAIN_07
            X_train, y_train_binaria,_,y_train_class, w_train, X_test, y_test_binaria,_, y_test_class, w_test,_, _ = split_train_test_apred(n_experimento,mes_train,
                                                                                                                                        mt,MES_A_PREDECIR,
                                                                                                                                        SEMILLA,SUBSAMPLEO,feature_subset=cols_drops,n_canaritos=5)
            y_predicciones_top_models=[]
            for orden_trial , trial in enumerate(top_bp):
                if tipo_bayesiana!="ZLGBM":
                    name_trial = name + f"_TRIAL_{trial}_TOP_{orden_trial}_MES_TEST_{mt}"
                    best_params_i = top_bp[trial]["params"]
                    best_iter_i = top_bp[trial]["best_iter_trial"]
                    logger.info(f"Comienzo del modelo del top del orden {orden_trial} : trial={trial} con hiperparams {best_params_i}")
                    logger.info(f"Comienzo del modelo del top del orden {orden_trial} : trial={trial} con best iter {best_iter_i}")
                else:
                    name_trial = name+ f"_MES_TEST_{mt}"
                    logger.info(f"Comienzo del ZLGBM")
                y_predicciones_lista=[]
                y_pred_sorted_dict={}
                ganancia_acumulada_dict={}
                estadisticas_ganancia_dict={}
                for i,semilla in enumerate(semillas):
                    if tipo_bayesiana!="ZLGBM":
                        logger.info(f"Comienzo de la semilla numero {semilla} del orden {i} de {len(semillas)} iteraciones para el orden del trial {orden_trial} *****************************************")
                    else:
                        logger.info(f"Comienzo de la semilla numero {semilla} del orden {i}  del zlgbm *****************************************")

                    name_semilla=f"{name_trial}_SEMILLA_{semilla}_fase_testeo"
                    # Entrenamiento de los modelos --------
                    if tipo_bayesiana =="OB":
                        model_lgbm = entrenamiento_lgbm(X_train , y_train_binaria,w_train ,best_iter_i,best_params_i ,name_semilla,output_path_models,semilla)
                    elif tipo_bayesiana =="ZS":
                        model_lgbm = entrenamiento_lgbm_zs(X_train , y_train_binaria,w_train ,best_iter_i,best_params_i ,name_semilla,output_path_models,semilla)
                    elif tipo_bayesiana =="ZLGBM":
                        model_lgbm = entrenamiento_zlgbm(X_train , y_train_binaria  ,name_semilla,output_path_models,semilla)

                    # Grafico features importances -------
                    grafico_feature_importance(model_lgbm,X_train,name_semilla,output_path_feat_imp)

                    # Predicciones en test para cada modelo -------------
                    y_pred_lgbm=prediccion_test_lgbm(X_test ,model_lgbm)

                    y_predicciones_lista.append(y_pred_lgbm)

                    # Estadistica de ganancias -----------
                    if (proceso_ppal =="experimento" or proceso_ppal =="test_exp"):
                        guardar_umbral = False
                    else:
                        guardar_umbral=True

                    estadisticas_ganancia , y_pred_sorted,ganancia_acumulada= calc_estadisticas_ganancia(y_test_class , y_pred_lgbm ,name_semilla , output_path_umbrales , semilla, guardar_umbral )
                    
                    estadisticas_ganancia_dict[semilla] = estadisticas_ganancia 
                    y_pred_sorted_dict[semilla] = y_pred_sorted
                    ganancia_acumulada_dict[semilla] = ganancia_acumulada

                # Creacion de lqs predicciones medias a partir de los ensambles de las semillas
                semilla = "ensamble_semillas"
                name_semilla=f"{name_trial}_SEMILLA_{semilla}_fase_testeo"
             
                logger.info("Comienzo del ensamblado de todas las semillas")
                y_pred_df = np.vstack(y_predicciones_lista)
                logger.info(f" shape de la matriz con todas las predicciones ensamblado{y_pred_df.shape}")
                y_pred_ensamble = y_pred_df.mean(axis=0)
                # Guardado de predicciones
                preparacion_ytest_proba_kaggle(X=X_test,y_pred=y_pred_ensamble,name=name_semilla,output_path=path_output_exp_prediction)

                y_predicciones_top_models.append(y_pred_ensamble)

                logger.info("Fin del ensamblado ")
                
                estadisticas_ganancia , y_pred_sorted,ganancia_acumulada= calc_estadisticas_ganancia(y_test_class , y_pred_ensamble ,name_semilla , output_path_umbrales , semilla, guardar_umbral )
                estadisticas_ganancia_dict[semilla] = estadisticas_ganancia 
                y_pred_sorted_dict[semilla] = y_pred_sorted
                ganancia_acumulada_dict[semilla] = ganancia_acumulada
                if guardar_umbral == False:
                    try:
                        with open(output_path_umbrales+f"{name_trial}.json", "w") as f:
                            json.dump(estadisticas_ganancia_dict, f, indent=4)
                    except Exception as e:
                        logger.error(f"Error al intentar guardar el dict de umbral como json --> {e}")
                    logger.info(f"Las estadisticas de las ganancias son : {estadisticas_ganancia}")
                    logger.info("Fin del calculo de las estadisticas de la ganancia")

                logger.info("Vamos a graficar la curva de ganancia con y_pred_ensamble")
                grafico_curvas_ganancia(y_pred_sorted_dict ,ganancia_acumulada_dict,estadisticas_ganancia_dict,name_trial,output_path_graf_curva_ganancia)
                grafico_hist_ganancia(estadisticas_ganancia_dict , name_trial,output_path_graf_ganancia_hist_total)
            if tipo_bayesiana=="OB":
                logger.info("Comienzo ensamblado de los top models")
                name_final = name + f"_ENSAMBLE_FINAL_MES_TEST_{mt}"
                y_pred_ensamble_modelos_matrix = np.vstack(y_predicciones_top_models)
                logger.info(f" shape de la matriz con todas las predicciones ensamblado{y_pred_ensamble_modelos_matrix.shape}")
                y_pred_ensamble_final = y_pred_ensamble_modelos_matrix.mean(axis=0)
                pd.Series(y_pred_ensamble_final, index=X_test.index).to_csv(path_output_exp_prediction+ name_final)

                guardar_umbral = True
                semilla = "ENSAMBLE_FINAL_TOP_MODELS"
                estadisticas_ganancia , y_pred_sorted,ganancia_acumulada= calc_estadisticas_ganancia(y_test_class , y_pred_ensamble_final ,name_final , output_path_umbrales , semilla, guardar_umbral )
                grafico_curvas_ganancia(y_pred_sorted ,ganancia_acumulada,estadisticas_ganancia,name_final,output_path_graf_curva_ganancia)
                
    elif (proceso_ppal =="prediccion_final" or  proceso_ppal =="test_prediccion_final"):
        logger.info(f"========================Comienzo de la prediccion final=============================")
        mes_train = MES_TRAIN_PRED
        X_train, y_train_binaria,_,y_train_class, w_train, _, _,_, _, _,X_apred, y_apred = split_train_test_apred(n_experimento,mes_train,
                                                                                                                                        MES_TEST,MES_A_PREDECIR,
                                                                                                                                        SEMILLA,SUBSAMPLEO,feature_subset=cols_drops,n_canaritos=5)
        y_apred_top_models=[]
        for orden_trial , trial in enumerate(top_bp):
            if tipo_bayesiana!="ZLGBM":
                    name_trial = name + f"_TRIAL_{trial}_TOP_{orden_trial}"
                    best_params_i = top_bp[trial]["params"]
                    best_iter_i = top_bp[trial]["best_iter_trial"]
                    logger.info(f"Comienzo del modelo del top del orden {orden_trial} : trial={trial} con hiperparams {best_params_i}")
                    logger.info(f"Comienzo del modelo del top del orden {orden_trial} : trial={trial} con best iter {best_iter_i}")
            else:
                    name_trial = name   
                    logger.info(f"Comienzo del ZLGBM")

            y_predicciones_lista=[]
            for i,semilla in enumerate(semillas):
                if tipo_bayesiana!="ZLGBM":
                    logger.info(f"Comienzo de la semilla numero {semilla} del orden {i} de {len(semillas)} iteraciones para el orden del trial {orden_trial} *****************************************")
                
                else:
                    logger.info(f"Comienzo de la semilla numero {semilla} del orden {i}   *****************************************")
                name_semilla=f"{name_trial}_SEMILLA_{semilla}_final_train"
                if tipo_bayesiana =="OB":
                    model_lgbm = entrenamiento_lgbm(X_train , y_train_binaria,w_train ,best_iter_i,best_params_i ,name_semilla,output_path_models,semilla)
                elif tipo_bayesiana =="ZS":
                    model_lgbm = entrenamiento_lgbm_zs(X_train , y_train_binaria,w_train ,best_iter_i,best_params_i ,name_semilla,output_path_models,semilla)
                elif tipo_bayesiana =="ZLGBM":
                    model_lgbm = entrenamiento_zlgbm(X_train , y_train_binaria  ,name_semilla,output_path_models,semilla)

                # Grafico features importances
                grafico_feature_importance(model_lgbm,X_train,name_semilla,output_path_feat_imp)

                # Predicciones en test para cada modelo
                logger.info(f"Comienzo de la predicciones de apred  : {X_apred['foto_mes'].unique()} para cada modelo")
                y_pred_lgbm=prediccion_test_lgbm(X_apred ,model_lgbm)
                y_predicciones_lista.append(y_pred_lgbm)
            
            if proceso_ppal =="prediccion_final" :
                proceso_experimento = "experimento"
            elif proceso_ppal =="test_prediccion_final":
                proceso_experimento = "test_exp"
            if tipo_bayesiana != "ZLGBM":
                estadisticas_ganancia_file =f"{proceso_experimento}_{numero}_LGBM_{len(semillas)}_SEMILLAS"+ f"_TRIAL_{trial}_TOP_{orden_trial}_MES_TEST_{MES_TEST[-1]}.json"
            else:
                estadisticas_ganancia_file =f"{proceso_experimento}_{numero}_LGBM_{len(semillas)}_SEMILLAS" + f"_MES_TEST_{MES_TEST[-1]}.json"

            file = path_output_exp_umbral+estadisticas_ganancia_file 
            logger.info(f"Comienzo de la carga de las estadisticas de ganancias {file}")            
            # try :
            #     with open(file, "r") as f:
            #         estadisticas_ganancia = json.load(f)
            #     logger.info(f"Carga de los datos umbrales {estadisticas_ganancia_file} exitosa")

            # except Exception as e:
            #     logger.error(f"Error al tratar de cargar umbrales {estadisticas_ganancia_file} por {e}")
            #     raise
            # logger.info("Calculo del cliente optimo mean")
            # cliente_optimo = estadisticas_ganancia["ensamble_semillas"]["cliente"]
            cliente_optimo = 10000
            logger.info(f"Cliente optimo del semillero del trial {trial} = {cliente_optimo}")
            logger.info(f"Comienzo del ensamble del semillero")
            y_pred_matrix = np.vstack(y_predicciones_lista)
            logger.info(f" shape de la matriz con todas las predicciones ensamblado{y_pred_matrix.shape}")
            y_pred_ensamble_trial_i = y_pred_matrix.mean(axis=0)
            logger.info("Fin del ensamblado del semillero ")
            # Predicciones binarias y proba 
            preparacion_ypred_kaggle(y_apred, y_pred_ensamble_trial_i ,cliente_optimo, name_trial ,path_output_prediccion_final)
            # Numero de clientes a enviar
            preparacion_nclientesbajas_zulip(X_apred,y_pred_ensamble_trial_i,cliente_optimo,name_trial ,path_output_prediccion_final)
            if tipo_bayesiana !="ZLGBM":
                y_apred_top_models.append(y_pred_ensamble_trial_i)
        if tipo_bayesiana !="ZLGBM":
            name_final = name + "_ENSAMBLE_FINAL"        
            if proceso_ppal =="prediccion_final" :
                proceso_experimento = "experimento"
            elif proceso_ppal =="test_prediccion_final":
                proceso_experimento = "test_exp"
            estadisticas_ganancia_file =f"{proceso_experimento}_{numero}_LGBM_{len(semillas)}_SEMILLAS"+ f"_ENSAMBLE_FINAL_MES_TEST_{MES_TEST[-1]}_umbral_optimo.json"
            file = path_output_exp_umbral+estadisticas_ganancia_file 

            logger.info(f"Comienzo de la carga de las estadisticas de ganancias {file}")            
            try :
                with open(file, "r") as f:
                    estadisticas_ganancia = json.load(f)
                logger.info(f"Carga de los datos umbrales {estadisticas_ganancia_file} exitosa")

            except Exception as e:
                logger.error(f"Error al tratar de cargar umbrales {estadisticas_ganancia_file} por {e}")
                raise
            logger.info("Calculo del cliente optimo mean")
            cliente_optimo = estadisticas_ganancia["cliente"]
            logger.info(f"Cliente optimo del ensamble de los top models = {cliente_optimo}")
            logger.info(f"Comienzo del ensamble del semillero")
            y_pred_ensamble_modelos_matrix = np.vstack(y_apred_top_models)
            logger.info(f" shape de la matriz con todas las predicciones ensamblado{y_pred_ensamble_modelos_matrix.shape}")
            y_pred_ensamble_final = y_pred_ensamble_modelos_matrix.mean(axis=0)
            preparacion_ypred_kaggle(y_apred, y_pred_ensamble_final ,cliente_optimo, name_final ,path_output_prediccion_final)
            preparacion_nclientesbajas_zulip(X_apred,y_pred_ensamble_final,cliente_optimo,name_final ,path_output_prediccion_final)

logger.info(f">>> Ejecucion finalizada. Revisar logs para mas detalles.")

