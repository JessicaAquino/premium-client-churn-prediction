#experimento_i.py
# EXPERIMENTO : Ensamble con lgb y xgb. Es el 8 pero pongo el subsampleo. Eliminaos cprestamos_personales y mprestamos_personales
import numpy as np
import pandas as pd
import logging
import json

from src.config import *
from src.loader import cargar_datos
from src.preprocesamiento import conversion_binario,split_train_test_apred
from src.constr_lista_cols import contruccion_cols
from src.feature_engineering import feature_engineering_lag,feature_engineering_delta,feature_engineering_max_min,feature_engineering_ratio,feature_engineering_linreg,feature_engineering_normalizacion,feature_engineering_drop_cols,feature_engineering_rank
from src.lgbm_train_test import entrenamiento_lgbm,grafico_feature_importance,prediccion_test_lgbm ,umbral_optimo_calc,grafico_curvas_ganancia,evaluacion_public_private , graf_hist_ganancias,preparacion_ypred_kaggle
from src.xgb_train_test import entrenamiento_xgb , grafico_feature_importance_xgb ,prediccion_test_xgb
## ---------------------------------------------------------Configuraciones Iniciales -------------------------------
## Carga de variables

logger=logging.getLogger(__name__)

def lanzar_experimento(fecha:str ,semillas:list[int],n_experimento:int,proceso_ppal:str ="experimento"): 
    #"""---------------------- CAMBIAR INPUTS --------------------------------------------------------"""
    numero=n_experimento
    #"""----------------------------------------------------------------------------------------------"""
    n_semillas = len(semillas)
    name=f"{fecha}_EXPERIMENTO_{numero}_{proceso_ppal}"
    logger.info(f"PROCESO PRINCIPAL ---> {proceso_ppal}")
    logger.info(f"Comienzo del experimento : {name} con {n_semillas} semillas")
    

    # Defini el path de los outputs de los modelos, de los graf de hist gan, de graf curva ganan, de umbrales, de feat import
    if proceso_ppal =="experimento" :
        output_path_models = path_output_exp_model
        output_path_feat_imp = path_output_exp_feat_imp
        output_path_graf_ganancia_hist_semillas=path_output_exp_graf_gan_hist_semillas
        output_path_graf_ganancia_hist_total =path_output_exp_graf_gan_hist_total
        output_path_graf_ganancia_hist_grilla =path_output_exp_graf_gan_hist_grilla
        output_path_graf_curva_ganancia = path_output_exp_graf_curva_ganancia
        output_path_umbrales=path_output_exp_umbral
        logger.info(f"LANZAMIENTO PARA EXPERIMENTO {numero} CON {n_semillas} SEMILLAS")
        
    elif proceso_ppal == "prediccion_final":
        logger.info(f"LANZAMIENTO PARA PREDICCION FINAL CON {n_semillas} SEMILLAS")
        output_path_models = path_output_finales_model
        output_path_feat_imp =path_output_finales_feat_imp
   

    ## 0. load datos
    df=cargar_datos(FILE_INPUT_DATA)
    print(df.head())

    

                            ## A - AGREGADO DE FEATURES

    # 1. Contruccion de las columnas
    cols_a_dropear=["mprestamos_personales","cprestamos_personales"]
    df = feature_engineering_drop_cols(df , cols_a_dropear)
    columnas=contruccion_cols(df)
    cols_lag_delta_max_min_regl=columnas[0]
    cols_ratios=columnas[1]
    
    # 2. Feature Engineering
    # df = feature_engineering_rank(df,["mcuentas_saldo"])
    # df=feature_engineering_drop_cols(df,["mcuentas_saldo"])    
    df=feature_engineering_lag(df,cols_lag_delta_max_min_regl,2)
    df=feature_engineering_delta(df,cols_lag_delta_max_min_regl,2)
    df=feature_engineering_ratio(df,cols_ratios)
    df=feature_engineering_linreg(df,cols_lag_delta_max_min_regl)

                                 ## B - ELIMINACION DE FEATURES
    # 1. Contruccion de las columnas
    # feat_imp_file_name='2025-10-05_11-38-34_final_train_lgbm_data_frame_feat_imp.xlsx'
    # feat_imp_file=feat_imp_path+feat_imp_file_name
    #cols_dropear=contrs_cols_dropear_feat_imp(df,feat_imp_file,0.02)

    # ## 2. Feat engin
    # df=feature_engineering_drop_cols(df,cols_dropear)


    #3. spliteo train - test - apred - Subsampleo
    if proceso_ppal =="prediccion_final":
        MES_TRAIN.append(MES_04)
    df = conversion_binario(df)
    X_train, y_train_binaria,y_train_class, w_train, X_test, y_test_binaria, y_test_class, w_test,X_apred, y_apred = split_train_test_apred(df,MES_TRAIN,MES_TEST,MES_A_PREDECIR,SEMILLA,0.4)

    # 4. Carga de mejores Hiperparametros


    logger.info("Ingreso de hiperparametros de una Bayesiana ya realizada")
        
            
    #"""---------------------- CAMBIAR INPUTS --------------------------------------------------------"""
    bayesiana_fecha_hora_lgbm= '2025-10-29_11-32-15'
    numero_bayesiana_lgbm ='2'
    bayesiana_fecha_hora_xgb = '2025-10-29_11-32-15'
    numero_bayesiana_xgb='2'
    #"""-----------------------------------------------------------------------------------------------"""

    name_best_params_file_lgbm=f"best_params_bayesiana_{numero_bayesiana_lgbm}_lgbm_{bayesiana_fecha_hora_lgbm}.json"
    name_best_iter_file_lgbm=f"best_iter_bayesiana_{numero_bayesiana_lgbm}_lgbm_{bayesiana_fecha_hora_lgbm}.json"

    try:
        with open(path_output_bayesian_bestparams+name_best_params_file_lgbm, "r") as f:
            best_params_lgbm = json.load(f)
            logger.info(f"Correcta carga de los best params del LGBM : {best_params_lgbm}")

        with open(path_output_bayesian_best_iter+name_best_iter_file_lgbm, "r") as f:
            best_iter_lgbm = json.load(f)
            logger.info(f"Correcta carga de la best iter del LGBM : {best_iter_lgbm}")
    except Exception as e:
        logger.error(f"No se pudo encontrar los best params ni best iter del LGBM por el error {e}")
        raise


    name_best_params_file_xgb=f"best_params_bayesiana_{numero_bayesiana_xgb}_xgb_{bayesiana_fecha_hora_xgb}.json"
    name_best_iter_file_xgb=f"best_iter_bayesiana_{numero_bayesiana_xgb}_xgb_{bayesiana_fecha_hora_xgb}.json"

    try:
        with open(path_output_bayesian_bestparams+name_best_params_file_xgb, "r") as f:
            best_params_xgb = json.load(f)
            logger.info(f"Correcta carga de los best params del XGB : {best_params_xgb}")

        with open(path_output_bayesian_best_iter+name_best_iter_file_xgb, "r") as f:
            best_iter_xgb = json.load(f)
            logger.info(f"Correcta carga de la best iter del XGB : {best_iter_xgb}")
    except Exception as e:
        logger.error(f"No se pudo encontrar los best params ni best iter del XGB por el error {e}")
        raise

        
## 5. Primer Entrenamiento lgbm con la mejor iteracion y los mejores hiperparametros en [01,02,03] y evaluamos en 04 

    if proceso_ppal =="experimento":
        logger.info(f"Entro en el proceso experimento !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        y_predicciones_lista=[]
        y_pred_sorted_dict={}
        ganancia_acumulada_dict={}
        umbrales_dict={}

        lista_df_ganancias_clientes_por_semilla_n_split_50=[]
        lista_df_ganancias_prob_por_semilla_n_split_50=[]

        lista_df_ganancias_clientes_por_semilla_n_split_1=[]
        lista_df_ganancias_prob_por_semilla_n_split_1=[]

        for i,semilla in enumerate(semillas):

            logger.info(f"Comienzo de la semilla numero {semilla} del orden {i} de {len(semillas)} iteraciones *****************************************")
            # Entrenamiento de los modelos
            name_1rst_train_lgbm=f"{name}_SEMILLA_{semilla}_1rst_train_lgbm"
            model_lgbm = entrenamiento_lgbm(X_train , y_train_binaria,w_train ,best_iter_lgbm,best_params_lgbm ,name_1rst_train_lgbm,output_path_models,semilla)
            
            name_1rst_train_xgb=f"{name}_SEMILLA_{semilla}_1rst_train_xgb"
            model_xgb= entrenamiento_xgb(X_train, y_train_binaria,w_train ,best_iter_xgb,best_params_xgb ,name_1rst_train_xgb,output_path_models,semilla)
        
            # Grafico features importances
            grafico_feature_importance(model_lgbm,X_train,name_1rst_train_lgbm,output_path_feat_imp)
            grafico_feature_importance_xgb(model_xgb,X_train,name_1rst_train_xgb,output_path_feat_imp)

            # Predicciones en test 04 para cada modelo
            y_pred_lgbm=prediccion_test_lgbm(X_test ,model_lgbm)
            y_pred_xgb=prediccion_test_xgb(X_test ,model_xgb)


            name_1rst_train = f"{name}_SEMILLA_{semilla}_1rst_train_ensamble_xgb_lgbm"
            y_pred_ensamble = (y_pred_lgbm+y_pred_xgb)/2
            y_predicciones_lista.append(y_pred_ensamble)

            # Umbral optimo
            if proceso_ppal == "experimento":
                guardar_umbral = False
            else:
                guardar_umbral=True

            dict_calc_umbrales= umbral_optimo_calc(y_test_class , y_pred_ensamble ,name_1rst_train , output_path_umbrales , semilla, guardar_umbral )
            
            umbrales=dict_calc_umbrales["umbrales"]
            umbrales_dict[semilla] = umbrales 

            y_pred_sorted = dict_calc_umbrales["y_pred_sorted"]
            y_pred_sorted_dict[semilla] = y_pred_sorted

            ganancia_acumulada = dict_calc_umbrales["ganancia_acumulada"]
            ganancia_acumulada_dict[semilla] = ganancia_acumulada


            # Evaluacion public private con n_split = 1 
            df_long_cliente_semilla_i_n_split_1=evaluacion_public_private(X_test ,y_test_class,y_pred_ensamble,"n_cliente",umbrales["cliente"],semilla,1)
            df_long_prob_semilla_i_n_split_1 = evaluacion_public_private(X_test ,y_test_class,y_pred_ensamble,"prob",umbrales["umbral_optimo"],semilla,1)

            lista_df_ganancias_clientes_por_semilla_n_split_1.append(df_long_cliente_semilla_i_n_split_1)
            lista_df_ganancias_prob_por_semilla_n_split_1.append(df_long_prob_semilla_i_n_split_1)


            # Evaluacion public private con n_split = 50
            df_long_cliente_semilla_i_n_split_50=evaluacion_public_private(X_test ,y_test_class,y_pred_ensamble,"n_cliente",umbrales["cliente"],semilla,50)
            df_long_prob_semilla_i_n_split_50 = evaluacion_public_private(X_test ,y_test_class,y_pred_ensamble,"prob",umbrales["umbral_optimo"],semilla,50)

            lista_df_ganancias_clientes_por_semilla_n_split_50.append(df_long_cliente_semilla_i_n_split_50)
            lista_df_ganancias_prob_por_semilla_n_split_50.append(df_long_prob_semilla_i_n_split_50)


        # Creacion de lqs predicciones medias a partir de los ensambles de las semillas
        semilla = "ensamble_semillas"
        logger.info("Comienzo del ensamblado de todas las semillas")
        y_pred_df = np.vstack(y_predicciones_lista)
        logger.info(f" shape de la matriz con todas las predicciones ensamblado{y_pred_df.shape}")
        y_pred_ensamble = y_pred_df.mean(axis=0)
        logger.info("Fin del ensamblado ")
        name_1rst_train = f"{name}_ensambles_semillas_1rst_train_ensamble_xgb_lgbm"
        dict_calc_umbrales= umbral_optimo_calc(y_test_class , y_pred_ensamble ,name_1rst_train , output_path_umbrales , semilla, guardar_umbral )

        umbrales=dict_calc_umbrales["umbrales"]
        umbrales_dict[semilla] = umbrales 

        y_pred_sorted = dict_calc_umbrales["y_pred_sorted"]
        y_pred_sorted_dict[semilla] = y_pred_sorted

        ganancia_acumulada = dict_calc_umbrales["ganancia_acumulada"]
        ganancia_acumulada_dict[semilla] = ganancia_acumulada


        name=f"{fecha}_EXPERIMENTO_{numero}"
        name_1rst_train = f"{name}_1rst_train_ensamble_xgb_lgbm"
        if guardar_umbral == False:
            try:
                with open(output_path_umbrales+f"{name}.json", "w") as f:
                    json.dump(umbrales_dict, f, indent=4)
            except Exception as e:
                logger.error(f"Error al intentar guardar el dict de umbral como json --> {e}")
            logger.info(f"Los datos de umbrales moviles son : {umbrales}")
            logger.info("Fin de la prediccion de umbral movil")

        logger.info("Vamos a graficar la curva de ganancia con la y_pred_ensamble")
        semillas_con_pred_ensamble = y_pred_sorted_dict.keys()
        grafico_curvas_ganancia(y_pred_sorted_dict ,ganancia_acumulada_dict,umbrales_dict,semillas_con_pred_ensamble,name_1rst_train,output_path_graf_curva_ganancia)

        # Graficos de histogramas con las semillas juntas con 1 split
        df_long_cliente_n_split_1 = pd.concat(lista_df_ganancias_clientes_por_semilla_n_split_1,axis=0)
        df_long_prob_n_split_1 = pd.concat(lista_df_ganancias_prob_por_semilla_n_split_1,axis=0)
        graf_hist_ganancias(df_long_cliente_n_split_1,name+"_cliente_nsplits_1", output_path_graf_ganancia_hist_semillas , semillas)
        graf_hist_ganancias(df_long_prob_n_split_1,name+"_prob_nsplits_1", output_path_graf_ganancia_hist_semillas ,semillas)

        # Graficos en un histogramas con todas las semillas y los splits juntos
        df_long_cliente_n_split_50 = pd.concat(lista_df_ganancias_clientes_por_semilla_n_split_50,axis=0)
        df_long_prob_n_split_50 = pd.concat(lista_df_ganancias_prob_por_semilla_n_split_50,axis=0)
        graf_hist_ganancias(df_long_cliente_n_split_50,name+"_cliente_nsplits_50", output_path_graf_ganancia_hist_total , semillas)
        graf_hist_ganancias(df_long_prob_n_split_50,name+"_prob_nsplits_50", output_path_graf_ganancia_hist_total, semillas)

        # Graficos en grillas de histograma, una por cada semilla con sus N particiones
        graf_hist_ganancias(lista_df_ganancias_clientes_por_semilla_n_split_50,name+"_cliente_nsplits_50", output_path_graf_ganancia_hist_grilla , semillas)
        graf_hist_ganancias(lista_df_ganancias_prob_por_semilla_n_split_50,name+"_prob_nsplits_50", output_path_graf_ganancia_hist_grilla,semillas)

    
    elif proceso_ppal =="prediccion_final":
        y_predicciones_lista=[]
        for i,semilla in enumerate(semillas):
            logger.info(f"Comienzo de la semilla numero {semilla} del orden {i} de {len(semillas)} iteraciones *****************************************")
            # Carga del umbral optimo de todas las tiradas : 

    #"""---------------------- CAMBIAR INPUTS --------------------------------------------------------"""
            # fecha_name_umbral='2025-10-10_13-53-26'
            # fecha_name_umbral='TEST_TEST_TEST_TEST'
            fecha_name_umbral='2025-11-02_14-57-33'
            numero_umbral='10'
    #"""----------------------------------------------------------------------------------------------"""
            umbrales_file=f"{fecha_name_umbral}_EXPERIMENTO_{numero_umbral}.json"
            file = path_output_exp_umbral+umbrales_file 
            logger.info(f"Comienzo de la carga de los datos de umbrales {umbrales_file}")            
            try :
                with open(file, "r") as f:
                    data_ganancias = json.load(f)
                logger.info(f"Carga de los datos umbrales {umbrales_file} exitosa")

            except Exception as e:
                logger.error(f"Error al tratar de cargar umbrales {umbrales_file} por {e}")
                raise
            logger.info("Calculo del cliente optimo mean")
            # clientes_optimos = []
            # for semilla_gan in data_ganancias.keys():
            #     clientes_optimos.append(data_ganancias[semilla_gan]["cliente"])
            # cliente_optimo_mean = np.mean(clientes_optimos)

            cliente_optimo_mean = data_ganancias["ensamble_semillas"]["cliente"]


            logger.info(f"Cliente optimo mean = {cliente_optimo_mean}")
            #entrenamiento de los modelos
            logger.info(f"Comienzo de los entrenamientos del modelo  : {X_apred['foto_mes'].unique()} para cada modelo")

            name_final_train_lgbm=f"{name}_SEMILLA_{semilla}_final_train_lgbm"
            model_lgbm = entrenamiento_lgbm(X_train , y_train_binaria,w_train ,best_iter_lgbm,best_params_lgbm ,name_final_train_lgbm,output_path_models,semilla)
            
            name_final_train_xgb=f"{name}_SEMILLA_{semilla}_final_train_xgb"
            model_xgb= entrenamiento_xgb(X_train, y_train_binaria,w_train ,best_iter_xgb,best_params_xgb ,name_final_train_xgb,output_path_models,semilla)
        
            # Grafico features importances
            grafico_feature_importance(model_lgbm,X_train,name_final_train_lgbm,output_path_feat_imp)
            grafico_feature_importance_xgb(model_xgb,X_train,name_final_train_xgb,output_path_feat_imp)

            # Predicciones en test 04 para cada modelo
            logger.info(f"Comienzo de la predicciones de apred  : {X_apred['foto_mes'].unique()} para cada modelo")
            y_pred_lgbm=prediccion_test_lgbm(X_apred ,model_lgbm)
            y_pred_xgb=prediccion_test_xgb(X_apred ,model_xgb)
            logger.info(f"Fin de la prediccion de apred : {X_apred['foto_mes'].unique()} para cada modelo")

            logger.info(f"Comienzo del ensamble de ambos modelos")
            y_pred_ensamble_i = (y_pred_lgbm+y_pred_xgb)/2
            y_predicciones_lista.append(y_pred_ensamble_i)
        logger.info("Comienzo del ensamblado de todas las semillas")
        y_pred_df = np.vstack(y_predicciones_lista)
        logger.info(f" shape de la matriz con todas las predicciones ensamblado{y_pred_df.shape}")
        y_pred_ensamble = y_pred_df.mean(axis=0)
        logger.info("Fin del ensamblado ")
        # Predicciones en test 04 para cada modelo
        y_apred=preparacion_ypred_kaggle(y_apred, y_pred_ensamble ,cliente_optimo_mean , name ,path_output_prediccion_final)


logger.info(f">>> Ejecucion finalizada. Revisar logs para mas detalles.")

