#experimento1.py
# EXPERIMENTO 4 : Ensamble con los 3 meses en cada modelo, uno sin feat eng y el otro con
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import json
import lightgbm as lgb

from src.config import *
from src.loader import cargar_datos
from src.preprocesamiento import conversion_binario,split_train_binario
from src.constr_lista_cols import contruccion_cols
from src.feature_engineering import feature_engineering_lag,feature_engineering_delta,feature_engineering_max_min,feature_engineering_ratio,feature_engineering_linreg,feature_engineering_normalizacion,feature_engineering_drop_cols
from src.lgbm_train_test import entrenamiento_lgbm,grafico_feature_importance,prediccion_test_lgbm ,umbral_optimo_calc,grafico_curvas_ganancia,evaluacion_public_private , graf_hist_ganancias,preparacion_ypred_kaggle
## ---------------------------------------------------------Configuraciones Iniciales -------------------------------
## Carga de variables

logger=logging.getLogger(__name__)

def lanzar_experimento_4(fecha:str ,semillas:list[int],proceso_ppal:str ="experimento"): #semillas, si queremos hacer una prediccion final, poner solo una semilla en una lista
    n_semillas = len(semillas)
    name=f"{fecha}_EXPERIMENTO_4"
    logger.info(f"PROCESO PRINCIPAL ---> {proceso_ppal}")
    logger.info(f"Comienzo del experimento 4 : {name} con {n_semillas} semillas")
    

    # Defini el path de los outputs de los modelos, de los graf de hist gan, de graf curva ganan, de umbrales, de feat import
    if proceso_ppal =="experimento" :
        output_path_models = path_output_exp_model
        output_path_feat_imp = path_output_exp_feat_imp
        output_path_graf_ganancia_hist_semillas=path_output_exp_graf_gan_hist_semillas
        output_path_graf_ganancia_hist_total =path_output_exp_graf_gan_hist_total
        output_path_graf_ganancia_hist_grilla =path_output_exp_graf_gan_hist_grilla
        output_path_graf_curva_ganancia = path_output_exp_graf_curva_ganancia
        output_path_umbrales=path_output_exp_umbral
        logger.info(f"LANZAMIENTO PARA EXPERIMENTO CON {n_semillas} SEMILLAS")
        
    elif proceso_ppal == "prediccion_final":
        logger.info(f"LANZAMIENTO PARA PREDICCION FINAL CON {n_semillas} SEMILLAS")
        output_path_models = model_path
        output_path_feat_imp =feat_imp_path
        output_path_graf_ganancia_hist_semillas=graf_hist_ganancia_semillas_path
        output_path_graf_ganancia_hist_total=graf_hist_ganancia_total_path
        output_path_graf_ganancia_hist_grilla=graf_hist_ganancia_grilla_path
        output_path_graf_curva_ganancia = graf_curva_ganancia_path
        output_path_umbrales=umbrales_path
        # output_path_prediccion_final=

    ## 0. load datos
    df=cargar_datos(PATH_INPUT_DATA)
    print(df.head())

    

                            ## A - AGREGADO DE FEATURES

    ## 1. Contruccion de las columnas
    columnas=contruccion_cols(df)
    cols_lag_delta_max_min_regl=columnas[0]
    cols_ratios=columnas[1]
#############################################################################-----  DESCOMENTAR  ------############################################
    # ## 2. Feature Engineering
    #             #01
  
                #03
    df_c=df.copy()

    df=feature_engineering_lag(df,cols_lag_delta_max_min_regl,2)
    df=feature_engineering_delta(df,cols_lag_delta_max_min_regl,2)
    df=feature_engineering_ratio(df,cols_ratios)
    df=feature_engineering_linreg(df,cols_lag_delta_max_min_regl)



    #                             ## B - ELIMINACION DE FEATURES
    # ## 1. Contruccion de las columnas
    # feat_imp_file_name='2025-10-05_11-38-34_final_train_lgbm_data_frame_feat_imp.xlsx'
    # feat_imp_file=feat_imp_path+feat_imp_file_name
    # cols_dropear=contrs_cols_dropear_feat_imp(df,feat_imp_file,0.02)
    # ## 2. Feat engin
    # df=feature_engineering_drop_cols(df,cols_dropear)



    #3. spliteo train - test - apred
    if proceso_ppal =="prediccion_final":
        MES_TRAIN.append(MES_04)
    df = conversion_binario(df)
    X_train, y_train_binaria,y_train_class, w_train, X_test, y_test_binaria, y_test_class, w_test,X_apred, y_apred = split_train_binario(df,MES_TRAIN,MES_TEST,MES_A_PREDECIR)

    df_c = conversion_binario(df_c)
    X_train_c, _,_, _, X_test_c, _, _, _,X_apred_c, _ = split_train_binario(df,MES_TRAIN,MES_TEST,MES_A_PREDECIR)
   

## 4. Carga de mejores Hiperparametros


    logger.info("Ingreso de hiperparametros de una Bayesiana ya realizada")
        
            
 
    bayesiana_fecha_hora= '2025-10-05_23-29-49'
    # bayesiana_fecha_hora='2025-09-26_17-37-58'

    name_best_params_file=f"best_params_binaria_{bayesiana_fecha_hora}.json"
    name_best_iter_file=f"best_iter_binaria_{bayesiana_fecha_hora}.json"

    try:
        with open(bestparams_path+name_best_params_file, "r") as f:
            best_params = json.load(f)
            logger.info(f"Correcta carga de los best params : {best_params}")

        with open(best_iter_path+name_best_iter_file, "r") as f:
            best_iter = json.load(f)
            logger.info(f"Correcta carga de la best iter : {best_iter}")
    except Exception as e:
        logger.error(f"No se pudo encontrar los best params ni best iter por el error {e}")
        raise

        
## 5. Primer Entrenamiento lgbm con la mejor iteracion y los mejores hiperparametros en [01,02,03] y evaluamos en 04 

    if proceso_ppal =="entrenamiento":
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
            name_1rst_train=f"{name}_SEMILLA_{semilla}_1rst_train"
            model_lgbm = entrenamiento_lgbm(X_train , y_train_binaria,w_train ,best_iter,best_params ,name_1rst_train,output_path_models,semilla)
            
            name_1rst_train_c=f"{name}_SEMILLA_{semilla}_1rst_train_copy"
            model_lgbm_c = entrenamiento_lgbm(X_train_c , y_train_binaria,w_train ,best_iter,best_params ,name_1rst_train_c,output_path_models,semilla)
        
            # Grafico features importances
            grafico_feature_importance(model_lgbm,X_train,name_1rst_train,output_path_feat_imp)
            grafico_feature_importance(model_lgbm_c,X_train_c,name_1rst_train_c,output_path_feat_imp)

            # Predicciones en test 04 para cada modelo
            y_pred_lgbm_o=prediccion_test_lgbm(X_test ,model_lgbm)
            y_pred_lgbm_c=prediccion_test_lgbm(X_test_c ,model_lgbm_c)

            y_pred_lgbm = (y_pred_lgbm_o+y_pred_lgbm_c)/2

            

            # Umbral optimo
            if proceso_ppal == "experimento":
                guardar_umbral = False
            else:
                guardar_umbral=True

            dict_calc_umbrales= umbral_optimo_calc(y_test_class , y_pred_lgbm ,name_1rst_train , output_path_umbrales , semilla, guardar_umbral )
            
            umbrales=dict_calc_umbrales["umbrales"]
            umbrales_dict[semilla] = umbrales 

            y_pred_sorted = dict_calc_umbrales["y_pred_sorted"]
            y_pred_sorted_dict[semilla] = y_pred_sorted

            ganancia_acumulada = dict_calc_umbrales["ganancia_acumulada"]
            ganancia_acumulada_dict[semilla] = ganancia_acumulada


            # Evaluacion public private con n_split = 1 
            df_long_cliente_semilla_i_n_split_1=evaluacion_public_private(X_test ,y_test_class,y_pred_lgbm,"n_cliente",umbrales["cliente"],semilla,1)
            df_long_prob_semilla_i_n_split_1 = evaluacion_public_private(X_test ,y_test_class,y_pred_lgbm,"prob",umbrales["umbral_optimo"],semilla,1)

            lista_df_ganancias_clientes_por_semilla_n_split_1.append(df_long_cliente_semilla_i_n_split_1)
            lista_df_ganancias_prob_por_semilla_n_split_1.append(df_long_prob_semilla_i_n_split_1)


            # Evaluacion public private con n_split = 50
            df_long_cliente_semilla_i_n_split_50=evaluacion_public_private(X_test ,y_test_class,y_pred_lgbm,"n_cliente",umbrales["cliente"],semilla,50)
            df_long_prob_semilla_i_n_split_50 = evaluacion_public_private(X_test ,y_test_class,y_pred_lgbm,"prob",umbrales["umbral_optimo"],semilla,50)

            lista_df_ganancias_clientes_por_semilla_n_split_50.append(df_long_cliente_semilla_i_n_split_50)
            lista_df_ganancias_prob_por_semilla_n_split_50.append(df_long_prob_semilla_i_n_split_50)


        name=f"{fecha}_EXPERIMENTO_4"
        if guardar_umbral == False:
            try:
                with open(output_path_umbrales+f"{name}.json", "w") as f:
                    json.dump(umbrales_dict, f, indent=4)
            except Exception as e:
                logger.error(f"Error al intentar guardar el dict de umbral como json --> {e}")
            logger.info(f"Los datos de umbrales moviles son : {umbrales}")
            logger.info("Fin de la prediccion de umbral movil")

        grafico_curvas_ganancia(y_pred_sorted_dict ,ganancia_acumulada_dict,umbrales_dict,semillas,name_1rst_train,output_path_graf_curva_ganancia)

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


    if proceso_ppal =="prediccion_final":

        logger.info(f"Comienzo de la prediccion final  *****************************************")
        # Carga del umbral optimo de todas las tiradas : 
        umbrales_file="2025-10-09_21-47-44_EXPERIMENTO_4.json"
        path = path_output_exp_umbral
        file = path + umbrales_file 
        logger.info(f"Comienzo de la carga de los datos de umbrales {umbrales_file}")
        try :
            with open(file, "r") as f:
                data_ganancias = json.load(f)
            logger.info(f"Carga de los datos umbrales {umbrales_file} exitosa")

        except Exception as e:
            logger.error(f"Error al tratar de cargar umbrales {umbrales_file} por {e}")
            raise
        logger.info("Calculo del cliente optimo mean")
        clientes_optimos = []
        for semilla in data_ganancias.keys():
            clientes_optimos.append(data_ganancias[semilla]["cliente"])

        cliente_optimo_mean = np.mean(clientes_optimos)
        logger.info(f"Cliente optimo mean = {cliente_optimo_mean}")

        # Predicciones en test 04 para cada modelo
        logger.info(f"Comienzo de la predicciones de apred  : {X_apred['foto_mes'].unique()} para cada modelo")
        y_pred_lgbm_o=prediccion_test_lgbm(X_apred ,model_lgbm)
        y_pred_lgbm_c=prediccion_test_lgbm(X_apred_c ,model_lgbm_c)
        logger.info(f"Fin de la prediccion de apred : {X_apred['foto_mes'].unique()} para cada modelo")
        logger.info(f"Comienzo del ensamble de ambos modelos")
        y_pred_lgbm = (y_pred_lgbm_o+y_pred_lgbm_c)/2
        y_apred=preparacion_ypred_kaggle(y_apred, y_pred_lgbm ,cliente_optimo_mean , name ,prediccion_final_path)


# ## 6. FINAL TRAIN con mejores hiperp, mejor iter y mejor umbral
# name_final_train="final_train"
# # X_train_final= pd.concat([X_train , X_test],axis=0)
# # logger.info(f"meses en train {X_train_final['foto_mes'].unique()}")
# # logger.info(f"train shape {X_train_final.shape}")
# # y_train_binaria_final = pd.concat([y_train_binaria , y_test_binaria],axis=0)
# # w_train_final=pd.concat([w_train,w_test],axis=0)

# MES_TRAIN.append(MES_TEST)
# X_train_final, y_train_binaria_final,y_train_class_final, w_train_final, X_test, y_test_binaria, y_test_class, w_test,X_apred, y_apred = split_train_binario(df,MES_TRAIN,MES_TEST,MES_A_PREDECIR)
# # umbral_optimo=0.03083123681364618 
# umbral_optimo =0.025
# model_lgbm_final = entrenamiento_lgbm(X_train_final , y_train_binaria_final,w_train_final ,best_iter,best_params , fecha,name_final_train)
# grafico_feature_importance(model_lgbm_final,X_train_final,name_final_train,fecha)
# y_apred=X_apred[["numero_de_cliente"]]

# y_apred_final=prediccion_apred(X_apred ,y_apred,model_lgbm_final,umbral_optimo,fecha,comentario)

logger.info(f">>> Ejecucion finalizada. Revisar logs para mas detalles.")

