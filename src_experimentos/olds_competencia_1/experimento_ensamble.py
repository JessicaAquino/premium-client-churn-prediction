#experimento1.py
# EXPERIMENTO ENSAMBLE 
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
from src.lgbm_train_test import entrenamiento_lgbm,grafico_feature_importance,prediccion_test_lgbm ,umbral_optimo_calc,grafico_curvas_ganancia,evaluacion_public_private , graf_hist_ganancias
## ---------------------------------------------------------Configuraciones Iniciales -------------------------------
## Carga de variables

logger=logging.getLogger(__name__)

def lanzar_experimento_ensamble(fecha:str ,semillas:list[int],proceso_ppal:str ="experimento"): #semillas, si queremos hacer una prediccion final, poner solo una semilla en una lista
    n_semillas = len(semillas)
    name=f"{fecha}_EXPERIMENTO_ENSAMBLE"
    logger.info(f"Comienzo del experimento ENSAMBLE : {name} con {n_semillas} semillas")

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
    df_01 = df[~df["foto_mes"].isin( [MES_02 , MES_03])]
    df_02 = df[~df["foto_mes"].isin([MES_01 , MES_03])]
    df_03 = df[~df["foto_mes"].isin([MES_01 , MES_02])]

      # ## 2. Feature Engineering
    #             #01
    df_01=feature_engineering_ratio(df_01,cols_ratios)
                #02
    df_02=feature_engineering_lag(df_02,cols_lag_delta_max_min_regl,1)
    df_02=feature_engineering_delta(df_02,cols_lag_delta_max_min_regl,1)
    df_02=feature_engineering_ratio(df_02,cols_ratios)
                #03
    df_03=feature_engineering_lag(df_03,cols_lag_delta_max_min_regl,2)
    df_03=feature_engineering_delta(df_03,cols_lag_delta_max_min_regl,2)
    df_03=feature_engineering_ratio(df_03,cols_ratios)
    df_03=feature_engineering_linreg(df_03,cols_lag_delta_max_min_regl)




    #                             ## B - ELIMINACION DE FEATURES
    # ## 1. Contruccion de las columnas
    # feat_imp_file_name='2025-10-05_11-38-34_final_train_lgbm_data_frame_feat_imp.xlsx'
    # feat_imp_file=feat_imp_path+feat_imp_file_name
    # cols_dropear=contrs_cols_dropear_feat_imp(df,feat_imp_file,0.02)
    # ## 2. Feat engin
    # df=feature_engineering_drop_cols(df,cols_dropear)



    #3. spliteo train - test - apred
    df_01 = conversion_binario(df_01)
    X_train_01, y_train_binaria_01,y_train_class_01, w_train_01, X_test_para_01, y_test_binaria, y_test_class, w_test,X_apred_para_01, y_apred = split_train_binario(df_01,[MES_01],MES_TEST,MES_A_PREDECIR)
    df_02 = conversion_binario(df_02)
    X_train_02, y_train_binaria_02,y_train_class_02, w_train_02,X_test_para_02, _, _, _,X_apred_para_02, _= split_train_binario(df_02,[MES_02],MES_TEST,MES_A_PREDECIR)
    df_03 = conversion_binario(df_03)
    X_train_03, y_train_binaria_03,y_train_class_03, w_train_03, X_test_para_03, _, _, _,X_apred_para_03, _ = split_train_binario(df_03,[MES_03],MES_TEST,MES_A_PREDECIR)


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

    best_params = {
    "num_leaves": 127,
    "learning_rate": 0.03564895,
    "min_data_in_leaf": 400,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.9}

## 5. Primer Entrenamiento lgbm con la mejor iteracion y los mejores hiperparametros en [01,02,03] y evaluamos en 04 
    y_pred_sorted_dict={}
    ganancia_acumulada_dict={}
    umbrales_dict={}

    lista_df_ganancias_clientes_por_semilla_n_split_50=[]
    lista_df_ganancias_prob_por_semilla_n_split_50=[]

    lista_df_ganancias_clientes_por_semilla_n_split_1=[]
    lista_df_ganancias_prob_por_semilla_n_split_1=[]
    for i,semilla in enumerate(semillas):
        logger.info(f"Comienzo de la semilla numero {semilla} del orden {i}*****************************************")
        # Entrenamiento de los modelos
        name_1rst_train_01=f"{name}_SEMILLA_{semilla}_1rst_train_01"
        model_lgbm_01 = entrenamiento_lgbm(X_train_01 , y_train_binaria_01,w_train_01 ,best_iter,best_params ,name_1rst_train_01,output_path_models,semilla)
        name_1rst_train_02=f"{name}_SEMILLA_{semilla}_1rst_train_02"
        model_lgbm_02 = entrenamiento_lgbm(X_train_02 , y_train_binaria_02,w_train_02 ,best_iter,best_params ,name_1rst_train_02,output_path_models,semilla)
        name_1rst_train_03=f"{name}_SEMILLA_{semilla}_1rst_train_03"
        model_lgbm_03 = entrenamiento_lgbm(X_train_03 , y_train_binaria_03,w_train_03 ,best_iter,best_params ,name_1rst_train_03,output_path_models,semilla)
    
        # Grafico features importances    
        grafico_feature_importance(model_lgbm_01,X_train_01,name_1rst_train_01,output_path_feat_imp)
        grafico_feature_importance(model_lgbm_02,X_train_02,name_1rst_train_02,output_path_feat_imp)
        grafico_feature_importance(model_lgbm_03,X_train_03,name_1rst_train_03,output_path_feat_imp)
        # Predicciones en test 04 para cada modelo
        y_pred_lgbm_01=prediccion_test_lgbm(X_test_para_01 ,model_lgbm_01)
        y_pred_lgbm_02=prediccion_test_lgbm(X_test_para_02 ,model_lgbm_02)
        y_pred_lgbm_03=prediccion_test_lgbm(X_test_para_03 ,model_lgbm_03)

        y_pred_lgbm = (y_pred_lgbm_01 + y_pred_lgbm_02+y_pred_lgbm_03)/3
        name_1rst_train=f"{name}_SEMILLA_{semilla}_1rst_train_ensamble"


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
        df_long_cliente_semilla_i_n_split_1=evaluacion_public_private(X_test_para_01 ,y_test_class,y_pred_lgbm,"n_cliente",umbrales["cliente"],semilla,1)
        df_long_prob_semilla_i_n_split_1 = evaluacion_public_private(X_test_para_01 ,y_test_class,y_pred_lgbm,"prob",umbrales["umbral_optimo"],semilla,1)

        lista_df_ganancias_clientes_por_semilla_n_split_1.append(df_long_cliente_semilla_i_n_split_1)
        lista_df_ganancias_prob_por_semilla_n_split_1.append(df_long_prob_semilla_i_n_split_1)


        # Evaluacion public private con n_split = 50
        df_long_cliente_semilla_i_n_split_50=evaluacion_public_private(X_test_para_01 ,y_test_class,y_pred_lgbm,"n_cliente",umbrales["cliente"],semilla,50)
        df_long_prob_semilla_i_n_split_50 = evaluacion_public_private(X_test_para_01 ,y_test_class,y_pred_lgbm,"prob",umbrales["umbral_optimo"],semilla,50)

        lista_df_ganancias_clientes_por_semilla_n_split_50.append(df_long_cliente_semilla_i_n_split_50)
        lista_df_ganancias_prob_por_semilla_n_split_50.append(df_long_prob_semilla_i_n_split_50)


    name=f"{fecha}_EXPERIMENTO_ENSAMBLE"
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

# logger.info(f">>> Ejecucion finalizada. Revisar logs para mas detalles. {nombre_log}")

