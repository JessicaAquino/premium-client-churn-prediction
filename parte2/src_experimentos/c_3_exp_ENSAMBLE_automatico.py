#experimento_i.py
# EXPERIMENTO : Ensamble con lgb. 
import pandas as pd
import logging
import json
from functools import reduce
import duckdb


from src.config import *
from src.configuracion_inicial import creacion_df_small
from src.constr_lista_cols import cols_a_dropear_variable_entera,contrs_cols_dropear_por_features_sufijos ,cols_a_dropear_variable_originales_o_percentiles
from src.preprocesamiento import split_train_test_apred
from src.lgbm_train_test import preparacion_nclientesbajas_zulip,entrenamiento_zlgbm,entrenamiento_lgbm,entrenamiento_lgbm_zs,grafico_feature_importance,prediccion_test_lgbm ,calc_estadisticas_ganancia,grafico_curvas_ganancia, grafico_hist_ganancia ,preparacion_ypred_kaggle,preparacion_ytest_proba_kaggle
## ---------------------------------------------------------Configuraciones Iniciales -------------------------------
## Carga de variables

logger=logging.getLogger(__name__)

def lanzar_experimento_lgbm(fecha:str ,semillas:list[int],n_experimento:int,proceso_ppal:str ="experimento"): 
    #"""---------------------- CAMBIAR INPUTS --------------------------------------------------------"""
    numero=n_experimento
    #"""----------------------------------------------------------------------------------------------"""
    n_semillas =50
    name=f"{proceso_ppal}_{numero}_LGBM_{len(semillas)}_SEMILLAS"
    logger.info(f"PROCESO PRINCIPAL ---> {proceso_ppal}")
    logger.info(f"Comienzo del experimento : {name} con {n_semillas} semillas")

    
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
    names_exp_finals_preds=["..comp3_conf3_exp301b","..comp3_conf3_exp302","..comp3_conf3_exp313",".c3_exp101",".c3_exp103"]
    # n_semillas= ["50","50","50","25","5"]
    numero_del_ensamble = "13_"

    from functools import reduce

    if proceso_ppal in ("experimento", "test_exp"):
        
        logger.info("Entro en el proceso experimento !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        if isinstance(MES_TEST, int):
            mes_test = [MES_TEST]
        elif isinstance(MES_TEST, list):
            mes_test = MES_TEST
        else:
            raise ValueError(f"Tipo no soportado para MES_TEST: {type(MES_TEST)}")

        for mt in mes_test:
            logger.info(f"======================== Comienzo del analisis en el mes test : {mt} ==============================")

            lists_df = []
            for i, n_exp in enumerate(names_exp_finals_preds):
                name_file = (
                    path_output_exp_prediction
                    +"outputs_experimentos_outputs_prediction_" +f"experimento_{n_exp}_LGBM_{n_semillas}_SEMILLAS_MES_TEST_{mt}_SEMILLA_ensamble_semillas_fase_testeo"
                    "prediccion_test_proba.csv"
                )
                logger.info(f"name file : {name_file}")
                df_i = pd.read_csv(name_file)
                lists_df.append(df_i)

            dfs_renamed = []
            for i, df in enumerate(lists_df):
                if "Predicted" not in df.columns:
                    raise ValueError(f"No encuentro columna 'Predicted' en el archivo #{i}: cols={df.columns}")

                df_tmp = df[["numero_de_cliente", "Predicted"]].copy()
                df_tmp = df_tmp.rename(columns={"Predicted": f"pred_model_{i}"})
                dfs_renamed.append(df_tmp)

            df_ensamble = reduce(
                lambda left, right: pd.merge(left, right, on="numero_de_cliente", how="inner"),
                dfs_renamed
            )
            logger.info(f"Ensamblado el df con shape : {df_ensamble.shape}")

            cols_pred = [c for c in df_ensamble.columns if c.startswith("pred_model_")]
            df_ensamble["Predicted"] = df_ensamble[cols_pred].mean(axis=1)
            sql_test = f"""
                select numero_de_cliente, clase_ternaria
                from df_completo
                where foto_mes = {mt}
            """
            logger.info(f"sql test query : {sql_test}")

            conn = duckdb.connect(PATH_DATA_BASE_DB)
            test_data = conn.execute(sql_test).df()
            conn.close()

            df_final = pd.merge(df_ensamble, test_data, on="numero_de_cliente", how="inner")

            y_test_class = df_final["clase_ternaria"].to_numpy()
            y_pred_ensamble = df_final["Predicted"].to_numpy()

            semilla = "ensamble"
            name_final = numero_del_ensamble+f"ENSAMBLE_FINAL_MES_TEST_{mt}"
            for n_exp in names_exp_finals_preds:
                name_final = name_final + "_"+n_exp

            guardar_umbral = True
            estadisticas_ganancia, y_pred_sorted, ganancia_acumulada = calc_estadisticas_ganancia(
                y_test_class,
                y_pred_ensamble,
                name_final,
                output_path_umbrales,
                semilla,
                guardar_umbral
            )

            grafico_curvas_ganancia(
                y_pred_sorted,
                ganancia_acumulada,
                estadisticas_ganancia,
                name_final,
                output_path_graf_curva_ganancia
            )

    elif proceso_ppal in ("prediccion_final", "test_prediccion_final"):
        logger.info("======================== Comienzo de la prediccion final ==========================")

        lists_df_bin = []
        lists_df_proba = []

        # Levanto los CSVs por modelo
        for i, n_exp in enumerate(names_exp_finals_preds):
            name_file_bin = (
                path_output_prediccion_final
                + "outputs_finales_outputs_final_prediction_" + f"prediccion_final_{n_exp}_LGBM_{n_semillas}_SEMILLAS_pred_finales_binaria.csv"
            )
            name_file_proba = (
                path_output_prediccion_final
                +"outputs_finales_outputs_final_prediction_"+ f"prediccion_final_{n_exp}_LGBM_{n_semillas}_SEMILLAS_pred_finales_proba.csv"
            )

            logger.info(f"name file bin: {name_file_bin}")
            logger.info(f"name file proba: {name_file_proba}")

            df_i_bin = pd.read_csv(name_file_bin)
            df_i_proba = pd.read_csv(name_file_proba)

            lists_df_bin.append(df_i_bin)
            lists_df_proba.append(df_i_proba)

        #   ENSAMBLE DE PROBAS

        dfs_renamed_proba = []
        for i, df in enumerate(lists_df_proba):
            if "Predicted" in df.columns:
                col_pred = "Predicted"
            elif "predicted" in df.columns:
                col_pred = "predicted"
            else:
                raise ValueError(f"No encuentro columna Predicted/predicted en proba #{i}. Cols: {df.columns}")

            df_tmp_proba = df[["numero_de_cliente", col_pred]].copy()
            df_tmp_proba = df_tmp_proba.rename(columns={col_pred: f"pred_proba_{i}"})
            dfs_renamed_proba.append(df_tmp_proba)

        df_ensamble_proba = reduce(
            lambda left, right: pd.merge(left, right, on="numero_de_cliente", how="inner"),
            dfs_renamed_proba
        )

        cols_pred_proba = [c for c in df_ensamble_proba.columns if c.startswith("pred_proba_")]
        df_ensamble_proba["pred_proba_mean"] = df_ensamble_proba[cols_pred_proba].mean(axis=1)

        #   ENSAMBLE BINARIO (VOTING)
        
        dfs_renamed_bin = []
        for i, df in enumerate(lists_df_bin):
            if "Predicted" in df.columns:
                col_pred_bin = "Predicted"
            elif "predicted" in df.columns:
                col_pred_bin = "predicted"
            else:
                raise ValueError(f"No encuentro columna Predicted/predicted en binaria #{i}. Cols: {df.columns}")

            df_tmp_bin = df[["numero_de_cliente", col_pred_bin]].copy()
            df_tmp_bin = df_tmp_bin.rename(columns={col_pred_bin: f"pred_bin_{i}"})
            dfs_renamed_bin.append(df_tmp_bin)

        df_ensamble_bin = reduce(
            lambda left, right: pd.merge(left, right, on="numero_de_cliente", how="inner"),
            dfs_renamed_bin
        )

        cols_pred_bin = [c for c in df_ensamble_bin.columns if c.startswith("pred_bin_")]
        df_ensamble_bin["votes_sum"] = df_ensamble_bin[cols_pred_bin].sum(axis=1)

        n_models = len(cols_pred_bin)
        df_ensamble_bin["pred_bin_vote"] = (df_ensamble_bin["votes_sum"] >= (n_models / 2)).astype(int)

        # =========================
        #   UNIR PROBA + BINARIO
        # =========================

        df_final_pred = pd.merge(
            df_ensamble_proba[["numero_de_cliente", "pred_proba_mean"]],
            df_ensamble_bin[["numero_de_cliente", "pred_bin_vote"]],
            on="numero_de_cliente",
            how="inner"
        )

        logger.info(f"DF final pred shape: {df_final_pred.shape}")

        # =========================
        #   ENVIAR A ZULIP
        # =========================

        y_pred_proba = df_final_pred["pred_proba_mean"].to_numpy()
        preparacion_nclientesbajas_zulip(
            df_final_pred,
            y_pred_proba,
            umbral_cliente=UMBRAL_CLIENTE,
            name="pred_final_proba",
            output_path=path_output_prediccion_final
        )

        y_pred_bin = df_final_pred["pred_bin_vote"].to_numpy()
        preparacion_nclientesbajas_zulip(
            df_final_pred,
            y_pred_bin,
            umbral_cliente=UMBRAL_CLIENTE,
            name="pred_final_bin",
            output_path=path_output_prediccion_final
        )

            
logger.info(f">>> Ejecucion finalizada. Revisar logs para mas detalles.")

