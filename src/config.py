# src/config.py
import os
import yaml
import logging

logger = logging.getLogger(__name__)

# Ruta del archivo de configuración (ajustá si lo tenés en otro lado)
PATH_CONFIG = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")

try:
    with open(PATH_CONFIG, "r") as f:
        cfg = yaml.safe_load(f)
        fe         = cfg["configuracion_feat_eng"]
        exp        = cfg["configuracion_experimentos"]
        bayes      = cfg["configuracion_bayesiana"]
        canaritos  = cfg["configuracion_canaritos"]
        sem        = cfg["configuracion_semilla"]
        gcp        = cfg["configuracion_gcp"]
        paths      = cfg["configuracion_paths"]
        inp        = paths["path_inputs"]
        out        = paths["path_outputs"]
        out_bayes  = paths["path_outputs_bayesian"]
        out_final  = paths["path_outputs_finales"]
        out_exp    = paths["path_outputs_experimentos"]
        


        COMPETENCIA =cfg["COMPETENCIA"]
        PROCESO_PPAL = cfg["PROCESO_PPAL"]
        SUBSAMPLEO = cfg["SUBSAMPLEO"]


        if COMPETENCIA == 1:
            comp    = cfg["configuracion_competencia_1"]
        elif COMPETENCIA == 2:
            comp    = cfg["configuracion_competencia_2"]


        # ================= Configuración  semillas =================
        SEMILLA    = sem.get("SEMILLA", 773767)
        SEMILLAS   = sem.get("SEMILLAS", [259621, 282917, 413417, 773767, 290827])
        
        
        # =================== Configuracion  feature eng ===================
        N_FE    = fe["N_FE"]
        VENTANA = fe["VENTANA"]
        ORDEN_LAGS = fe["ORDEN_LAGS"]
        
        # =================== Configuracion  experimentos ===================
        N_EXPERIMENTO       = exp["N_EXPERIMENTO"]
        N_SEMILLAS_EXP      = exp["N_SEMILLAS_EXP"]
        TOP_MODELS          = exp["TOP_MODELS"]
        MODEL_EXP           = exp["MODEL_EXP"]

        # =================== Optimización LGBM ===================
        TIPO_BAYESIANA      = bayes.get("TIPO_BAYESIANA")
        N_BAYESIANA         = bayes.get("N_BAYESIANA")
        MODEL_BAY           = bayes.get("MODEL_BAY")
        N_SEMILLAS_BAY      = bayes.get("N_SEMILLAS_BAY")
        N_TRIALS            = bayes.get("N_TRIALS", 35)
        N_BOOSTS            = bayes.get("N_BOOSTS", 1000)
        MES_VAL_BAYESIANA   = bayes.get("MES_VAL_BAYESIANA", 202103) 
        UMBRAL              = bayes.get("UMBRAL", 0.025)
        GANANCIA            = bayes.get("GANANCIA", 780000)
        ESTIMULO            = bayes.get("ESTIMULO", 20000)
        N_FOLDS             = bayes.get("N_FOLDS", 5)

        # =================== Configuracion canaritos ================
        N_CANARITOS             = canaritos.get("N_CANARITOS")
        MIN_DATA_IN_LEAF        = canaritos.get("MIN_DATA_IN_LEAF")
        LEARNING_RATE           = canaritos.get("LEARNING_RATE")
        GRADIENT_BOUND          = canaritos.get("GRADIENT_BOUND")
        NUM_LEAVES              = canaritos.get("NUM_LEAVES")
        FEATURE_FRACTION        = canaritos.get("FEATURE_FRACTION")
        BAGGING_FRACTION        = canaritos.get("BAGGING_FRACTION")
        BAGGING_FREQ            = canaritos.get("BAGGING_FREQ")
        MAX_BIN                 = canaritos.get("MAX_BIN")
        NUM_BOOST_ROUND         = canaritos.get("NUM_BOOST_ROUND")
        EARLY_STOPPING_ROUNDS   = canaritos.get("EARLY_STOPPING_ROUNDS")


        # ---------------- Entorno (GCP vs local) ----------------
        in_gcp = bool(gcp.get("IN_GCP", False))
        GCP_PATH = paths["place_path"]["GCP_PATH"]
        LOCAL_PATH = paths["place_path"]["LOCAL_PATH"]
        if in_gcp:
            PLACE_PATHS = GCP_PATH
        else:
            PLACE_PATHS = LOCAL_PATH

        # ================= Rutas de INPUT / LOG ==================
        PATH_DATA_BASE_DB     =    inp["PATH_DATA_BASE_DB"] # --------------------------------------------<
        PATH_INPUT_DATA       = PLACE_PATHS + inp["PATH_INPUT_DATA"]
        FILE_INPUT_DATA       = PLACE_PATHS + comp["FILE_INPUT_DATA"]
        FILE_INPUT_DATA_CRUDO = PLACE_PATHS + comp["FILE_INPUT_DATA_CRUDO"]
        PATH_LOGS             = PLACE_PATHS + paths["PATH_LOGS"]
        PATH_LOG_GLOBAL       = paths["PATH_LOG_GLOBAL"]

        # ==================== OUTPUTS BASES ======================
        PATH_OUTPUT              = PLACE_PATHS + out["PATH_OUTPUT"]
        PATH_OUTPUT_DATA         = PLACE_PATHS + out["PATH_OUTPUT_DATA"]
        PATH_OUTPUT_BAYESIAN     = PLACE_PATHS + out["PATH_OUTPUT_BAYESIAN"]
        PATH_OUTPUT_FINALES      = PLACE_PATHS + out["PATH_OUTPUT_FINALES"]
        PATH_OUTPUT_EXPERIMENTOS = PLACE_PATHS + out["PATH_OUTPUT_EXPERIMENTOS"]
        PATH_OUTPUT_EDA          = PLACE_PATHS + out["PATH_OUTPUT_EDA"]

        # ============= PATH_OUTPUT_BAYESIAN (detallados) =========
        path_output_bayesian_db         =  out_bayes["PATH_OUTPUT_BAYESIAN_DB"]# --------------------------------------------<
        path_output_bayesian_bestparams = PLACE_PATHS + out_bayes["PATH_OUTPUT_BAYESIAN_BESTPARAMS"]
        path_output_bayesian_best_iter  = PLACE_PATHS + out_bayes["PATH_OUTPUT_BAYESIAN_BEST_ITER"]
        path_output_bayesian_graf       = PLACE_PATHS + out_bayes["PATH_OUTPUT_BAYESIAN_GRAF"]

        # =============== PATH_OUTPUT_FINALES (detallados) ========
        path_output_finales_model        = PLACE_PATHS + out_final["PATH_OUTPUT_FINALES_MODEL"]
        path_output_finales_feat_imp     = PLACE_PATHS + out_final["PATH_OUTPUT_FINALES_FEAT_IMP"]
        path_output_prediccion_final     = PLACE_PATHS + out_final["PATH_OUTPUT_FINALES_PREDICCION_FINAL"]

        # ======= PATH_OUTPUT_EXPERIMENTOS (derivados directos) ===
        path_output_exp_model                  = PLACE_PATHS + out_exp["PATH_OUTPUT_EXP_MODEL"]
        path_output_exp_feat_imp               = PLACE_PATHS + out_exp["PATH_OUTPUT_EXP_FEAT_IMP"]
        path_output_exp_graf_gan_hist_grilla   = PLACE_PATHS + out_exp["PATH_OUTPUT_EXP_GRAF_GAN_HIST_GRILLA"]
        path_output_exp_graf_gan_hist_total    = PLACE_PATHS + out_exp["PATH_OUTPUT_EXP_GRAF_HIST_TOTAL"]
        path_output_exp_graf_gan_hist_semillas = PLACE_PATHS + out_exp["PATH_OUTPUT_EXP_GRAF_HIST_SEMILLA"]
        path_output_exp_graf_curva_ganancia    = PLACE_PATHS + out_exp["PATH_OUTPUT_EXP__GRAF_CURVA_GANANCIA"]
        path_output_exp_umbral                 = PLACE_PATHS + out_exp["PATH_OUTPUT_EXP_UMBRAL"]
        path_output_exp_prediction             = PLACE_PATHS + out_exp["PATH_OUTPUT_EXP_PREDICTION"]
        # ================= MESES ==============
        MES_TRAIN      = comp.get("MES_TRAIN", [202101, 202102, 202103])
        MES_TRAIN_04      = comp.get("MES_TRAIN_04", [202101, 202102, 202103])
        MES_TRAIN_06      = comp.get("MES_TRAIN_06", [202101, 202102, 202103])
        MES_TRAIN_08      = comp.get("MES_TRAIN_08", [202101, 202102, 202103])
        MES_TEST       = comp.get("MES_TEST", [202104])
        MES_A_PREDECIR = comp.get("MES_A_PREDECIR", 202106)
        MES_01         = comp.get("MES_01", 202101)
        MES_02         = comp.get("MES_02", 202102)
        MES_03         = comp.get("MES_03", 202103)
        MES_04         = comp.get("MES_04", 202104)
        MES_05         = comp.get("MES_05", 202105)



except Exception as e:
    logger.error(f"Error al cargar el archivo de configuración: {e}")
    raise
