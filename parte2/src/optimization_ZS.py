import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import polars as pl
from flaml.default import preprocess_and_suggest_hyperparams
import duckdb

from src.config import *
from src.preprocesamiento import undersampling
# from src.loader import convertir_clase_ternaria_a_target

logger = logging.getLogger(__name__)


def _resolve_seed() -> int:
    if isinstance(SEMILLA, list):
        return SEMILLA[0]
    return int(SEMILLA)


def convertir_clase_ternaria_a_target(df: pd.DataFrame, baja_2_1=True) -> pd.DataFrame:
    """
    Convierte clase_ternaria a target binario reemplazando en el mismo atributo:
    - Continua = 0
    y segun los argumentos baja_2_1
    baja_2_1 = true entonces: BAJA+1 y BAJA+2 = 1
    baja_2_1 = false entonces: BAJA+1 = 0 y BAJA+2 = 1

    Args:
        df: DataFrame con columna 'clase_ternaria'
        baja_2_1: Booleano que indica si se considera BAJA+1 como positivo

    Returns:
        pd.DataFrame: DataFrame con clase_ternaria convertida a valores binarios (0, 1)
    """

    # logger.info("Convirtiendo clase_ternaria a target binario")

    # Contar valores originales para logging (antes de modificar)
    n_continua_orig = (df["clase_ternaria"] == "Continua").sum()
    n_baja1_orig = (df["clase_ternaria"] == "BAJA+1").sum()
    n_baja2_orig = (df["clase_ternaria"] == "BAJA+2").sum()
    logger.debug(
        "Distribución original - Continua: %d, BAJA+1: %d, BAJA+2: %d",
        n_continua_orig,
        n_baja1_orig,
        n_baja2_orig,
    )

    # Modificar el DataFrame usando .loc para evitar SettingWithCopyWarning
    if baja_2_1:
        # Convertir clase_ternaria a binario usando numpy.where para mejorar rendimiento
        df.loc[:, "clase_ternaria"] = np.where(
            df["clase_ternaria"] == "Continua", 0, 1
        ).astype(np.int8)
    else:
        # Convertir BAJA+2 a 1, todo lo demás a 0
        df.loc[:, "clase_ternaria"] = np.where(
            df["clase_ternaria"] == "BAJA+2", 1, 0
        ).astype(np.int8)

    # Log de la conversión
    n_ceros = (df["clase_ternaria"] == 0).sum()
    n_unos = (df["clase_ternaria"] == 1).sum()
    logger.debug(
        "Distribución binaria - 0: %d, 1: %d (%.2f%% positivos)",
        n_ceros,
        n_unos,
        (n_unos / (n_ceros + n_unos) * 100) if (n_ceros + n_unos) else 0.0,
    )

    return df

def _split_train_validation() -> Tuple[pd.DataFrame, pd.DataFrame]:
    logger.info("Entra a _split_train_validation")
    if isinstance(MES_VAL_BAYESIANA , list):
        mes_train = [m for m in MES_TRAIN if m not in MES_VAL_BAYESIANA]
    elif isinstance(MES_VAL_BAYESIANA , int):
        mes_train = [m for m in MES_TRAIN if m != MES_VAL_BAYESIANA]
    mes_val = MES_VAL_BAYESIANA
    logger.info(f"Meses train : {mes_train}")
    logger.info(f"Meses val : {mes_val}")
    
    mes_train_sql = f"{mes_train[0]}"
    for m in mes_train[1:]:    
        mes_train_sql += f",{m}"
    sql_train=f"""select * EXCLUDE(clase_peso , clase_binaria)
                from df_completo
                where foto_mes IN ({mes_train_sql})"""
    
    mes_val_sql = f"{mes_val}"
    sql_val=f"""select * EXCLUDE(clase_peso , clase_binaria)
                from df_completo
                where foto_mes = {mes_val_sql}"""
    conn=duckdb.connect(PATH_DATA_BASE_DB)
    df_train = conn.execute(sql_train).df()
    df_val = conn.execute(sql_val).df()
    conn.close() 
    logger.info(f"Tamaño df_train : {df_train.shape}")
    logger.info(f"Tamaño df_val : {df_val.shape}")

    df_train=undersampling(df_train , SUBSAMPLEO,SEMILLA)

    # if isinstance(MES_TRAIN, list):
    #     df_train = df[df["foto_mes"].isin(MES_TRAIN)].copy()
    # else:
    #     df_train = df[df["foto_mes"] == MES_TRAIN].copy()

    # df_val = df[df["foto_mes"] == MES_VAL_BAYESIANA].copy()

    df_train = convertir_clase_ternaria_a_target(df_train, baja_2_1=True)
    df_val = convertir_clase_ternaria_a_target(df_val, baja_2_1=False)

    df_train["clase_ternaria"] = df_train["clase_ternaria"].astype(np.int8)
    df_val["clase_ternaria"] = df_val["clase_ternaria"].astype(np.int8)

    logger.info(f"df_train clase ternaria value coutns : {df_train['clase_ternaria'].value_counts()}")
    logger.info(f"df_val clase ternaria value coutns : {df_val['clase_ternaria'].value_counts()}")
    logger.info(f"Salio de _split_train_validation ")

    return df_train, df_val


def _prepare_matrices(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    feature_subset: Optional[Any] = None,
) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    logger.info(f"Entro a _prepare_matrices")
    if feature_subset is not None:
        X_train = df_train[feature_subset].copy()
        X_val = df_val[feature_subset].copy()
    else:
        X_train = df_train.drop(columns=["clase_ternaria"])
        X_val = df_val.drop(columns=["clase_ternaria"])

    y_train = df_train["clase_ternaria"].to_numpy(dtype=np.int8)
    y_val = df_val["clase_ternaria"].to_numpy(dtype=np.int8)
    logger.info(f"Salio de _prepare_matrices")

    return X_train, y_train, X_val, y_val

# def calcular_ganancia(y_pred:np.ndarray , y_true:np.ndarray):
#         indx_sorted = np.argsort(y_pred)[::-1]
#         ganancias = np.where(y_true ==1,GANANCIA,0) - np.where(y_true==0,ESTIMULO,0)
#         ganancias_sorted = ganancias[indx_sorted]
#         ganancias_acumuladas = np.cumsum(ganancias_sorted)
#         ganancia_maxima = np.max(ganancias_acumuladas)
#         indx_gan_max = np.argmax(ganancias_acumuladas)
#         ganancia_media_meseta = np.mean(ganancias_acumuladas[indx_gan_max-500 : indx_gan_max+500])
#         ganancia_maxima , ganancias_acumuladas

def calcular_ganancia(y_pred, y_true):
    """
    Calcula la ganancia máxima acumulada ordenando las predicciones de mayor a menor.

    Args:
        y_true: Valores reales (0 o 1)
        y_pred: Predicciones (probabilidades o scores continuos)

    Returns:
        tuple[float, np.ndarray]: Ganancia máxima acumulada y la serie acumulada completa.
    """
    logger.info("Entro a calcular_ganancia")
    def _to_polars_series(
        values, name: str, dtype: pl.DataType | None = None
    ) -> pl.Series:
        logger.info("Entro a _to_polars_series")
        # Convertir a lista primero para evitar dtype 'object'
        if isinstance(values, pl.Series):
            series = pl.Series(name, values.to_list())
        elif isinstance(values, pd.Series):
            series = pl.Series(name, values.to_list())
        else:
            # Convertir iterable a lista si no lo es ya
            if not isinstance(values, (list, tuple)):
                values = list(values)
            series = pl.Series(name, values)

        if dtype is not None:
            try:
                series = series.cast(dtype, strict=False)
            except pl.ComputeError:
                # Fallback si no puede castear (ej. Object a Int32)
                series = series.cast(pl.Float64, strict=False)
        logger.info("Saliendo de _to_polars_series")
        return series

    y_true_series = _to_polars_series(y_true, "y_true", dtype=pl.Float64)
    y_pred_series = _to_polars_series(y_pred, "y_pred_proba", dtype=pl.Float64)

    if y_true_series.is_empty() or y_pred_series.is_empty():
        logger.debug("Ganancia calculada: 0 (datasets vacíos)")
        return 0.0, np.array([], dtype=float)

    if y_true_series.len() != y_pred_series.len():
        raise ValueError("y_true y y_pred deben tener la misma longitud")

    acumulado_df = (
        pl.DataFrame({"y_true": y_true_series, "y_pred_proba": y_pred_series})
        .sort("y_pred_proba", descending=True)
        .with_columns(
            [
                # Calcular ganancia individual para cada registro (forzar i64 para evitar overflow)
                pl.when(pl.col("y_true").round(0) == 1.0)
                .then(pl.lit(GANANCIA, dtype=pl.Int64))
                .otherwise(pl.lit(-ESTIMULO, dtype=pl.Int64))
                .alias("ganancia_individual")
            ]
        )
        .with_columns(
            [
                # Ganancia acumulada: suma acumulativa de ganancias individuales
                pl.col("ganancia_individual")
                .cum_sum()
                .alias("ganancia_acumulada")
            ]
        )
    )

    ganancia_acumulada_series = acumulado_df["ganancia_acumulada"]
    ganancia_total = ganancia_acumulada_series.max()
    # Evitar overflow: si supera int32, devolver como float
    if ganancia_total > 2_147_483_647:
        ganancia_total = float(ganancia_total)
    ganancias_acumuladas = ganancia_acumulada_series.to_numpy()
    indx_gan_max = int(np.argmax(ganancias_acumuladas))
    ganancia_media_meseta = np.mean(ganancias_acumuladas[indx_gan_max-500:indx_gan_max+500])

    logger.info(f"Ganancia calculada: {ganancia_total:,.0f}/ ganancia media_meseta = {ganancia_media_meseta}")
    # f"(GANANCIA_ACIERTO={GANANCIA}, COSTO_ESTIMULO={ESTIMULO})")

    return ganancia_media_meseta, ganancias_acumuladas

def _calcular_ganancia_desde_probabilidades(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Tuple[float, float]:
    """
    Calcula ganancia máxima y umbral sugerido usando la función centralizada.

    Args:
        y_true: Valores reales (0 o 1)
        y_pred: Predicciones (probabilidades)

    Returns:
        Tuple[float, float]: (ganancia_maxima, umbral_sugerido)
    """
    if y_pred.ndim > 1:
        y_pred = y_pred[:, 1]

    # Usar función centralizada para calcular ganancia
    ganancia_maxima, ganancias_acumuladas = calcular_ganancia(y_pred=y_pred, y_true=y_true)

    # Calcular umbral sugerido
    if ganancias_acumuladas.size > 0:
        # Ordenar predicciones de mayor a menor
        orden = np.argsort(y_pred)[::-1]
        y_pred_sorted = y_pred[orden]
        idx_max = int(np.argmax(ganancias_acumuladas))
        umbral_sugerido = float(y_pred_sorted[idx_max])
    else:
        umbral_sugerido = 0.5

    return ganancia_maxima, umbral_sugerido


def preparar_datos_zero_shot(
    feature_subset: Optional[Any] = None,
) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    df_train, df_val = _split_train_validation()
    return _prepare_matrices(df_train, df_val, feature_subset=feature_subset)


def _sugerir_y_entrenar_con_flaml(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
) -> Tuple[Dict[str, Any], Dict[str, Any], np.ndarray, np.ndarray]:
    (
        hyperparams,
        estimator_class,
        X_train_transformed,
        y_train_transformed,
        feature_transformer,
        label_transformer,
    ) = preprocess_and_suggest_hyperparams("classification", X_train, y_train, "lgbm")

    estimator_kwargs = dict(hyperparams)
    estimator_kwargs.setdefault("random_state", _resolve_seed())
    estimator_kwargs.setdefault("n_jobs", -1)

    modelo = estimator_class(**estimator_kwargs)
    modelo.fit(X_train_transformed, y_train_transformed)

    if feature_transformer is not None:
        X_val_transformed = feature_transformer.transform(X_val)
    else:
        X_val_transformed = X_val

    if hasattr(modelo, "predict_proba"):
        proba_val = modelo.predict_proba(X_val_transformed)[:, 1]
    else:
        # Fallback a decision_function si no hay predict_proba
        proba_val = modelo.predict(X_val_transformed)

    if label_transformer is not None:
        y_val_transformed = label_transformer.transform(y_val)
    else:
        y_val_transformed = y_val

    return hyperparams, modelo.get_params(), np.asarray(proba_val), np.asarray(y_val_transformed)


def _construir_parametros_lightgbm(
    hyperparams: Dict[str, Any],
    modelo_params: Dict[str, Any],
) -> Dict[str, Any]:
    rename_map = {
        "n_estimators": "num_iterations",
        "subsample": "bagging_fraction",
        "colsample_bytree": "feature_fraction",
        "min_child_samples": "min_data_in_leaf",
        "n_jobs": "num_threads",
        "random_state": "seed",
    }

    combinados = {**modelo_params, **hyperparams}
    resultado: Dict[str, Any] = {}

    for clave, valor in combinados.items():
        nueva_clave = rename_map.get(clave, clave)
        resultado[nueva_clave] = valor

    resultado["objective"] = "binary"
    resultado["metric"] = "None"
    resultado["verbose"] = -1
    resultado["verbosity"] = -1
    resultado["seed"] = int(resultado.get("seed", _resolve_seed()))

    if "bagging_fraction" not in resultado and "subsample" in combinados:
        resultado["bagging_fraction"] = combinados["subsample"]
    if "feature_fraction" not in resultado and "colsample_bytree" in combinados:
        resultado["feature_fraction"] = combinados["colsample_bytree"]

    # Remover llaves no soportadas por LightGBM nativo
    for clave in ["n_estimators", "subsample", "colsample_bytree", "n_jobs", "random_state"]:
        resultado.pop(clave, None)

    return resultado


def _persistir_resultados(
    archivo_base: str,
    params_flaml: Dict[str, Any],
    params_lightgbm: Dict[str, Any],
    ganancia_validacion: float,
    umbral_sugerido: float,
    proba_val: np.ndarray,
) -> Dict[str, str]:
    # os.makedirs("resultados", exist_ok=True)

    iter_path = os.path.join(path_output_bayesian_bestparams, f"best_params{archivo_base}.json")
    best_path = os.path.join(path_output_bayesian_bestparams, f"best_params{archivo_base}_archivo_completo.json")

    # Obtener número de trial (basado en cantidad de registros existentes)
    if os.path.exists(iter_path):
        try:
            with open(iter_path, "r", encoding="utf-8") as f:
                contenido_existente = json.load(f)
            if not isinstance(contenido_existente, list):
                contenido_existente = []
            trial_number = len(contenido_existente)
        except json.JSONDecodeError:
            contenido_existente = []
            trial_number = 0
    else:
        contenido_existente = []
        trial_number = 0

    # Preparar configuración
    configuracion = {
        "semilla": SEMILLA if isinstance(SEMILLA, list) else [SEMILLA],
        "mes_train": MES_TRAIN if isinstance(MES_TRAIN, list) else [MES_TRAIN],
    }

    # Crear registro en el formato solicitado
    registro = {
        "trial_number": trial_number,
        "params": params_lightgbm,
        "value": float(ganancia_validacion),
        "datetime": datetime.now().isoformat(),
        "state": "COMPLETE",
        "configuracion": configuracion,
    }

    contenido_existente.append(registro)

    with open(iter_path, "w", encoding="utf-8") as f:
        json.dump(contenido_existente, f, indent=2)

    with open(best_path, "w", encoding="utf-8") as f:
        json.dump({"params_lightgbm": params_lightgbm, "params_flaml": params_flaml}, f, indent=2)

    return {"iteraciones": iter_path, "best_params": best_path}


def optimizar_zero_shot(archivo_base: str,
    feature_subset: Optional[Any] = None
) -> Dict[str, Any]:
    if archivo_base is None:
        archivo_base = STUDY_NAME
    """
    Descripción:
    Optimiza los hiperparámetros de un modelo LightGBM usando FLAML para un problema de clasificación binaria.

    Args:
        df: DataFrame con todos los datos
        feature_subset: Subconjunto de características a usar (opcional)
        archivo_base: Nombre base del archivo (si es None, usa el de config.yaml)
    Returns:
        Dict[str, Any]: Diccionario con los mejores parámetros encontrados
    """
    X_train, y_train, X_val, y_val = preparar_datos_zero_shot( feature_subset)

    (
        hyperparams,
        modelo_params,
        proba_val,
        y_val_transformed,
    ) = _sugerir_y_entrenar_con_flaml(X_train, y_train, X_val, y_val)

    ganancia_val, umbral_sugerido = _calcular_ganancia_desde_probabilidades(
        y_val_transformed.astype(np.int32),
        np.clip(proba_val, 0.0, 1.0),
    )

    params_lightgbm = _construir_parametros_lightgbm(hyperparams, modelo_params)

    paths = _persistir_resultados(
        archivo_base,
        hyperparams,
        params_lightgbm,
        ganancia_val,
        umbral_sugerido,
        proba_val,
    )

    logger.info(
        "FLAML Zero-Shot - Ganancia VALID=%s | Umbral=%s",
        f"{ganancia_val:,.0f}",
        f"{umbral_sugerido:.4f}",
    )

    return {
        "ganancia_validacion": ganancia_val,
        "umbral_sugerido": umbral_sugerido,
        "best_params_lightgbm": params_lightgbm,
        "best_params_flaml": hyperparams,
        "paths": paths,
    }