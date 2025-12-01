"""
R_to_py MEJORADO: Conversión completa de workflow R a Python
Workflow con 3 etapas: Train (201901-202102) -> Test (202104, 202106) -> Final (202108)

MEJORAS INCORPORADAS:
- Canaritos con Polars (más eficiente, canaritos al principio)
- Guardado de cada modelo del ensemble individualmente
- Predicciones detalladas (prob + Predicted + metadata)
- Archivo Kaggle simple

Autor: Data Scientist Junior
Fecha: 2025-11-14
"""

import pandas as pd
import numpy as np
import polars as pl
import lightgbm as lgb
import gc
import os
import logging
import json
from datetime import datetime
from sklearn.utils import resample
import warnings
warnings.filterwarnings('ignore')

# Configurar logging básico (se reconfigurará en main con archivo)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURACIÓN - DEFINIR TODO AQUÍ
# ============================================================================

# -----------------------------
# Parámetros del negocio
# -----------------------------
COSTO_ESTIMULO = 20000
GANANCIA_ACIERTO = 780000

# -----------------------------
# Experimento
# -----------------------------
EXPERIMENTO = "apo-506"
SEMILLA_PRIMIGENIA = 102191
APO = 1
KSEMILLERIO = 1

# -----------------------------
# Dataset
# -----------------------------
DATASET_PATH = "~/datasets/competencia_02_crudo.csv.gz"

# -----------------------------
# Periodos (Estructura de 3 etapas)
# -----------------------------
# TRAIN: Todos los meses desde 201901 hasta 202102
FOTO_MES_TRAIN_INICIO = 201901
FOTO_MES_TRAIN_FIN = 202102

# TEST: Dos meses de validación
FOTO_MES_TEST_1 = 202104
FOTO_MES_TEST_2 = 202106

# FINAL: Predicción final
FOTO_MES_FINAL = 202108

# -----------------------------
# Semillas
# -----------------------------
SEMILLAS_EXPERIMENTO = 1    # Para testing/optimización (1 semilla, sin ensamble)
SEMILLAS_FINAL = 1          # Para predicción final (1 semilla, sin ensamble)

# -----------------------------
# Feature Engineering
# -----------------------------
QCANARITOS = 5  # Cantidad de variables aleatorias (canaritos) - dejar en 5

# Lags y Deltas
FEATURE_ENGINEERING_LAGS = True  # Activar/desactivar lags y deltas
LAGS_ORDEN = [1, 2]  # Órdenes de lags a crear (1 y 2)

# -----------------------------
# Undersampling
# -----------------------------
UNDERSAMPLING = True
UNDERSAMPLING_RATIO = 0.05  # Proporción de clase mayoritaria a mantener (0.1 = 10%)

# -----------------------------
# LightGBM - Parámetros (estilo zlightgbm)
# -----------------------------
MIN_DATA_IN_LEAF = 20
LEARNING_RATE = 1.0
GRADIENT_BOUND = 0.1
NUM_LEAVES = 300
FEATURE_FRACTION = 0.8
BAGGING_FRACTION = 0.8
BAGGING_FREQ = 5
MAX_BIN = 31  # Reducir para ahorrar memoria
NUM_BOOST_ROUND = 1000
EARLY_STOPPING_ROUNDS = 200

# -----------------------------
# Cortes para evaluar
# -----------------------------
CORTES = [8000, 8500, 9000, 9500, 10000, 10500, 11000, 
          11500, 12000, 12500, 13000, 13500, 14000, 14500, 15000]

# -----------------------------
# Rutas
# -----------------------------
BASE_PATH = "./exp"


# ============================================================================
# FUNCIÓN DE GANANCIA CON POLARS
# ============================================================================

def calcular_ganancia(y_pred, y_true):
    """
    Calcula la ganancia máxima acumulada ordenando las predicciones de mayor a menor.
    
    Args:
        y_true: Valores reales (0 o 1)
        y_pred: Predicciones (probabilidades o scores continuos)
        
    Returns:
        tuple[float, np.ndarray]: Ganancia máxima acumulada y la serie acumulada completa.
    """
    def _to_polars_series(
        values, name: str, dtype: pl.DataType | None = None
    ) -> pl.Series:
        """Convierte valores a serie de Polars"""
        if isinstance(values, pl.Series):
            series = pl.Series(name, values.to_list())
        elif isinstance(values, pd.Series):
            series = pl.Series(name, values.to_list())
        else:
            if not isinstance(values, (list, tuple)):
                values = list(values)
            series = pl.Series(name, values)
        
        if dtype is not None:
            try:
                series = series.cast(dtype, strict=False)
            except pl.ComputeError:
                series = series.cast(pl.Float64, strict=False)
        
        return series
    
    # Convertir a series de Polars
    y_true_series = _to_polars_series(y_true, "y_true", dtype=pl.Float64)
    y_pred_series = _to_polars_series(y_pred, "y_pred_proba", dtype=pl.Float64)
    
    # Validaciones
    if y_true_series.is_empty() or y_pred_series.is_empty():
        logger.debug("Ganancia calculada: 0 (datasets vacíos)")
        return 0.0, np.array([], dtype=float)
    
    if y_true_series.len() != y_pred_series.len():
        raise ValueError("y_true y y_pred deben tener la misma longitud")
    
    # Calcular ganancia
    acumulado_df = (
        pl.DataFrame({"y_true": y_true_series, "y_pred_proba": y_pred_series})
        .sort("y_pred_proba", descending=True)
        .with_columns([
            pl.when(pl.col("y_true").round(0) == 1.0)
            .then(pl.lit(GANANCIA_ACIERTO, dtype=pl.Int64))
            .otherwise(pl.lit(-COSTO_ESTIMULO, dtype=pl.Int64))
            .alias("ganancia_individual")
        ])
        .with_columns([
            pl.col("ganancia_individual")
            .cum_sum()
            .alias("ganancia_acumulada")
        ])
    )
    
    ganancia_acumulada_series = acumulado_df["ganancia_acumulada"]
    ganancia_total = ganancia_acumulada_series.max()
    
    if ganancia_total > 2_147_483_647:
        ganancia_total = float(ganancia_total)
    
    ganancias_acumuladas = ganancia_acumulada_series.to_numpy()
    
    # Logging desactivado para evitar spam (se llama muchas veces en testing)
    # logger.info(f"Ganancia calculada: {ganancia_total:,.0f}")
    
    return ganancia_total, ganancias_acumuladas


def ganancia_lgb_binary(y_pred, y_true):
    """
    Función de ganancia para LightGBM en clasificación binaria.
    Compatible con callbacks de LightGBM.
    
    Args:
        y_pred: Predicciones de probabilidad del modelo
        y_true: Dataset de LightGBM con labels verdaderos
        
    Returns:
        tuple: (eval_name, eval_result, is_higher_better)
    """
    y_true_labels = y_true.get_label()
    y_pred_binary = (y_pred > 0.025).astype(int)
    ganancia_total, _ = calcular_ganancia(y_pred=y_pred_binary, y_true=y_true_labels)
    return "ganancia", ganancia_total, True


# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def limpiar_memoria():
    """Limpia la memoria RAM"""
    gc.collect()


def crear_directorio(path):
    """Crea un directorio si no existe"""
    os.makedirs(path, exist_ok=True)


def crear_directorio_experimento():
    """
    Crea directorio del experimento con timestamp
    Retorna la ruta completa
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_path = os.path.join(BASE_PATH, f"{EXPERIMENTO}_{timestamp}")
    crear_directorio(exp_path)
    logger.info(f"Directorio del experimento: {exp_path}")
    return exp_path


def configurar_logging_con_archivo(exp_path):
    """
    Configura logging para guardar en archivo además de consola
    
    Args:
        exp_path: Ruta del directorio del experimento
    """
    log_file = os.path.join(exp_path, "experimento.log")
    
    # Crear handler para archivo
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # Crear handler para consola
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formato
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Configurar logger raíz
    root_logger = logging.getLogger()
    root_logger.handlers.clear()  # Limpiar handlers previos
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    root_logger.setLevel(logging.INFO)
    
    logger.info(f"Logging configurado. Guardando en: {log_file}")
    
    return log_file


def guardar_configuracion(exp_path):
    """
    Guarda todos los parámetros configurados en un archivo JSON
    """
    config = {
        "metadata": {
            "experimento": EXPERIMENTO,
            "timestamp": datetime.now().isoformat(),
            "fecha_ejecucion": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "negocio": {
            "costo_estimulo": COSTO_ESTIMULO,
            "ganancia_acierto": GANANCIA_ACIERTO
        },
        "dataset": {
            "path": DATASET_PATH
        },
        "periodos": {
            "train_inicio": FOTO_MES_TRAIN_INICIO,
            "train_fin": FOTO_MES_TRAIN_FIN,
            "test_1": FOTO_MES_TEST_1,
            "test_2": FOTO_MES_TEST_2,
            "final": FOTO_MES_FINAL
        },
        "semillas": {
            "semilla_primigenia": SEMILLA_PRIMIGENIA,
            "semillas_experimento": SEMILLAS_EXPERIMENTO,
            "semillas_final": SEMILLAS_FINAL,
            "ksemillerio": KSEMILLERIO
        },
        "feature_engineering": {
            "qcanaritos": QCANARITOS,
            "lags_enabled": FEATURE_ENGINEERING_LAGS,
            "lags_orden": LAGS_ORDEN
        },
        "undersampling": {
            "enabled": UNDERSAMPLING,
            "ratio": UNDERSAMPLING_RATIO
        },
        "lightgbm": {
            "min_data_in_leaf": MIN_DATA_IN_LEAF,
            "learning_rate": LEARNING_RATE,
            "gradient_bound": GRADIENT_BOUND,
            "num_leaves": NUM_LEAVES,
            "feature_fraction": FEATURE_FRACTION,
            "bagging_fraction": BAGGING_FRACTION,
            "bagging_freq": BAGGING_FREQ,
            "max_bin": MAX_BIN,
            "num_boost_round": NUM_BOOST_ROUND,
            "early_stopping_rounds": EARLY_STOPPING_ROUNDS
        },
        "cortes": CORTES,
        "apo": APO
    }
    
    config_path = os.path.join(exp_path, "configuracion.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    
    logger.info(f"Configuración guardada en: {config_path}")
    return config_path


def generar_semillas(semilla_base, cantidad):
    """Genera una lista de semillas determinísticas"""
    np.random.seed(semilla_base)
    return np.random.randint(1, 2**31-1, size=cantidad).tolist()


def generar_rango_meses(inicio, fin):
    """
    Genera lista de meses en formato YYYYMM entre inicio y fin.
    Ejemplo: generar_rango_meses(201901, 201903) -> [201901, 201902, 201903]
    """
    meses = []
    anio_ini = inicio // 100
    mes_ini = inicio % 100
    anio_fin = fin // 100
    mes_fin = fin % 100
    
    anio_actual = anio_ini
    mes_actual = mes_ini
    
    while (anio_actual * 100 + mes_actual) <= (anio_fin * 100 + mes_fin):
        meses.append(anio_actual * 100 + mes_actual)
        mes_actual += 1
        if mes_actual > 12:
            mes_actual = 1
            anio_actual += 1
    
    return meses


# ============================================================================
# PREPROCESAMIENTO
# ============================================================================

def calcular_clase_ternaria(df):
    """
    Calcula la clase_ternaria según la permanencia del cliente
    """
    logger.info("Calculando clase_ternaria...")
    
    df['periodo0'] = (df['foto_mes'] // 100) * 12 + (df['foto_mes'] % 100)
    df = df.sort_values(['numero_de_cliente', 'periodo0']).reset_index(drop=True)
    
    df['periodo1'] = df.groupby('numero_de_cliente')['periodo0'].shift(-1)
    df['periodo2'] = df.groupby('numero_de_cliente')['periodo0'].shift(-2)
    
    periodo_ultimo = df['periodo0'].max()
    periodo_anteultimo = periodo_ultimo - 1
    
    df['clase_ternaria'] = 'CONTINUA'
    
    mask_baja1 = (df['periodo0'] < periodo_ultimo) & \
                 (df['periodo1'].isna() | (df['periodo0'] + 1 < df['periodo1']))
    df.loc[mask_baja1, 'clase_ternaria'] = 'BAJA+1'
    
    mask_baja2 = (df['periodo0'] < periodo_anteultimo) & \
                 (df['periodo0'] + 1 == df['periodo1']) & \
                 (df['periodo2'].isna() | (df['periodo0'] + 2 < df['periodo2']))
    df.loc[mask_baja2, 'clase_ternaria'] = 'BAJA+2'
    
    df = df.drop(['periodo0', 'periodo1', 'periodo2'], axis=1)
    
    logger.info("Distribución de clases por periodo:")
    dist = df.groupby(['foto_mes', 'clase_ternaria']).size().reset_index(name='count')
    for _, row in dist.head(20).iterrows():
        logger.info(f"  {row['foto_mes']}: {row['clase_ternaria']} = {row['count']}")
    
    return df


# ============================================================================
# CANARITOS CON POLARS (MEJORADO)
# ============================================================================

def agregar_canaritos_polars(df, num_canaritos=None, semilla=None):
    """
    Agrega variables aleatorias (canaritos) AL PRINCIPIO del dataset usando Polars.
    Esta es la versión optimizada del trainer_zlgbm.
    
    Args:
        df: DataFrame de pandas
        num_canaritos: Cantidad de canaritos a agregar
        semilla: Semilla para reproducibilidad
    
    Returns:
        DataFrame de pandas con canaritos al principio
    """
    if num_canaritos is None:
        num_canaritos = QCANARITOS
    if semilla is None:
        semilla = SEMILLA_PRIMIGENIA
        
    logger.info(f"Agregando {num_canaritos} canaritos con Polars (AL PRINCIPIO)...")
    
    # Convertir a Polars
    df_pl = pl.from_pandas(df)
    original_cols = df_pl.columns
    num_filas = df_pl.height
    
    # Nombres de canaritos
    canary_cols = [f"canarito_{i}" for i in range(1, num_canaritos + 1)]
    
    # Generar TODA la matriz de canaritos de una vez (eficiente)
    np.random.seed(semilla)
    rand_matrix = np.random.rand(num_filas, num_canaritos)
    
    # Agregar columnas de canaritos
    for i, name in enumerate(canary_cols):
        df_pl = df_pl.with_columns(pl.Series(name, rand_matrix[:, i]))
    
    # IMPORTANTE: Poner los canaritos AL PRINCIPIO (como en zLightGBM)
    df_pl = df_pl.select(canary_cols + original_cols)
    
    # Convertir de vuelta a pandas
    df_result = df_pl.to_pandas()
    
    logger.info(f"  ✓ {num_canaritos} canaritos agregados al principio")
    logger.info(f"  ✓ Primeras columnas: {list(df_result.columns[:min(10, len(df_result.columns))])}")
    
    return df_result


# ============================================================================
# FEATURE ENGINEERING: LAGS Y DELTAS
# ============================================================================

def agregar_lags_y_deltas(df, ordenes=None):
    """
    Agrega lags y deltas (diferencias) de variables históricas
    
    Args:
        df: DataFrame con los datos
        ordenes: Lista de órdenes de lags a crear (ej: [1, 2])
    
    Returns:
        DataFrame con lags y deltas agregados
    """
    if ordenes is None:
        ordenes = LAGS_ORDEN
    
    if not FEATURE_ENGINEERING_LAGS:
        logger.info("Feature engineering de lags/deltas desactivado")
        return df
    
    logger.info(f"Agregando lags y deltas (órdenes: {ordenes})...")
    inicio = datetime.now()
    
    # Ordenar por cliente y periodo
    df = df.sort_values(['numero_de_cliente', 'foto_mes']).reset_index(drop=True)
    
    # Identificar columnas lagueables
    # Todo es lagueable MENOS: numero_de_cliente, foto_mes, clase_ternaria, canaritos
    cols_excluir = ['numero_de_cliente', 'foto_mes', 'clase_ternaria']
    # Excluir canaritos (pueden tener nombre canarito{i} o canarito_{i})
    cols_excluir += [col for col in df.columns if col.startswith('canarito')]
    
    cols_lagueables = [col for col in df.columns if col not in cols_excluir]
    
    logger.info(f"  Columnas lagueables: {len(cols_lagueables)}")
    logger.info(f"  Órdenes de lag: {ordenes}")
    
    # Crear lags para cada orden
    for orden in ordenes:
        logger.info(f"  Creando lags de orden {orden}...")
        
        # Crear lags usando groupby + shift
        for col in cols_lagueables:
            nombre_lag = f'{col}_lag{orden}'
            df[nombre_lag] = df.groupby('numero_de_cliente')[col].shift(orden)
        
        # Limpiar memoria después de cada orden
        limpiar_memoria()
    
    # Crear deltas (diferencias)
    logger.info(f"  Creando deltas...")
    for orden in ordenes:
        for col in cols_lagueables:
            nombre_delta = f'{col}_delta{orden}'
            nombre_lag = f'{col}_lag{orden}'
            
            # Delta = valor actual - valor lag
            df[nombre_delta] = df[col] - df[nombre_lag]
        
        # Limpiar memoria después de cada orden
        limpiar_memoria()
    
    # Contar features creados
    n_lags = len(cols_lagueables) * len(ordenes)
    n_deltas = len(cols_lagueables) * len(ordenes)
    n_total = n_lags + n_deltas
    
    duracion = datetime.now() - inicio
    logger.info(f"  ✓ Features creados: {n_total} ({n_lags} lags + {n_deltas} deltas)")
    logger.info(f"  ✓ Duración: {duracion}")
    logger.info(f"  ✓ Shape final: {df.shape}")
    
    return df


# ============================================================================
# UNDERSAMPLING
# ============================================================================

def aplicar_undersampling(df, ratio=None, semilla=None):
    """
    Aplica undersampling a la clase mayoritaria (CONTINUA)
    
    Args:
        df: DataFrame con columna 'clase_ternaria'
        ratio: Proporción de clase mayoritaria a mantener (ej: 0.1 = 10%)
        semilla: Semilla para reproducibilidad
    
    Returns:
        DataFrame con undersampling aplicado
    """
    if ratio is None:
        ratio = UNDERSAMPLING_RATIO
    if semilla is None:
        semilla = SEMILLA_PRIMIGENIA
    
    logger.info(f"Aplicando undersampling (ratio={ratio})...")
    
    # Separar clases
    df_continua = df[df['clase_ternaria'] == 'CONTINUA']
    df_baja1 = df[df['clase_ternaria'] == 'BAJA+1']
    df_baja2 = df[df['clase_ternaria'] == 'BAJA+2']
    
    logger.info(f"  Antes - CONTINUA: {len(df_continua):,}, BAJA+1: {len(df_baja1):,}, BAJA+2: {len(df_baja2):,}")
    
    # Submuestrear CONTINUA
    n_continua_mantener = int(len(df_continua) * ratio)
    df_continua_sampled = resample(
        df_continua,
        n_samples=n_continua_mantener,
        replace=False,
        random_state=semilla
    )
    
    # Combinar
    df_balanced = pd.concat([df_continua_sampled, df_baja1, df_baja2], ignore_index=True)
    df_balanced = df_balanced.sample(frac=1, random_state=semilla).reset_index(drop=True)
    
    logger.info(f"  Después - CONTINUA: {len(df_continua_sampled):,}, BAJA+1: {len(df_baja1):,}, BAJA+2: {len(df_baja2):,}")
    logger.info(f"  Total registros: {len(df):,} -> {len(df_balanced):,}")
    
    return df_balanced


# ============================================================================
# PREPARACIÓN DE DATOS (3 ETAPAS)
# ============================================================================

def preparar_datos_train_test_final(df):
    """
    Prepara los datos para el workflow de 3 etapas:
    1. TRAIN: 201901-202102 (todos los meses)
    2. TEST: 202104 y 202106 (validación)
    3. FINAL: 202108 (predicción final)
    
    Returns:
        Tupla de (df_train, df_test1, df_test2, df_final, feature_cols)
    """
    logger.info("Preparando datos para workflow de 3 etapas...")
    
    # Generar lista de meses de train
    meses_train = generar_rango_meses(FOTO_MES_TRAIN_INICIO, FOTO_MES_TRAIN_FIN)
    logger.info(f"Meses de entrenamiento: {meses_train[0]} a {meses_train[-1]} ({len(meses_train)} meses)")
    
    # Filtrar datos
    df_train = df[df['foto_mes'].isin(meses_train)].copy()
    df_test1 = df[df['foto_mes'] == FOTO_MES_TEST_1].copy()
    df_test2 = df[df['foto_mes'] == FOTO_MES_TEST_2].copy()
    df_final = df[df['foto_mes'] == FOTO_MES_FINAL].copy()
    
    logger.info(f"Train: {len(df_train):,} registros ({len(meses_train)} meses)")
    logger.info(f"Test 1 ({FOTO_MES_TEST_1}): {len(df_test1):,} registros")
    logger.info(f"Test 2 ({FOTO_MES_TEST_2}): {len(df_test2):,} registros")
    logger.info(f"Final ({FOTO_MES_FINAL}): {len(df_final):,} registros")
    
    # Aplicar undersampling solo a train
    if UNDERSAMPLING:
        df_train = aplicar_undersampling(df_train)
    
    # Definir columnas de features
    cols_excluir = ['numero_de_cliente', 'foto_mes', 'clase_ternaria']
    feature_cols = [col for col in df_train.columns if col not in cols_excluir]
    logger.info(f"Features: {len(feature_cols)}")
    
    return df_train, df_test1, df_test2, df_final, feature_cols


# ============================================================================
# ENTRENAMIENTO CON LIGHTGBM (estilo zlightgbm)
# ============================================================================

def entrenar_lgbm(X_train, y_train, X_val, y_val, semilla, usar_ganancia=False):
    """
    Entrena un modelo LightGBM con parámetros estilo zlightgbm
    """
    # Parámetros base
    lgbm_params = {
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'learning_rate': LEARNING_RATE,
        'num_leaves': NUM_LEAVES,
        'feature_fraction': FEATURE_FRACTION,
        'bagging_fraction': BAGGING_FRACTION,
        'bagging_freq': BAGGING_FREQ,
        'min_data_in_leaf': MIN_DATA_IN_LEAF,
        'max_bin': MAX_BIN,
        'verbose': -1,
        'seed': semilla,
        'force_row_wise': True,
    }
    
    if GRADIENT_BOUND is not None:
        lgbm_params['gradient_bound'] = GRADIENT_BOUND
    
    # Crear datasets
    train_data = lgb.Dataset(X_train, label=y_train, free_raw_data=True)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, free_raw_data=True)
    
    # Métrica personalizada
    feval = ganancia_lgb_binary if usar_ganancia else None
    
    # Entrenar
    modelo = lgb.train(
        lgbm_params,
        train_data,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[val_data],
        valid_names=['valid'],
        feval=feval,
        callbacks=[lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False)]
    )
    
    return modelo


# ============================================================================
# GUARDADO MEJORADO DE MODELOS (NUEVO)
# ============================================================================

def guardar_modelo_individual(modelo, semilla, exp_path):
    """
    Guarda un modelo individual del ensemble
    
    Args:
        modelo: Modelo LightGBM entrenado
        semilla: Semilla usada para entrenar
        exp_path: Ruta del experimento
    """
    modelos_dir = os.path.join(exp_path, "modelos")
    crear_directorio(modelos_dir)
    
    # Guardar como texto (formato LightGBM estándar)
    model_txt_path = os.path.join(modelos_dir, f"modelo_seed{semilla}.txt")
    modelo.save_model(model_txt_path)
    
    logger.debug(f"  Modelo seed={semilla} guardado en: {model_txt_path}")


# ============================================================================
# ETAPA 2: TESTING (202104 y 202106)
# ============================================================================

def etapa_testing(df_train, df_test1, df_test2, feature_cols, exp_path):
    """
    Etapa 2: Testing con datos de 202104 y 202106
    Entrena con train, predice en test1 y test2, calcula ganancias por corte
    
    Args:
        df_train: DataFrame de entrenamiento
        df_test1: DataFrame de test 1
        df_test2: DataFrame de test 2
        feature_cols: Lista de columnas de features
        exp_path: Ruta del directorio del experimento
    
    Returns:
        Tupla de (df_testing, mejor_corte, predicciones_test1, predicciones_test2)
    """
    logger.info("="*80)
    logger.info("ETAPA 2: TESTING (202104 y 202106)")
    logger.info("="*80)
    
    # Preparar datos
    X_train = df_train[feature_cols]
    y_train = (df_train['clase_ternaria'] == 'BAJA+2').astype(int)
    
    X_test1 = df_test1[feature_cols]
    y_test1 = (df_test1['clase_ternaria'] == 'BAJA+2').astype(int)
    
    X_test2 = df_test2[feature_cols]
    y_test2 = (df_test2['clase_ternaria'] == 'BAJA+2').astype(int)
    
    logger.info(f"Train - Registros: {len(X_train):,}, BAJA+2: {y_train.sum():,} ({y_train.mean()*100:.2f}%)")
    logger.info(f"Test 1 - Registros: {len(X_test1):,}, BAJA+2: {y_test1.sum():,} ({y_test1.mean()*100:.2f}%)")
    logger.info(f"Test 2 - Registros: {len(X_test2):,}, BAJA+2: {y_test2.sum():,} ({y_test2.mean()*100:.2f}%)")
    
    # Generar semillas
    semillas = generar_semillas(SEMILLA_PRIMIGENIA, SEMILLAS_EXPERIMENTO)
    
    # Matrices para acumular ganancias
    matriz_gan_test1 = np.zeros((SEMILLAS_EXPERIMENTO, len(CORTES)))
    matriz_gan_test2 = np.zeros((SEMILLAS_EXPERIMENTO, len(CORTES)))
    
    # Acumuladores para predicciones promedio
    predicciones_acum_test1 = np.zeros(len(X_test1))
    predicciones_acum_test2 = np.zeros(len(X_test2))
    
    # Lista para guardar todos los modelos
    modelos_entrenados = []
    
    # Entrenar con múltiples semillas
    for idx_sem, semilla in enumerate(semillas):
        logger.info(f"  Entrenando con semilla {idx_sem + 1}/{SEMILLAS_EXPERIMENTO} (seed={semilla})...")
        
        # Crear pequeño conjunto de validación para early stopping
        n_val = min(5000, len(X_train) // 10)
        indices = np.arange(len(X_train))
        np.random.seed(semilla)
        np.random.shuffle(indices)
        idx_val = indices[:n_val]
        
        X_val_mini = X_train.iloc[idx_val]
        y_val_mini = y_train.iloc[idx_val]
        
        # Entrenar
        modelo = entrenar_lgbm(
            X_train, y_train,
            X_val_mini, y_val_mini,
            semilla,
            usar_ganancia=False
        )
        
        # NUEVO: Guardar modelo individual
        guardar_modelo_individual(modelo, semilla, exp_path)
        modelos_entrenados.append(modelo)
        
        # Predecir en test1
        y_pred_test1 = modelo.predict(X_test1)
        predicciones_acum_test1 += y_pred_test1
        
        df_pred_test1 = pd.DataFrame({
            'y_true': y_test1.values,
            'y_pred': y_pred_test1
        }).sort_values('y_pred', ascending=False).reset_index(drop=True)
        
        # Predecir en test2
        y_pred_test2 = modelo.predict(X_test2)
        predicciones_acum_test2 += y_pred_test2
        
        df_pred_test2 = pd.DataFrame({
            'y_true': y_test2.values,
            'y_pred': y_pred_test2
        }).sort_values('y_pred', ascending=False).reset_index(drop=True)
        
        # Calcular ganancia para cada corte
        for idx_corte, corte in enumerate(CORTES):
            # Test 1
            n_envios = min(corte, len(df_pred_test1))
            df_pred_test1['pred_binary'] = 0
            df_pred_test1.loc[:n_envios-1, 'pred_binary'] = 1
            ganancia1, _ = calcular_ganancia(
                y_pred=df_pred_test1['pred_binary'].values,
                y_true=df_pred_test1['y_true'].values
            )
            matriz_gan_test1[idx_sem, idx_corte] = ganancia1
            
            # Test 2
            n_envios = min(corte, len(df_pred_test2))
            df_pred_test2['pred_binary'] = 0
            df_pred_test2.loc[:n_envios-1, 'pred_binary'] = 1
            ganancia2, _ = calcular_ganancia(
                y_pred=df_pred_test2['pred_binary'].values,
                y_true=df_pred_test2['y_true'].values
            )
            matriz_gan_test2[idx_sem, idx_corte] = ganancia2
        
        del modelo
        limpiar_memoria()
    
    # Promediar predicciones (ENSAMBLE)
    predicciones_promedio_test1 = predicciones_acum_test1 / SEMILLAS_EXPERIMENTO
    predicciones_promedio_test2 = predicciones_acum_test2 / SEMILLAS_EXPERIMENTO
    
    # NUEVO: Crear predicciones DETALLADAS con metadata
    pred_test1_detallado = pd.DataFrame({
        'numero_de_cliente': df_test1['numero_de_cliente'].values,
        'foto_mes': df_test1['foto_mes'].values,
        'prob': predicciones_promedio_test1,
        'clase_real': df_test1['clase_ternaria'].values
    }).sort_values('prob', ascending=False).reset_index(drop=True)
    
    pred_test2_detallado = pd.DataFrame({
        'numero_de_cliente': df_test2['numero_de_cliente'].values,
        'foto_mes': df_test2['foto_mes'].values,
        'prob': predicciones_promedio_test2,
        'clase_real': df_test2['clase_ternaria'].values
    }).sort_values('prob', ascending=False).reset_index(drop=True)
    
    # ========================================================================
    # GENERAR SUBMISSIONS PARA CADA MES DE TESTING
    # ========================================================================
    logger.info("\nGenerando submissions para meses de testing...")
    
    # Submissions para TEST 1 (202104)
    logger.info(f"\nGenerando submissions para {FOTO_MES_TEST_1}...")
    generar_submissions(pred_test1_detallado, exp_path, cortes=CORTES, sufijo=f"_test_{FOTO_MES_TEST_1}")
    
    # Submissions para TEST 2 (202106)
    logger.info(f"\nGenerando submissions para {FOTO_MES_TEST_2}...")
    generar_submissions(pred_test2_detallado, exp_path, cortes=CORTES, sufijo=f"_test_{FOTO_MES_TEST_2}")
    
    # Calcular ganancias promedio por corte
    ganancias_test1 = matriz_gan_test1.mean(axis=0)
    ganancias_test2 = matriz_gan_test2.mean(axis=0)
    ganancia_promedio = (ganancias_test1 + ganancias_test2) / 2
    
    # Encontrar mejor corte
    idx_mejor = np.argmax(ganancia_promedio)
    mejor_corte = CORTES[idx_mejor]
    
    logger.info(f"\n🎯 Mejor corte: {mejor_corte}")
    logger.info(f"   Ganancia Test 1: ${ganancias_test1[idx_mejor]:,.0f}")
    logger.info(f"   Ganancia Test 2: ${ganancias_test2[idx_mejor]:,.0f}")
    logger.info(f"   Ganancia Promedio: ${ganancia_promedio[idx_mejor]:,.0f}")
    
    # Crear DataFrame con resultados de testing
    df_testing = pd.DataFrame({
        'corte': CORTES,
        'ganancia_test1': ganancias_test1,
        'ganancia_test2': ganancias_test2,
        'ganancia_promedio': ganancia_promedio
    })
    
    # Guardar resultados
    logger.info("\nGuardando resultados de testing...")
    
    # 1. Evaluación de cortes
    eval_path = os.path.join(exp_path, "evaluacion_testing.csv")
    df_testing.to_csv(eval_path, index=False)
    logger.info(f"  ✓ evaluacion_testing.csv")
    
    # 2. Matrices de ganancia
    matriz_path1 = os.path.join(exp_path, "matriz_test1.csv")
    pd.DataFrame(matriz_gan_test1, columns=[f'corte_{c}' for c in CORTES]).to_csv(matriz_path1, index=False)
    logger.info(f"  ✓ matriz_test1.csv")
    
    matriz_path2 = os.path.join(exp_path, "matriz_test2.csv")
    pd.DataFrame(matriz_gan_test2, columns=[f'corte_{c}' for c in CORTES]).to_csv(matriz_path2, index=False)
    logger.info(f"  ✓ matriz_test2.csv")
    
    # 3. NUEVO: Predicciones DETALLADAS (con prob + clase_real)
    pred_test1_path = os.path.join(exp_path, "predicciones_test1_detallado.csv")
    pred_test1_detallado.to_csv(pred_test1_path, index=False)
    logger.info(f"  ✓ predicciones_test1_detallado.csv ({len(pred_test1_detallado):,} registros)")
    
    pred_test2_path = os.path.join(exp_path, "predicciones_test2_detallado.csv")
    pred_test2_detallado.to_csv(pred_test2_path, index=False)
    logger.info(f"  ✓ predicciones_test2_detallado.csv ({len(pred_test2_detallado):,} registros)")
    
    return df_testing, mejor_corte, pred_test1_detallado, pred_test2_detallado


# ============================================================================
# ETAPA 3: PREDICCIÓN FINAL (202108)
# ============================================================================

def etapa_final(df_train, df_final, feature_cols, exp_path):
    """
    Etapa 3: Predicción final en 202108
    Entrena con train, predice en final, genera submissions
    
    Args:
        df_train: DataFrame de entrenamiento
        df_final: DataFrame final (202108)
        feature_cols: Lista de columnas de features
        exp_path: Ruta del directorio del experimento
    
    Returns:
        DataFrame con predicciones ordenadas y Predicted para cada corte
    """
    logger.info("="*80)
    logger.info("ETAPA 3: PREDICCIÓN FINAL (202108)")
    logger.info("="*80)
    
    # Preparar datos
    X_train = df_train[feature_cols]
    y_train = (df_train['clase_ternaria'] == 'BAJA+2').astype(int)
    
    X_final = df_final[feature_cols]
    
    logger.info(f"Train: {len(X_train):,} registros")
    logger.info(f"Final ({FOTO_MES_FINAL}): {len(X_final):,} registros")
    logger.info(f"Entrenando con {SEMILLAS_FINAL} semillas (ENSAMBLE)...")
    
    # Generar semillas
    semillas = generar_semillas(SEMILLA_PRIMIGENIA, SEMILLAS_FINAL)
    
    # Acumulador de predicciones
    predicciones_acum = np.zeros(len(X_final))
    
    # Entrenar múltiples modelos
    for idx, semilla in enumerate(semillas, 1):
        logger.info(f"  Entrenando modelo {idx}/{SEMILLAS_FINAL} (seed={semilla})...")
        
        # Crear conjunto de validación mínimo
        n_val = min(5000, len(X_train) // 10)
        indices = np.arange(len(X_train))
        np.random.seed(semilla)
        np.random.shuffle(indices)
        idx_val = indices[:n_val]
        
        X_val_mini = X_train.iloc[idx_val]
        y_val_mini = y_train.iloc[idx_val]
        
        # Entrenar
        modelo = entrenar_lgbm(
            X_train, y_train,
            X_val_mini, y_val_mini,
            semilla,
            usar_ganancia=False
        )
        
        # NUEVO: Guardar modelo individual
        guardar_modelo_individual(modelo, semilla, exp_path)
        
        # Predecir
        predicciones = modelo.predict(X_final)
        predicciones_acum += predicciones
        
        del modelo
        limpiar_memoria()
    
    # Promediar predicciones (ENSAMBLE)
    predicciones_promedio = predicciones_acum / SEMILLAS_FINAL
    
    # NUEVO: Crear DataFrame DETALLADO con metadata
    resultado = pd.DataFrame({
        'numero_de_cliente': df_final['numero_de_cliente'].values,
        'foto_mes': df_final['foto_mes'].values,
        'prob': predicciones_promedio
    })
    
    resultado = resultado.sort_values('prob', ascending=False).reset_index(drop=True)
    
    logger.info(f"\nPredicciones generadas: {len(resultado):,}")
    logger.info(f"Top 10 probabilidades: {resultado['prob'].head(10).values}")
    
    # ========================================================================
    # GUARDAR PREDICCIONES COMPLETAS DETALLADAS
    # ========================================================================
    logger.info("\nGuardando predicciones finales...")
    
    # NUEVO: Predicciones DETALLADAS con prob + metadata
    pred_detallado_path = os.path.join(exp_path, f"predicciones_final_detallado_{FOTO_MES_FINAL}.csv")
    resultado.to_csv(pred_detallado_path, index=False)
    logger.info(f"  ✓ predicciones_final_detallado_{FOTO_MES_FINAL}.csv ({len(resultado):,} registros)")
    
    return resultado


# ============================================================================
# GENERACIÓN DE SUBMISSIONS (MEJORADO)
# ============================================================================

def generar_submissions(predicciones, exp_path, cortes=None, sufijo=""):
    """
    Genera archivos de submission para cada corte
    MEJORA: Incluye archivo Kaggle simple Y predicciones detalladas
    
    Args:
        predicciones: DataFrame con predicciones ordenadas
        exp_path: Ruta del directorio del experimento
        cortes: Lista de cortes a evaluar
        sufijo: Sufijo para diferenciar archivos (ej: "_test_202104", "_final_202108")
    
    Returns:
        DataFrame con resumen de cortes
    """
    if cortes is None:
        cortes = CORTES
    
    # Determinar el mes del dataset
    if 'foto_mes' in predicciones.columns:
        mes = predicciones['foto_mes'].iloc[0]
        logger.info(f"Generando {len(cortes)} submissions para mes {mes}...")
    else:
        logger.info(f"Generando {len(cortes)} submissions{sufijo}...")
    
    kaggle_dir = os.path.join(exp_path, "kaggle")
    crear_directorio(kaggle_dir)
    
    resultados = []
    
    for corte in cortes:
        pred_temp = predicciones.copy()
        pred_temp['Predicted'] = (pred_temp.index < corte).astype(int)
        
        # MEJORA 1: Archivo Kaggle SIMPLE (solo numero_de_cliente, Predicted)
        filename_simple = f"KA{EXPERIMENTO}_{corte}{sufijo}.csv"
        filepath_simple = os.path.join(kaggle_dir, filename_simple)
        
        pred_temp[['numero_de_cliente', 'Predicted']].to_csv(
            filepath_simple, index=False
        )
        
        # MEJORA 2: Archivo DETALLADO (prob + Predicted + foto_mes)
        filename_detallado = f"KA{EXPERIMENTO}_{corte}{sufijo}_detallado.csv"
        filepath_detallado = os.path.join(kaggle_dir, filename_detallado)
        
        pred_temp.to_csv(filepath_detallado, index=False)
        
        envios = pred_temp['Predicted'].sum()
        resultados.append({
            'corte': corte,
            'envios': envios,
            'archivo_kaggle': filename_simple,
            'archivo_detallado': filename_detallado,
            'sufijo': sufijo
        })
        
        if corte % 2500 == 0 or corte == cortes[0]:
            logger.info(f"  Corte {corte}: {envios} envíos")
            logger.info(f"    → {filename_simple} (Kaggle)")
            logger.info(f"    → {filename_detallado} (Detallado)")
    
    # Crear y guardar DataFrame de resultados
    df_resultados = pd.DataFrame(resultados)
    resultados_path = os.path.join(exp_path, f"resultados_cortes{sufijo}.csv")
    df_resultados.to_csv(resultados_path, index=False)
    
    logger.info(f"\nArchivos generados:")
    logger.info(f"  ✓ {len(cortes)} submissions Kaggle (simples) en: {kaggle_dir}")
    logger.info(f"  ✓ {len(cortes)} archivos detallados en: {kaggle_dir}")
    logger.info(f"  ✓ resultados_cortes{sufijo}.csv")
    
    return df_resultados


# ============================================================================
# WORKFLOW PRINCIPAL
# ============================================================================

def main():
    """Función principal del workflow de 3 etapas MEJORADO"""
    print("="*80)
    print("R_to_py MEJORADO: Workflow Completo de 3 Etapas")
    print("="*80)
    print("\nMEJORAS:")
    print("  ✓ Canaritos con Polars (eficiente, al principio)")
    print("  ✓ Cada modelo del ensemble guardado individualmente")
    print("  ✓ Predicciones detalladas (prob + metadata)")
    print("  ✓ Archivos Kaggle simples + detallados")
    print("="*80)
    
    inicio_ejecucion = datetime.now()
    
    # Información inicial (antes de configurar archivo de log)
    print(f"\nInicio: {inicio_ejecucion.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Experimento: {EXPERIMENTO}")
    
    # ========================================================================
    # CREAR DIRECTORIO Y CONFIGURAR LOGGING CON ARCHIVO
    # ========================================================================
    exp_path = crear_directorio_experimento()
    log_file = configurar_logging_con_archivo(exp_path)
    
    # A partir de aquí, todo se guarda en archivo Y se muestra en consola
    logger.info("="*80)
    logger.info("R_to_py MEJORADO: Workflow Completo de 3 Etapas")
    logger.info("="*80)
    logger.info(f"Inicio: {inicio_ejecucion.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Experimento: {EXPERIMENTO}")
    logger.info(f"Ganancia por acierto: ${GANANCIA_ACIERTO:,}")
    logger.info(f"Costo por estímulo: ${COSTO_ESTIMULO:,}")
    logger.info(f"Canaritos: {QCANARITOS} (con Polars, al principio)")
    logger.info(f"Lags y Deltas: {FEATURE_ENGINEERING_LAGS} (órdenes: {LAGS_ORDEN if FEATURE_ENGINEERING_LAGS else 'N/A'})")
    logger.info(f"Undersampling: {UNDERSAMPLING} (ratio={UNDERSAMPLING_RATIO})")
    logger.info("")
    
    # Guardar configuración en JSON
    config_path = guardar_configuracion(exp_path)
    logger.info(f"Configuración guardada en: {config_path}")
    logger.info("")
    
    # ========================================================================
    # PASO 1: CARGA Y PREPROCESAMIENTO
    # ========================================================================
    logger.info("="*80)
    logger.info("PASO 1: Carga y preprocesamiento")
    logger.info("="*80)
    
    logger.info(f"Cargando dataset desde {DATASET_PATH}...")
    df = pd.read_csv(DATASET_PATH, compression='gzip')
    logger.info(f"Dataset cargado: {df.shape}")
    
    # Calcular clase_ternaria
    df = calcular_clase_ternaria(df)
    
    # MEJORADO: Agregar canaritos con Polars
    df = agregar_canaritos_polars(df, QCANARITOS)
    
    # Agregar lags y deltas
    if FEATURE_ENGINEERING_LAGS:
        df = agregar_lags_y_deltas(df, LAGS_ORDEN)
    
    limpiar_memoria()
    print()
    
    # ========================================================================
    # PASO 2: PREPARACIÓN DE DATOS (3 ETAPAS)
    # ========================================================================
    logger.info("="*80)
    logger.info("PASO 2: Preparación de datos (3 etapas)")
    logger.info("="*80)
    logger.info(f"Train: {FOTO_MES_TRAIN_INICIO} a {FOTO_MES_TRAIN_FIN}")
    logger.info(f"Test: {FOTO_MES_TEST_1} y {FOTO_MES_TEST_2}")
    logger.info(f"Final: {FOTO_MES_FINAL}")
    print()
    
    df_train, df_test1, df_test2, df_final, feature_cols = preparar_datos_train_test_final(df)
    print()
    
    # ========================================================================
    # PASO 3: ETAPA TESTING (202104 y 202106)
    # ========================================================================
    df_testing, mejor_corte, pred_test1, pred_test2 = etapa_testing(
        df_train, df_test1, df_test2, feature_cols, exp_path
    )
    print()
    
    # ========================================================================
    # PASO 4: ETAPA FINAL (202108)
    # ========================================================================
    predicciones = etapa_final(df_train, df_final, feature_cols, exp_path)
    print()
    
    # ========================================================================
    # PASO 5: GENERACIÓN DE SUBMISSIONS
    # ========================================================================
    logger.info("="*80)
    logger.info("PASO 5: Generación de submissions para mes FINAL")
    logger.info("="*80)
    
    df_resultados = generar_submissions(predicciones, exp_path, sufijo=f"_final_{FOTO_MES_FINAL}")
    print()
    
    # ========================================================================
    # RESUMEN FINAL
    # ========================================================================
    fin_ejecucion = datetime.now()
    duracion = fin_ejecucion - inicio_ejecucion
    
    logger.info("="*80)
    logger.info("RESUMEN DE ARCHIVOS GENERADOS")
    logger.info("="*80)
    logger.info(f"\n📁 Directorio: {exp_path}\n")
    logger.info("📄 Configuración y Logs:")
    logger.info(f"  ✓ configuracion.json (parámetros del experimento)")
    logger.info(f"  ✓ experimento.log (log completo de ejecución)")
    logger.info("\n📊 Testing:")
    logger.info(f"  ✓ evaluacion_testing.csv (mejor corte: {mejor_corte})")
    logger.info(f"  ✓ matriz_test1.csv")
    logger.info(f"  ✓ matriz_test2.csv")
    logger.info(f"  ✓ predicciones_test1_detallado.csv ({len(pred_test1):,} registros)")
    logger.info(f"  ✓ predicciones_test2_detallado.csv ({len(pred_test2):,} registros)")
    logger.info(f"  ✓ resultados_cortes_test_{FOTO_MES_TEST_1}.csv")
    logger.info(f"  ✓ resultados_cortes_test_{FOTO_MES_TEST_2}.csv")
    logger.info("\n🎯 Final:")
    logger.info(f"  ✓ predicciones_final_detallado_{FOTO_MES_FINAL}.csv ({len(predicciones):,} registros)")
    logger.info(f"  ✓ resultados_cortes_final_{FOTO_MES_FINAL}.csv")
    logger.info(f"\n📤 Submissions (por mes):")
    logger.info(f"  ✓ kaggle/ - Mes {FOTO_MES_TEST_1}: {len(CORTES)} archivos Kaggle + {len(CORTES)} detallados")
    logger.info(f"  ✓ kaggle/ - Mes {FOTO_MES_TEST_2}: {len(CORTES)} archivos Kaggle + {len(CORTES)} detallados")
    logger.info(f"  ✓ kaggle/ - Mes {FOTO_MES_FINAL}: {len(CORTES)} archivos Kaggle + {len(CORTES)} detallados")
    logger.info(f"  ✓ Total: {len(CORTES) * 3 * 2} archivos ({len(CORTES) * 3} Kaggle + {len(CORTES) * 3} detallados)")
    logger.info(f"\n🔧 Modelos:")
    n_modelos_total = SEMILLAS_EXPERIMENTO + SEMILLAS_FINAL
    logger.info(f"  ✓ modelos/ ({n_modelos_total} modelos individuales guardados)")
    
    logger.info("\n" + "="*80)
    logger.info("WORKFLOW COMPLETADO EXITOSAMENTE!")
    logger.info("="*80)
    logger.info(f"Mejor corte sugerido (testing): {mejor_corte}")
    logger.info(f"Inicio: {inicio_ejecucion.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Fin: {fin_ejecucion.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Duración: {duracion}")
    logger.info(f"Log guardado en: {log_file}")
    logger.info("="*80)
    
    return predicciones, df_testing, df_resultados, pred_test1, pred_test2, exp_path


if __name__ == "__main__":
    main()
