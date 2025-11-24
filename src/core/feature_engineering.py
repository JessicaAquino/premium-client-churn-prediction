import polars as pl
import infra.logger_wrapper as log

from infra.duckdb_runner import run_duckdb_query

@log.process_log
def feature_engineering_pipeline(df: pl.DataFrame, config: dict) -> pl.DataFrame:
    """
    Ejecuta el pipeline de feature engineering completo

    Parameters:
    -----------
    data_path : str
        Ruta al archivo de datos
    config : dict
        Configuración del pipeline. Ejemplo:

        "lag": {
            "columns": ["col1", "col2"],
            "n": 2   # number of lags
        },
        "delta": {
            "columns": ["col1", "col2"],
            "n": 2   # number of deltas
        },
        "minmax": {
            "columns": ["col1", "col2"]
        },
        "ratio": {
            "pairs": [["monto", "cantidad"], ["ingresos", "clientes"]]
        },
        "linreg": {
            "columns": ["col1"],
            "window": 3  # optional, for flexibility
        }

    Returns:
    --------
    pl.DataFrame
        DataFrame con las nuevas features agregadas
    """

    sql = "SELECT *"

    window_clause = ""

    if "lag" in config:
        sql += add_lag_sql(config["lag"])

    if "delta" in config:
        sql += add_delta_sql(config["delta"])

    if "minmax" in config:
        sql += add_minmax_sql(config["minmax"])
    
    if "ratio" in config:
        sql += add_ratio_sql(config["ratio"])

    if "linreg" in config:
        linreg_str, window_clause = add_linreg_sql(config["linreg"])
        sql += linreg_str

    sql += " FROM df"
    if window_clause != "":
        sql += window_clause

    df = run_duckdb_query(df, sql)

    return df

def add_lag_sql(config_lag: dict) -> str:    
    lag_str = ""
    for col in config_lag["columns"]:
        for i in range(1, config_lag["n"] + 1):
            lag_str += f", lag({col}, {i}) OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes) AS {col}_lag_{i}"

    return lag_str

def add_delta_sql(config_delta: dict) -> str:
    delta_str = ""
    for col in config_delta["columns"]:
        for i in range(1, config_delta["n"] + 1):
            delta_str += f", {col} - {col}_lag_{i} AS {col}_delta_{i}"

    return delta_str

def add_minmax_sql(config_minmax: dict) -> str:
    min_max_sql = ""
    for col in config_minmax["columns"]:
        min_max_sql += f", MAX({col}) OVER (PARTITION BY numero_de_cliente) AS {col}_MAX, MIN({col}) OVER (PARTITION BY numero_de_cliente) AS {col}_MIN"

    return min_max_sql

def add_ratio_sql(config_ratio: dict) -> str:
    ratio_sql = ""
    for pair in config_ratio["pairs"]:
        ratio_sql += f", IF({pair[1]} = 0, 0, {pair[0]} / {pair[1]}) AS ratio_{pair[0]}_{pair[1]}"

    return ratio_sql

def add_linreg_sql(config_linreg: dict) -> tuple:
    linreg_sql = ""
    window_size = config_linreg.get("window", 3)
    for col in config_linreg["columns"]:
        linreg_sql += f", REGR_SLOPE({col}, cliente_antiguedad) OVER ventana_{window_size} AS slope_{col}"
    
    window_clause = f" WINDOW ventana_{window_size} AS (PARTITION BY numero_de_cliente ORDER BY foto_mes ROWS BETWEEN {window_size} PRECEDING AND CURRENT ROW)"

    return linreg_sql, window_clause