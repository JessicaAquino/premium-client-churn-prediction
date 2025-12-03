import polars as pl
import infra.logger_wrapper as log
import duckdb
from config.context import Context

@log.process_log
def create_df_chiquito(ctx: Context) -> list[tuple[str, str]]:
    """
    Returns a list of (column_name, column_type) for df_init using Polars instead of pandas.
    """

    conn = duckdb.connect(ctx.database)

    # Get schema as a Polars DataFrame
    schema_pl = conn.execute("DESCRIBE df_init").pl()

    conn.close()

    # Extract columns safely
    col_names = schema_pl.get_column("column_name").to_list()
    col_types = schema_pl.get_column("column_type").to_list()

    return list(zip(col_names, col_types))


@log.process_log
def col_selection(cols_with_types: list[tuple[str, str]]):
    """
    - Filtra columnas que NO sean features generadas (_lag, _delta, etc)
    - Descarta columnas prohibidas (cliente_edad, numero_de_cliente, etc)
    - Construye:
        * cols_percentil
        * cols_lag_delta_max_min_regl
        * cols_ratios
    """

    # Column names only
    all_cols = [c for c, _t in cols_with_types]

    # Columnas que NO deben ser usadas para FE histórica
    palabras_features_excluir = ["_lag", "_delta", "_slope", "_max", "_min", "_ratio", "_mean"]

    # Filtramos columnas *limpias*
    columnas_cleaned = [
        c for c in all_cols
        if not any(substr in c for substr in palabras_features_excluir)
    ]

    # Columnas que siempre deben ser descartadas
    col_drops = {
        "numero_de_cliente", "foto_mes", "mes", "active_quarter",
        "clase_ternaria", "clase_binaria", "clase_binaria_2", "clase_peso",
        "cliente_edad", "cliente_antiguedad",
        "Visa_fultimo_cierre", "Master_fultimo_cierre",
        "Visa_Fvencimiento", "Master_Fvencimiento"
    }

    # --- Lists by prefix ---
    lista_t = [c for c in columnas_cleaned if c.startswith("t") and c not in col_drops]
    lista_c = [c for c in columnas_cleaned if c.startswith("c") and c not in col_drops]
    lista_m = [c for c in columnas_cleaned if c.startswith("m") and c not in col_drops]

    lista_r = [
        c for c in columnas_cleaned
        if c not in (lista_t + lista_c + lista_m) and c not in col_drops
    ]

    # --- Columnas para lags, deltas, linreg, max/min ---
    cols_lag_delta_max_min_regl = lista_m + lista_c + lista_r + lista_t

    # --- Columnas para percentil ---
    cols_percentil = lista_m + [c for c in lista_r if "_m" in c]

    # --- Columnas para ratios (m vs c con mismo sufijo) ---
    cols_ratios = []
    for c in lista_c:
        suf = c[1:]
        match_m = next((m for m in lista_m if m[1:] == suf), None)
        if match_m:
            cols_ratios.append([match_m, c])

    return cols_percentil, cols_lag_delta_max_min_regl, cols_ratios



# @log.process_log
# def col_selection(cols_with_types: list[tuple[str, str]]) -> tuple[list[str], list[list[str]]]:
#     all_cols = [c for c, _t in cols_with_types]
    
#     # Columns to drop
#     col_drops = {
#         "numero_de_cliente", "foto_mes", "active_quarter",
#         "cliente_edad", "cliente_antiguedad",
#         "Visa_fultimo_cierre", "Master_fultimo_cierre",
#         "Visa_Fvencimiento", "Master_Fvencimiento",
#         "clase_ternaria","clase_binaria","clase_binaria_2","clase_peso"
#     }

#     usable_cols = [c for c in all_cols if c not in col_drops]

#     # --- Prefix-based splits ---
#     lista_t = [c for c in usable_cols if c.startswith("t")]
#     lista_c = [c for c in usable_cols if c.startswith("c")]
#     lista_m = [c for c in usable_cols if c.startswith("m")]
#     lista_r = [c for c in usable_cols if c not in (lista_t + lista_c + lista_m)]

#     # --- Features for lags, deltas, max/min, regression ---
#     cols_lag_delta_max_min_regl = lista_m + lista_c + lista_r

#     # --- Ratios: match c-columns with m-columns (same suffix) ---
#     cols_ratios = []
#     for c in lista_c:
#         suffix = c[1:]
#         match = next((m for m in lista_m if m[1:] == suffix), None)
#         if match:
#             cols_ratios.append([match, c])

#     return cols_lag_delta_max_min_regl, cols_ratios
