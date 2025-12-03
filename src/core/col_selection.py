import polars as pl
import src.infra.logger_wrapper as log
import duckdb
from config.context import Context

@log.process_log
def create_df_chiquito(ctx: Context) -> list[tuple[str, str]]:
    """
    Return a list of (column_name, column_type) for the table using Polars (no pandas).
    """

    conn = duckdb.connect(ctx.database)

    sql = "DESCRIBE df_init"

    schema_pl = conn.execute(sql).pl()

    conn.close()

    # Extract pairs (column_name, column_type)
    cols_with_types = list(
        zip(
            schema_pl[:, "column_name"].to_list(),
            schema_pl[:, "column_type"].to_list()
        )
    )

    return cols_with_types


@log.process_log
def col_selection(cols_with_types: list[tuple[str, str]]) -> tuple[list[str], list[list[str]]]:
    all_cols = [c for c, _t in cols_with_types]
    
    # Columns to drop
    col_drops = {
        "numero_de_cliente", "foto_mes", "active_quarter",
        "cliente_edad", "cliente_antiguedad",
        "Visa_fultimo_cierre", "Master_fultimo_cierre",
        "Visa_Fvencimiento", "Master_Fvencimiento",
        "clase_ternaria","clase_binaria","clase_binaria_2","clase_peso"
    }

    usable_cols = [c for c in all_cols if c not in col_drops]

    # --- Prefix-based splits ---
    lista_t = [c for c in usable_cols if c.startswith("t")]
    lista_c = [c for c in usable_cols if c.startswith("c")]
    lista_m = [c for c in usable_cols if c.startswith("m")]
    lista_r = [c for c in usable_cols if c not in (lista_t + lista_c + lista_m)]

    # --- Features for lags, deltas, max/min, regression ---
    cols_lag_delta_max_min_regl = lista_m + lista_c + lista_r

    # --- Ratios: match c-columns with m-columns (same suffix) ---
    cols_ratios = []
    for c in lista_c:
        suffix = c[1:]
        match = next((m for m in lista_m if m[1:] == suffix), None)
        if match:
            cols_ratios.append([match, c])

    return cols_lag_delta_max_min_regl, cols_ratios
