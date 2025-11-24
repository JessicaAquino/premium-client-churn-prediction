import polars as pl
import infra.logger_wrapper as log

@log.process_log
def col_selection(df: pl.DataFrame) -> tuple[list[str], list[list[str]]]:
    # Columns to drop
    col_drops = {
        "numero_de_cliente", "foto_mes", "active_quarter", "clase_ternaria",
        "cliente_edad", "cliente_antiguedad",
        "Visa_fultimo_cierre", "Master_fultimo_cierre",
        "Visa_Fvencimiento", "Master_Fvencimiento"
    }

    # --- Categorical vs Numerical ---
    cat_cols = []
    num_cols = []
    for c in df.columns:
        if c in col_drops:
            continue
        nunique = df.select(pl.col(c).n_unique()).item()
        if nunique <= 5:
            cat_cols.append(c)
        else:
            num_cols.append(c)

    # --- Prefix-based splits ---
    lista_t = [c for c in df.columns if c.startswith("t") and c not in col_drops]
    lista_c = [c for c in df.columns if c.startswith("c") and c not in col_drops]
    lista_m = [c for c in df.columns if c.startswith("m") and c not in col_drops]
    lista_r = [c for c in df.columns if c not in (lista_t + lista_c + lista_m + list(col_drops))]

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
