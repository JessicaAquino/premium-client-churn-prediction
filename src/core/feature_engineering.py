import logging
import duckdb
import infra.logger_wrapper as log
import core.col_selection as cs

from config.context import Context

logger = logging.getLogger(__name__)

@log.process_log
def feature_engineering_pipeline(ctx: Context):
    
    cols_with_types = cs.create_df_chiquito(ctx)
    all_cols = [c for c, _t in cols_with_types]

    logger.info(all_cols)

    new_columns = {
        "month": True,
        "ctrx_norm": True,
        "mpayroll_over_edad": True,
        "sum_prod_serv": True,
        "sum_ratio_ganancias_gastos": True}
    
    create_new_columns(ctx, all_cols, new_columns)

    # historic_fe = {
    #     "lag": 2,
    #     "delta": 2,
    #     "minmax": True,
    #     "ratio": True,
    #     "linreg": 3
    # }

    # create_lag_delta_linreg_minmax_ratio(ctx, historic_fe, cols_with_types)

    ORDEN_LAGS = 3
    VENTANA = 6

    # PERCENTIL
    df_init_chiquito = cs.create_df_chiquito(ctx)
    all_cols = [c for c, _t in df_init_chiquito]

    cols_percentil, _, _ = cs.col_selection(df_init_chiquito)
    feature_engineering_percentil(ctx, all_cols, cols_percentil, bins=20)

    # RATIOS
    df_init_chiquito = cs.create_df_chiquito(ctx)
    all_cols = [c for c, _t in df_init_chiquito]

    _, _, cols_ratios = cs.col_selection(df_init_chiquito)
    feature_engineering_ratio(ctx, all_cols, cols_ratios)
 

    df_init_chiquito = cs.create_df_chiquito(ctx)
    all_cols = [c for c, _t in df_init_chiquito]

    _, cols_lag_delta_max_min_regl, _ = cs.col_selection(df_init_chiquito)

    feature_engineering_lag(ctx, all_cols, cols_lag_delta_max_min_regl, ORDEN_LAGS)
    feature_engineering_delta(ctx, all_cols, cols_lag_delta_max_min_regl, ORDEN_LAGS)
    feature_engineering_linreg(ctx, all_cols, cols_lag_delta_max_min_regl, VENTANA)
    feature_engineering_max_min(ctx, all_cols, cols_lag_delta_max_min_regl, VENTANA)
    feature_engineering_lag_orden_fijo(ctx, all_cols, cols_lag_delta_max_min_regl, 6)
    feature_engineering_lag_orden_fijo(ctx, all_cols, cols_lag_delta_max_min_regl, 12)
    feature_engineering_delta_orden_fijo(ctx, all_cols, cols_lag_delta_max_min_regl, 6)
    feature_engineering_delta_orden_fijo(ctx, all_cols, cols_lag_delta_max_min_regl, 12)
    feature_engineering_mean(ctx, all_cols, cols_lag_delta_max_min_regl, VENTANA)

    return True

def create_new_columns(ctx: Context, all_cols: list[str], config: dict):
    
    sql = """
        CREATE OR REPLACE TABLE df_init AS 
        WITH base AS (
            SELECT * FROM df_init
        )
        SELECT *
    """
    
    if "month" in config:
        sql += add_month_num()

    if "ctrx_norm" in config:
        sql += add_ctrx_norm()
    
    if "mpayroll_over_edad" in config:
        sql += add_mpayroll_over_edad()

    if "sum_prod_serv" in config:
        sql += add_sum_prod_serv(all_cols)

    if "sum_ratio_ganancias_gastos" in config:
        sql += add_sum_ratio_ganancias_gastos(all_cols)

    sql += " FROM base"

    logger.info(f"Query new columns:\n{sql}")

    conn = duckdb.connect(ctx.database)
    conn.execute(sql)
    conn.close()

    return True

def add_month_num() -> str:
    month_str = ", foto_mes % 100 as mes"
    return month_str

def add_ctrx_norm() -> str:
    ctrx_norm = """
    , if(cliente_antiguedad= 1, ctrx_quarter * 5.0 , 
        if(cliente_antiguedad = 2 ,ctrx_quarter * 2.0, 
            if (cliente_antiguedad = 3 , ctrx_quarter * 1.2 , 
                ctrx_quarter))) 
            as ctrx_quarter_normalizado
    """
    return ctrx_norm

def add_mpayroll_over_edad() -> str:
    mpayroll_over_edad = """
    , mpayroll / cliente_edad as mpayroll_sobre_edad
    """
    return mpayroll_over_edad

def add_sum_prod_serv(all_cols: list[str]) -> str:
    """
    Generate SQL expressions that compute aggregated 'product/service' counts.
    """

    dict_prod_serv = {
        "master_visa_productos": [
            "Master_msaldototal", "Master_mconsumototal", "Master_mpagado", "Master_mlimitecompra",
            "Visa_msaldototal", "Visa_mconsumototal", "Visa_mpagado", "Visa_mlimitecompra"
        ],
        "cuentas_productos": [
            'mcuenta_corriente', 'mcaja_ahorro', 'mcaja_ahorro_dolares', 'mcuentas_saldo',
            'mcuenta_corriente_adicional', 'mcaja_ahorro_adicional'
        ],
        "tarjetas_productos": [c for c in all_cols if "tarjeta" in c and c[0] == "m"],
        "prestamos_productos": [c for c in all_cols if "prest" in c and c[0] == "m"],
        "inversiones_productos": [
            'mplazo_fijo_pesos', 'mplazo_fijo_dolares',
            'minversion1_pesos', 'minversion1_dolares', 'minversion2'
        ],
        "digitales_productos": [c for c in all_cols if c.startswith("t")],
        "servicios_productos": [
            'mpagodeservicios', 'mpagomiscuentas', 'mcuenta_debitos_automaticos',
            'mforex_buy', 'mforex_sell', 'mtransferencias_recibidas', 'mtransferencias_emitidas',
            'mextraccion_autoservicio', 'mcheques_depositados', 'mcheques_emitidos', 'mcajeros_propios_descuentos'
        ],
        "seguros_productos": [c for c in all_cols if 'segur' in c]
    }

    sql_parts = ""

    for group_name, cols in dict_prod_serv.items():

        if len(cols) == 0:
            # no columns detected for this group → skip safely
            continue

        feature_name = f"suma_de_{group_name}"

        # Build: IF(col1>0,1,0) + IF(col2>0,1,0) + ...
        sum_expr = " + ".join([
            f"IF(TRY_CAST({col} AS DOUBLE) > 0, 1, 0)"
            for col in cols
        ])

        sql_parts += f", ({sum_expr}) AS {feature_name}"

    return sql_parts

def add_sum_ratio_ganancias_gastos(all_cols: list[str]):
    ganancias_gastos = {
    "ganancias" : [
        # Ingresos por sueldo
        "mpayroll","mpayroll2",
        # Ahorro e inversiones (indican colchón financiero)
        "mplazo_fijo_pesos","mplazo_fijo_dolares","minversion1_pesos","minversion1_dolares","minversion2",
        # Plata que entra por transferencias
        "mtransferencias_recibidas",
        # Beneficios/descuentos (mejoran la situación neta)
        "mcajeros_propios_descuentos","mtarjeta_visa_descuentos","mtarjeta_master_descuentos"],
    
    "gastos": [# Comisiones y costos directos
        "mcomisiones","mcomisiones_mantenimiento","mcomisiones_otras",
        # Débitos y pagos de servicios (egresos automáticos)
        "mcuenta_debitos_automaticos","mpagodeservicios","mpagomiscuentas",
        # Deuda / préstamos (presión financiera)"mprestamos_personales",
        "mprestamos_prendarios","mprestamos_hipotecarios","mpasivos_margen",
        # Egresos por movimientos
        "mtransferencias_emitidas","mextraccion_autoservicio",
        "mcheques_emitidos","mcheques_depositados_rechazados","mcheques_emitidos_rechazados",
        # Uso de cajeros/ATM (generalmente salida de plata)
        "matm","matm_other"]
    }
    
    ganancias_clean = [c for c in ganancias_gastos["ganancias"] if c in all_cols]
    gastos_clean = [c for c in ganancias_gastos["gastos"] if c in all_cols]

    sql = ""

    # Build sum of gains
    if ganancias_clean:
        sum_gan = " + ".join([f"TRY_CAST({c} AS DOUBLE)" for c in ganancias_clean])
        sql += f", ({sum_gan}) AS monto_ganancias"
    else:
        sql += ", 0 AS monto_ganancias"

    # Build sum of expenses
    if gastos_clean:
        sum_gas = " + ".join([f"TRY_CAST({c} AS DOUBLE)" for c in gastos_clean])
        sql += f", ({sum_gas}) AS monto_gastos"
    else:
        sql += ", 0 AS monto_gastos"

    # Add ratio (safe divide)
    sql += """
    , CASE 
        WHEN monto_gastos IS NULL THEN NULL 
        ELSE monto_ganancias / (monto_gastos + 1) 
    END AS ganancia_gasto_dif
    """

    return sql

# def create_lag_delta_linreg_minmax_ratio(ctx: Context, feature_dict: dict, cols_with_types: list[tuple[str, str]]):

#     cols_lag_delta, cols_ratios = cs.col_selection(cols_with_types)
     
#     historic_fe = {
#         "lag": {"columns": cols_lag_delta, "n": feature_dict["lag"]},
#         "delta": {"columns": cols_lag_delta, "n": feature_dict["delta"]},
#         "minmax": {"columns": cols_lag_delta},
#         "ratio": {"pairs": cols_ratios},
#         "linreg": {"columns": cols_lag_delta[:10], "window": feature_dict["linreg"]}
#     }

#     sql = """
#         CREATE OR REPLACE TABLE df_init AS 
#         WITH base AS (
#             SELECT * FROM df_init
#         )
#         SELECT *
#     """

#     window_clause = ""

#     if "lag" in historic_fe:
#         sql += add_lag_sql(historic_fe["lag"])

#     if "delta" in historic_fe:
#         sql += add_delta_sql(historic_fe["delta"])

#     if "minmax" in historic_fe:
#         sql += add_minmax_sql(historic_fe["minmax"])
    
#     if "ratio" in historic_fe:
#         sql += add_ratio_sql(historic_fe["ratio"])

#     if "linreg" in historic_fe:
#         linreg_str, window_clause = add_linreg_sql(historic_fe["linreg"])
#         sql += linreg_str

#     sql += " FROM base"
#     if window_clause != "":
#         sql += " " + window_clause

#     logger.info(f"Query fe:\n{sql}")

#     conn = duckdb.connect(ctx.database)
#     conn.execute(sql)
#     conn.close()

# def add_lag_sql(config_lag: dict) -> str:    
#     lag_str = ""
#     for col in config_lag["columns"]:
#         for i in range(1, config_lag["n"] + 1):
#             lag_str += f", lag({col}, {i}) OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes) AS {col}_lag_{i}"

#     return lag_str

# def add_delta_sql(config_delta: dict) -> str:
#     delta_str = ""
#     for col in config_delta["columns"]:
#         for i in range(1, config_delta["n"] + 1):
#             delta_str += f", {col} - {col}_lag_{i} AS {col}_delta_{i}"

#     return delta_str

# def add_minmax_sql(config_minmax: dict) -> str:
#     min_max_sql = ""
#     for col in config_minmax["columns"]:
#         min_max_sql += f", MAX({col}) OVER (PARTITION BY numero_de_cliente) AS {col}_MAX, MIN({col}) OVER (PARTITION BY numero_de_cliente) AS {col}_MIN"

#     return min_max_sql

# def add_ratio_sql(config_ratio: dict) -> str:
#     ratio_sql = ""
#     for pair in config_ratio["pairs"]:
#         ratio_sql += f", IF({pair[1]} = 0, 0, {pair[0]} / {pair[1]}) AS ratio_{pair[0]}_{pair[1]}"

#     return ratio_sql

# def add_linreg_sql(config_linreg: dict) -> tuple:
#     linreg_sql = ""
#     window_size = config_linreg.get("window", 3)
#     for col in config_linreg["columns"]:
#         linreg_sql += f", REGR_SLOPE({col}, cliente_antiguedad) OVER ventana_{window_size} AS slope_{col}"
    
#     window_clause = f" WINDOW ventana_{window_size} AS (PARTITION BY numero_de_cliente ORDER BY foto_mes ROWS BETWEEN {window_size} PRECEDING AND CURRENT ROW)"

#     return linreg_sql, window_clause

@log.process_log
def feature_engineering_percentil(ctx: Context, all_cols: list[str] , columnas:list[str],bins:int=20):
    if any( c.endswith("_percentil") for c in all_cols):
        logger.info("Ya se realizo percentil")
        return
    logger.info("Todavia no se realizo percentil")
    sql ="""CREATE or REPLACE table df_init as """
    sql += f""" select *""" 
    for c in columnas:
        if c in all_cols:
            sql += f""", ntile({bins}) OVER (partition by foto_mes order by {c} ) as {c}_percentil"""
        else:
            logger.warning(f"No se encontro la columna {c} en el df")
    
    sql += " FROM df_init"

    logging.info(f"QUERY: \n{sql}")
    
    conn=duckdb.connect(ctx.database)
    conn.execute(sql)
    conn.close()

@log.process_log
def feature_engineering_ratio(ctx: Context, all_cols: list[str], columnas:list[list[str]] ):
    if any(c.endswith(f"_ratio") for c in all_cols):
        logger.info("Ya se hizo ratios")
        return
    logger.info("Todavia no se hizo ratios")
    sql="CREATE or REPLACE table df_init as "
    sql+="(SELECT *"
    for par in columnas:
        if par[0] in all_cols and par[1] in all_cols:
            sql+=f", if({par[1]}=0 ,0,{par[0]}/{par[1]}) as {par[0]}_{par[1]}_ratio"
        else:
            print(f"no se encontro el par de atributos {par}")

    sql+=" FROM df_init)"

    logging.info(f"QUERY: \n{sql}")

    conn = duckdb.connect(ctx.database)
    conn.execute(sql)
    conn.close()

@log.process_log
def feature_engineering_lag(ctx: Context, all_cols: list[str] ,columnas:list[str],orden_lag:int=1 ):
    
    orden_lag_ya_realizado=1
    marca_nueva_real=0
    while marca_nueva_real ==0 and orden_lag_ya_realizado <= orden_lag:
        if any(c.endswith(f"_lag_{orden_lag_ya_realizado}") for c in all_cols):
            logger.info(f"Ya se hizo lag_{orden_lag_ya_realizado}")
            orden_lag_ya_realizado+=1
        else:
            marca_nueva_real=1
    if orden_lag_ya_realizado > orden_lag:
        logger.info(f"Ya se hicieron todos los lags pedidos hasta orden {orden_lag_ya_realizado-1}")
        return
    
    logger.info(f"Ya se hicieron los lags hasta orden {orden_lag_ya_realizado-1}. Falta hasta orden {orden_lag}")
    sql = "CREATE or REPLACE table df_init as "
    sql +="(SELECT *"
    for attr in columnas:
        if attr in all_cols:
            for i in range(orden_lag_ya_realizado,orden_lag+1):
                sql+= f",lag(try_cast({attr} as double),{i}) OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes) AS {attr}_lag_{i}"
        else:
            logger.warning(f"No se encontro el atributo {attr} en df")
    sql+=" FROM df_init)"

    logging.info(f"QUERY: \n{sql}")

    # Ejecucion de la consulta SQL
    conn = duckdb.connect(ctx.database)
    conn.execute(sql)
    conn.close()

@log.process_log
def feature_engineering_delta(ctx: Context, all_cols: list[str] , columnas:list[str],orden_delta:int=1 ) :
    
    orden_delta_ya_realizado=1
    marca_nueva_real=0
    while marca_nueva_real ==0 and orden_delta_ya_realizado <= orden_delta:
        if any(c.endswith(f"_delta_{orden_delta_ya_realizado}")  for c in all_cols):
            logger.info(f"Ya se hizo delta_{orden_delta_ya_realizado}_")
            orden_delta_ya_realizado+=1
        else:
            marca_nueva_real=1
    if orden_delta_ya_realizado > orden_delta:
        logger.info(f"Ya se hicieron todos los deltas pedidos hasta orden {orden_delta_ya_realizado-1}")
        return
    logger.info(f"Ya se hicieron los deltas hasta orden {orden_delta_ya_realizado-1}. Falta hasta orden {orden_delta}")

    
    sql = "CREATE or REPLACE table df_init as "
    sql+="(SELECT *"
    for attr in columnas:
        if attr in all_cols:
            for i in range(orden_delta_ya_realizado,orden_delta+1):
                sql += (
                f", TRY_CAST({attr} AS DOUBLE) "
                f"- TRY_CAST({attr}_lag_{i} AS DOUBLE) AS {attr}_delta_{i}")
                # sql+= f", {attr}-{attr}_lag_{i} as delta_{i}_{attr}"
        else:
            logger.warning(f"No se encontro el atributo {attr} en df")
    sql+=" FROM df_init)"

    logging.info(f"QUERY: \n{sql}")

    conn = duckdb.connect(ctx.database)
    conn.execute(sql)
    conn.close()
 
@log.process_log
def feature_engineering_linreg(ctx: Context, all_cols: list[str], columnas:list[str],ventana:int=3) :
    if any(c.endswith("_slope") for c in all_cols):
        logger.info("Ya se hizo slope")
        return
    logger.info("Todavia no se hizo slope")
    sql = "Create or replace table df_init as "
    sql+="SELECT *"
    try:
        for attr in columnas:
            if attr in all_cols:
                sql+=f", regr_slope(try_cast({attr} as double) , try_cast(cliente_antiguedad as double) ) over ventana_{ventana} as {attr}_slope"
            else :
                print(f"no se encontro el atributo {attr}")
        sql+=f" FROM df_init window ventana_{ventana} as (partition by numero_de_cliente order by foto_mes rows between {ventana} preceding and current row)"
    except Exception as e:
        logger.error(f"Error en la regresion lineal : {e}")
        raise

    logging.info(f"QUERY: \n{sql}")

    conn = duckdb.connect(ctx.database)
    conn.execute(sql)
    conn.close()

@log.process_log
def feature_engineering_max_min(ctx: Context, all_cols: list[str] , columnas:list[str],ventana:int=3) :
    logger.info(f"df shape: {len(all_cols)}")
    palabras_max_min=["_max","_min"]
    if any(any(c.endswith(p) for p in palabras_max_min) for c in all_cols):
        logger.info("Ya se hizo max min")
        return
    
    sql="CREATE or REPLACE table df_init as "
    sql+="(SELECT *"
    try:

        for attr in columnas:
            if attr in all_cols:
                sql+=f", max(try_cast({attr} as double)  ) over ventana_{ventana} as {attr}_max ,min(try_cast({attr} as double)) over ventana_{ventana} as {attr}_min"
            else :
                print(f"no se encontro el atributo {attr}")
        sql+=f" FROM df_init window ventana_{ventana} as (partition by numero_de_cliente order by foto_mes rows between {ventana} preceding and current row))"
    except Exception as e:
        logger.error(f"Error en la max min : {e}")
        raise

    logging.info(f"QUERY: \n{sql}")

    conn=duckdb.connect(ctx.database)
    conn.execute(sql)
    conn.close()

@log.process_log
def feature_engineering_lag_orden_fijo(ctx: Context, all_cols: list[str] ,columnas:list[str],orden_lag:int=1 ):

    if any(c.endswith(f"_lag_{orden_lag}") for c in all_cols):
        logger.info(f"Ya se hizo lag de orden : {orden_lag}")
        return
    logger.info(f"Todavia no se hizo lag de orden : {orden_lag}")
    
    sql = "CREATE or REPLACE table df_init as "
    sql +="(SELECT *"
    for attr in columnas:
        if attr in all_cols:
            for i in range(orden_lag,orden_lag+1):
                sql+= f",lag(try_cast({attr} as double),{i}) OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes) AS {attr}_lag_{i}"
        else:
            logger.warning(f"No se encontro el atributo {attr} en df")
    sql+=" FROM df_init)"

    logging.info(f"QUERY: \n{sql}")

    # Ejecucion de la consulta SQL
    conn = duckdb.connect(ctx.database)
    conn.execute(sql)
    conn.close()

@log.process_log
def feature_engineering_delta_orden_fijo(ctx: Context, all_cols: list[str] , columnas:list[str],orden_delta:int=1 ) :
    
    if any(c.endswith(f"_delta_{orden_delta}") for c in all_cols):
        logger.info(f"Ya se hizo lag de orden : {orden_delta}")
        return
    logger.info(f"Todavia no se hizo lag de orden : {orden_delta}")
    
    sql = "CREATE or REPLACE table df_init as "
    sql+="(SELECT *"
    for attr in columnas:
        if attr in all_cols:
            for i in range(orden_delta,orden_delta+1):
                sql += (
                f", TRY_CAST({attr} AS DOUBLE) "
                f"- TRY_CAST({attr}_lag_{i} AS DOUBLE) AS {attr}_delta_{i}")
                # sql+= f", {attr}-{attr}_lag_{i} as delta_{i}_{attr}"
        else:
            logger.warning(f"No se encontro el atributo {attr} en df")
    sql+=" FROM df_init)"

    logging.info(f"QUERY: \n{sql}")

    conn = duckdb.connect(ctx.database)
    conn.execute(sql)
    conn.close()
    

@log.process_log
def feature_engineering_mean(ctx: Context, all_cols: list[str], columnas:list[str],ventana:int=3) :
    logger.info(f"Comienzo feature media")

    if any(c.endswith("_mean") for c in all_cols):
        logger.info("Ya se hizo mean")
        return
    logger.info("Todavia no se hizo mean")
    sql = "Create or replace table df_init as "
    sql+="SELECT *"
    try:
        for attr in columnas:
            if attr in all_cols:
                sql+=f", avg(try_cast({attr} as double) ) over ventana_{ventana} as {attr}_mean"
            else :
                print(f"no se encontro el atributo {attr}")
        sql+=f" FROM df_init window ventana_{ventana} as (partition by numero_de_cliente order by foto_mes rows between {ventana} preceding and current row)"
    except Exception as e:
        logger.error(f"Error en la media : {e}")
        raise

    logging.info(f"QUERY: \n{sql}")

    conn = duckdb.connect(ctx.database)
    conn.execute(sql)
    conn.close()
