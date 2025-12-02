import logging
import duckdb

import infra.logger_wrapper as log
from config.context import Context

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Pipeline Orchestration
# ----------------------------------------------------------------------
@log.process_log
def create_ternary_class_and_weight(ctx: Context):
    """
    Full pipeline:
    1. Load raw data
    2. Generate ternary target
    3. Count ternary classes
    4. Convert to binary target + weights
    """
    load_raw_data(ctx)
    generate_ternary_target(ctx)
    count_ternary_classes(ctx)
    convert_to_binary_and_weights(ctx)


# ----------------------------------------------------------------------
# Step 1 — Load Raw Data
# ----------------------------------------------------------------------
@log.process_log
def load_raw_data(ctx: Context):
    """
    Load raw CSVs into DuckDB table df_init.
    """
    query = f"""
        CREATE OR REPLACE TABLE df_init AS
            SELECT *
            FROM read_csv_auto('{ctx.path_datasets + ctx.file_dataset_02_raw}')
        UNION
            SELECT *
            FROM read_csv_auto('{ctx.path_datasets + ctx.file_dataset_03_raw}');
    """

    logger.info(f"Executing query:\n{query}")

    conn = duckdb.connect(ctx.path_datasets + ctx.file_database)
    conn.execute(query)
    conn.close()


# ----------------------------------------------------------------------
# Step 2 — Generate Ternary Target
# ----------------------------------------------------------------------
@log.process_log
def generate_ternary_target(ctx: Context):

    macro_sql = """
        CREATE OR REPLACE MACRO clase_ternaria(t) AS TABLE (
            WITH base AS (
                SELECT *,
                    (CAST(foto_mes / 100 AS INTEGER) * 12 + (foto_mes % 100)) AS period_idx
                FROM t
            ),
            seq AS (
                SELECT *,
                    LEAD(period_idx, 1) OVER (
                        PARTITION BY numero_de_cliente ORDER BY period_idx
                    ) AS p1,
                    LEAD(period_idx, 2) OVER (
                        PARTITION BY numero_de_cliente ORDER BY period_idx
                    ) AS p2
                FROM base
            ),
            bounds AS (
                SELECT MAX(period_idx) AS maxp FROM seq
            ),
            result AS (
                SELECT
                    * EXCLUDE (period_idx, p1, p2),
                    CASE
                        WHEN period_idx < (SELECT maxp FROM bounds)
                            AND (p1 IS NULL OR p1 > period_idx + 1)
                        THEN 'BAJA+1'

                        WHEN period_idx < (SELECT maxp FROM bounds) - 1
                            AND p1 = period_idx + 1
                            AND (p2 IS NULL OR p2 > period_idx + 2)
                        THEN 'BAJA+2'

                        WHEN period_idx <= (SELECT maxp FROM bounds) - 2
                        THEN 'CONTINUA'

                        ELSE NULL
                    END AS clase_ternaria
                FROM seq
            )
            SELECT * FROM result
        );
    """


    apply_sql = """
        CREATE OR REPLACE TABLE df_init AS
        SELECT *
        FROM clase_ternaria(df_init);
    """

    conn = duckdb.connect(ctx.path_datasets + ctx.file_database)

    logger.info("Creating macro clase_ternaria...")
    conn.execute(macro_sql)

    logger.info("Applying macro clase_ternaria(df_init)...")
    conn.execute(apply_sql)

    conn.close()


# ----------------------------------------------------------------------
# Step 3 — Count Ternary Targets
# ----------------------------------------------------------------------
@log.process_log
def count_ternary_classes(ctx: Context):
    """
    Count how many BAJA+1, BAJA+2, CONTINUA per month.
    """
    query = """
        SELECT
            foto_mes,
            COUNT(*) FILTER (WHERE clase_ternaria = 'BAJA+1') AS BAJA1,
            COUNT(*) FILTER (WHERE clase_ternaria = 'BAJA+2') AS BAJA2,
            COUNT(*) FILTER (WHERE clase_ternaria = 'CONTINUA') AS CONTINUA
        FROM df_init
        GROUP BY foto_mes
        ORDER BY foto_mes;
    """

    conn = duckdb.connect(ctx.path_datasets + ctx.file_database)
    df_count = conn.execute(query).df()
    conn.close()

    logger.info(f"Ternary target count:\n{df_count}")


# ----------------------------------------------------------------------
# Step 4 — Convert to Binary Target + Weights
# ----------------------------------------------------------------------
@log.process_log
def convert_to_binary_and_weights(ctx: Context):
    """
    Create duckdb table df_init with:
    - clase_binary        (0/1)
    - clase_binary2       (BAJA+2 indicator)
    - clase_weight        (weights)
    """
    conn = duckdb.connect(ctx.path_datasets + ctx.file_database)

    # Create binary target + weights
    sql_create = """
        CREATE OR REPLACE TABLE df_init AS
        SELECT
            *,
            CASE
                WHEN clase_ternaria = 'BAJA+2' THEN 1.00002
                WHEN clase_ternaria = 'BAJA+1' THEN 1.00001
                ELSE 1.0
            END AS clase_weight,
            CASE
                WHEN clase_ternaria = 'CONTINUA' THEN 0
                ELSE 1
            END AS clase_binary,
            CASE
                WHEN clase_ternaria = 'BAJA+2' THEN 1
                ELSE 0
            END AS clase_binary2
        FROM df_init;
    """

    conn.execute(sql_create)

    # Count binary distribution
    sql_count = """
        SELECT
            foto_mes,
            COUNT(*) FILTER (WHERE clase_weight = 1.00002) AS weight_baja2,
            COUNT(*) FILTER (WHERE clase_weight = 1.00001) AS weight_baja1,
            COUNT(*) FILTER (WHERE clase_weight = 1.0)     AS weight_continua,
            COUNT(*) FILTER (WHERE clase_binary = 1)       AS binary_bajas,
            COUNT(*) FILTER (WHERE clase_binary = 0)       AS binary_continua,
            COUNT(*) FILTER (WHERE clase_binary2 = 1)      AS binary_baja2,
            COUNT(*) FILTER (WHERE clase_binary2 = 0)      AS binary_baja1_continua
        FROM df_init
        GROUP BY foto_mes
        ORDER BY foto_mes;
    """

    df_count = conn.execute(sql_count).df()
    conn.close()

    logger.info(f"Binary class and weights:\n{df_count}")
