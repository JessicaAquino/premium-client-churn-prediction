import src.infra.logger_wrapper as log
import duckdb

def create_ternary_class_and_weight():
    load_data()
    ternary_class_generation()
    count_generated_targets()
    ternary_class_to_binary()

@log.process_log
def load_data():
    query = f"""
        CREATE OR REPLACE TABLE df_init AS
        SELECT *
        FROM read_csv_auto('{})
    """
    logger.info(f"Creacion de la base de datos en : {PATH_DATA_BASE_DB}")
    sql = f"""
    create or replace table df_inicial as 
    select *
    from read_csv_auto('{FILE_INPUT_DATA_CRUDO_2}')"""
    if COMPETENCIA ==3 : 
        sql+=f""" UNION 
        select * 
        from read_csv_auto('{FILE_INPUT_DATA_CRUDO_3}')"""
    
    logger.info(f"El query es : {sql}")

    conn=duckdb.connect(PATH_DATA_BASE_DB)
    conn.execute(sql)
    conn.close()