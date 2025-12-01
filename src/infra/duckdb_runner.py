import duckdb
import polars as pl

# TODO: Delete if not used

def run_duckdb_query(df: pl.DataFrame, sql: str) -> pl.DataFrame:
    """Executes a DuckDB SQL query over a DataFrame and returns the result."""
    with duckdb.connect(database=":memory:") as con:
        con.register("df", df)
        result = con.execute(sql).pl()
    return result
