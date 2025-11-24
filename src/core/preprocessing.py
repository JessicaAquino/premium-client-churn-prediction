import polars as pl
import pandas as pd
import numpy as np
import infra.logger_wrapper as log

@log.process_log
def preprocessing_pipeline(
    df: pl.DataFrame,
    positives: list[str],
    month_train: list[int],
    month_test: int
) -> tuple[
    pd.DataFrame,        # X_train
    np.ndarray,          # y_train_binary
    np.ndarray,          # w_train
    pd.DataFrame,        # X_test
    np.ndarray,          # y_test_binary
    np.ndarray,          # y_test_class
    np.ndarray           # w_test
]:
    df = add_weight_class(df)
    df = add_binary_class(df, positives)
    X_train, y_train_binary, w_train, X_test, y_test_binary, y_test_class, w_test = split_test_train(df, month_train, month_test)
    return X_train.to_pandas(), y_train_binary, w_train, X_test.to_pandas(), y_test_binary, y_test_class, w_test


@log.process_log
def add_weight_class(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.when(pl.col("clase_ternaria") == "BAJA+2").then(1.00002)
        .when(pl.col("clase_ternaria") == "BAJA+1").then(1.00001)
        .otherwise(1.0)
        .alias("clase_peso")
    )

@log.process_log
def add_binary_class(df: pl.DataFrame, positives: list[str]) -> pl.DataFrame:
    return df.with_columns(
        pl.when(pl.col("clase_ternaria").is_in(positives)).then(1)
        .otherwise(0)
        .alias("clase_binaria")
    )

@log.process_log
def split_test_train(
    df: pl.DataFrame,
    month_train: list[int],
    month_test: int
) -> tuple[
    pl.DataFrame,  # X_train
    np.ndarray,    # y_train_binary
    np.ndarray,    # w_train
    pl.DataFrame,  # X_test
    np.ndarray,    # y_test_binary
    np.ndarray,    # y_test_class
    np.ndarray     # w_test
]:
    train_data = df.filter(pl.col("foto_mes").is_in(month_train))
    test_data = df.filter(pl.col("foto_mes") == month_test)
    
    X_train = train_data.drop(['clase_ternaria', 'clase_peso', 'clase_binaria'])
    y_train_binary = train_data['clase_binaria'].to_numpy()
    w_train = train_data['clase_peso'].to_numpy()

    X_test = test_data.drop(['clase_ternaria', 'clase_peso', 'clase_binaria'])
    y_test_binary = test_data['clase_binaria'].to_numpy()
    y_test_class = test_data['clase_ternaria'].to_numpy()
    w_test = test_data['clase_peso'].to_numpy()

    return X_train, y_train_binary, w_train, X_test, y_test_binary, y_test_class, w_test
