import shutil
from pathlib import Path
from config.context import Context
import infra.logger_wrapper as log
import logging

logger = logging.getLogger(__name__)

@log.process_log
def copy_raw_datasets(ctx: Context):
    """
    Copy raw datasets from bucket to VM memory
    """
    
    bucket_base = Path(ctx.path_gcp_base) / ctx.path_datasets
    local_base  = Path.cwd() / ctx.path_datasets

    ds2 = ctx.file_dataset_02_raw
    ds3 = ctx.file_dataset_03_raw

    logger.info(f"Copying {ds2} ...")
    shutil.copy(bucket_base / ds2, local_base / ds2)

    logger.info(f"Copying {ds3} ...")
    shutil.copy(bucket_base / ds3, local_base / ds3)

    logger.info("All datasets copied successfully.")

    return True

@log.process_log
def save_database(ctx: Context):
    """
    Copy database from VM memory to bucket
    """

    bucket_base = Path(ctx.path_gcp_base) / ctx.path_datasets
    local_base  = Path.cwd() / ctx.path_datasets

    db_file = ctx.file_database

    shutil.copy(local_base / db_file, bucket_base / db_file)

    return True
