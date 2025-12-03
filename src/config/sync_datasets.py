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

@log.process_log
def delete_raw_datasets(ctx: Context):
    """
    Delete local raw datasets after the DuckDB database has been created.
    Only operates on the VM's datasets folder.
    """

    local_base = Path.cwd() / ctx.path_datasets

    ds2 = ctx.file_dataset_02_raw
    ds3 = ctx.file_dataset_03_raw

    # DS2
    try:
        target = local_base / ds2
        if target.exists():
            logger.info(f"Deleting local file {ds2} ...")
            target.unlink()
        else:
            logger.warning(f"File {ds2} does not exist locally. Skipping deletion.")
    except Exception as e:
        logger.error(f"Error deleting {ds2}: {e}")

    # DS3
    try:
        target = local_base / ds3
        if target.exists():
            logger.info(f"Deleting local file {ds3} ...")
            target.unlink()
        else:
            logger.warning(f"File {ds3} does not exist locally. Skipping deletion.")
    except Exception as e:
        logger.error(f"Error deleting {ds3}: {e}")

    logger.info("Raw dataset deletion completed.")
    return True
