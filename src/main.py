import logging

import config.conf as cf
import config.logger_config as lc
import config.sync_datasets as sd
import core.preprocessing_duckdb as pp
import infra.loader_utils as lu

from config.context import Context


logger = logging.getLogger(__name__)


def main(experiment_name: str = "CHALLENGE_03"):
    cfg = cf.load_config(experiment_name)
    ctx = Context(cfg)

    lu.ensure_dirs(
        ctx.path_logs,
        ctx.path_datasets,
        ctx.output
    )
    lc.setup_logging(ctx.path_logs)
    
    logger.info("Main execution started")

    for pr in ctx.process:
        if(pr == "copy_datasets"):
            sd.copy_raw_datasets(ctx)
        if(pr == "preprocessing"):
            pp.create_ternary_class_and_weight(ctx)
        if(pr == "save_database"):
            sd.save_database(ctx)
    
    logger.info("Main execution finished")

if __name__ == "__main__":
    main()
