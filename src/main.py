import logging
from config.conf import load_config
from config.context import Context
from config import logger_config as lc
from infra import loader_utils as lu
from pipelines.challenge_one_pipeline import ChallengeOnePipeline

logger = logging.getLogger(__name__)

def main(experiment_name: str = "CHALLENGE_01") -> int:
    cfg = load_config(experiment_name)
    ctx = Context(cfg)

    lu.ensure_dirs(
        ctx.path_logs,
        ctx.path_lgbm_opt,
        ctx.path_lgbm_opt_best_params,
        ctx.path_lgbm_opt_db,
        ctx.path_lgbm_model,
        ctx.path_prediction,
        ctx.path_graphics,
    )

    lc.setup_logging(ctx.path_logs)
    logger.info("Starting Challenge One Pipeline...", extra={"experiment": experiment_name})

    # Pipeline principal
    pipeline = ChallengeOnePipeline(ctx)
    results = pipeline.run()

    if ctx.top_n is not None:
        pipeline.kaggle_prediction(top_n=ctx.top_n)

    logger.info("Pipeline completed successfully", extra={"results": results})
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        logger.exception("Unhandled exception running Challenge One Pipeline")
        raise
