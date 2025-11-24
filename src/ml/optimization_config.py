
from dataclasses import dataclass
from config.context import Context

@dataclass(slots=True)
class OptimizationConfig:
    n_trials: int
    name: str

    gain_amount: int
    cost_amount: int

    n_folds: int
    n_boosts: int
    seeds: list[int]
    output_path: str
    training_db_path: str
    training_db_name: str


    @classmethod
    def from_context(cls, ctx: Context):
        return cls(
            n_trials=ctx.lgbm_n_trials,
            name=ctx.study_name,

            gain_amount=ctx.gain_amount,
            cost_amount=ctx.cost_amount,
            
            n_folds=ctx.lgbm_n_folds,
            n_boosts=ctx.lgbm_n_boosts,
            seeds=ctx.seeds,
            output_path=ctx.path_lgbm_opt,
            training_db_path=ctx.path_training_db,
            training_db_name=ctx.training_db
        )
