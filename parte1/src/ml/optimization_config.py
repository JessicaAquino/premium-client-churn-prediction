
from dataclasses import dataclass

@dataclass
class OptimizationConfig:
    n_trials: int
    name: str

    gain_amount: int
    cost_amount: int

    n_folds: int
    n_boosts: int
    seeds: list[int]
    output_path: str
