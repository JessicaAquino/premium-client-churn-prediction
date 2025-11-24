import config.conf as cf

class Context:
    def __init__(self, cfg: dict):
        paths = cfg.get('PATHS', {})
        files = cfg.get('FILES', {})

        # Experiment values
        self.study_name        = cfg.get('STUDY_NAME')
        self.seeds             = cfg.get('SEEDS')
        self.month_train       = cfg.get('MONTH_TRAIN')
        self.month_validation  = cfg.get('MONTH_VALIDATION')
        self.month_test        = cfg.get('MONTH_TEST')

        self.gain_amount       = cfg.get('GAIN')
        self.cost_amount       = cfg.get('COST')
        self.binary_positives  = cfg.get('BINARY_POSITIVES')
        self.top_n  = cfg.get('TOP_N')

        # LightGBM optimization hyperparams
        self.lgbm_n_trials     = cfg.get('LGBM_N_TRIALS')
        self.lgbm_n_folds      = cfg.get('LGBM_N_FOLDS')
        self.lgbm_n_boosts     = cfg.get('LGBM_N_BOOSTS')
        self.lgbm_threshold    = cfg.get('LGBM_THRESHOLD')

        # Paths
        self.path_logs         = paths.get('LOGS')
        self.path_training_db  = paths.get('TRAINING_DB')
        self.path_data         = paths.get('INPUT_DATA')
        self.path_lgbm_opt     = paths.get('OUTPUT_LGBM_OPTIMIZATION')
        self.path_lgbm_opt_best_params = paths.get('OUTPUT_LGBM_OPTIMIZATION_BEST_PARAMS')
        self.path_lgbm_opt_db      = paths.get('OUTPUT_LGBM_OPTIMIZATION_DB')
        self.path_lgbm_model   = paths.get('OUTPUT_LGBM_MODEL')
        self.path_prediction   = paths.get('OUTPUT_PREDICTION')
        self.path_graphics     = paths.get('OUTPUT_GRAPHICS')
        self.path_sql          = paths.get('SQL')

        # Files
        self.input_dataset     = files.get('INPUT_DATASET')
        self.training_db         = files.get('TRAINING_DB')

        self._validate_paths(["path_logs", "path_data", "path_prediction"])

    def _validate_paths(self, required_keys: list[str]):
        for key in required_keys:
            value = getattr(self, key, None)
            if not value:
                raise ValueError(f"Missing required path in config: {key}")
