from src.ml.optimization_config import OptimizationConfig
import numpy as np
import lightgbm as lgb

class LightGBMObjective:
    def __init__(self, X_train, y_train, w_train, cfg: OptimizationConfig):
        self.X_train = X_train
        self.y_train = y_train
        self.w_train = w_train
        self.cfg = cfg

    def gan_eval(self, y_pred, data):
        weight = data.get_weight()
        gain = np.where(weight == 1.00002, self.cfg.gain_amount, 0)
        cost = np.where(weight < 1.00002, self.cfg.cost_amount, 0)

        ganancia = gain - cost
        ganancia = ganancia[np.argsort(y_pred)[::-1]]
        return 'gan_eval', np.max(np.cumsum(ganancia)), True
    
    def __call__(self, trial):
        params = {
            'objective': 'binary',
            'metric': 'custom',
            'boosting_type': 'gbdt',
            'max_bin': 31,
            'first_metric_only': True,
            'boost_from_average': True,
            'feature_pre_filter': False,
            'num_leaves': trial.suggest_int('num_leaves', 80, 150),
            'learning_rate': trial.suggest_float('learning_rate', 0.010, 0.2),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 400, 1000),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.1, 0.7),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.1, 0.4),
            'seed': self.cfg.seeds[0],
            'verbose': -1
        }

        train_data = lgb.Dataset(self.X_train,
                            label=self.y_train,
                            weight=self.w_train)
        
        cv_results = lgb.cv(
            params,
            train_data,
            num_boost_round=self.cfg.n_boosts,
            feval=self.gan_eval,
            stratified=True,
            nfold=self.cfg.n_folds,
            seed=self.cfg.seeds[0],
            callbacks=[
                lgb.early_stopping(stopping_rounds=int(50 + 5 / params['learning_rate']), verbose=False),
                lgb.log_evaluation(period=200)
            ]
        )

        max_gan = max(cv_results['valid gan_eval-mean'])
        trial.set_user_attr("best_iter", cv_results['valid gan_eval-mean'].index(max_gan) + 1)
        return max_gan * self.cfg.n_folds
