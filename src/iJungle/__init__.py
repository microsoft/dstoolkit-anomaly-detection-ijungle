from iJungle import train, config

__version__ = config.__version__
_MODEL_DIR = config._MODEL_DIR

model_train_fun = train.model_train_fun
model_eval_fun = train.model_eval_fun
grid_train = train.grid_train
grid_eval = train.grid_eval
get_grid_eval_results = train.get_grid_eval_results
best_iforest = train.best_iforest
train_bundle = train.train_bundle
select_overhead_data = train.select_overhead_data
best_iforest_params = train.best_iforest_params
