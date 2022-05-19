import sys
sys.path.append('./scripts')
import torch
import torchvision.transforms as T
# setting path
from exp_p_y.params import params
from utils_test import *
from model_run import fix_seed, basic_pipeline, optuna_pipeline
exp_name = "Pruebas tamaño/resolución"

def main(params, torch_rand=True):
    fix_seed(params, torch_rand)
    basic_pipeline(params, experiment_name=exp_name)

def main_optuna(n_trials):
    for seed in params['seeds']:
        params["seed_"] = seed
        fix_seed(params)
        optuna_pipeline(params, exp_name, n_trials=n_trials)

# cx = 500
params['n_epochs'] = 10


params['cube_shape_x'] = 1500
# modelo_simple(main, params)
# modelo_complejo(main, params)
eff_0(main, params)
eff_2(main, params)
eff_5(main, params)
# # eff_7(main, params)

params['cube_shape_x'] = 2000
modelo_simple(main, params)
modelo_complejo(main, params)
eff_0(main, params)
eff_2(main, params)
eff_5(main, params)
# # eff_7(main, params)

params['cube_shape_x'] = 250
# modelo_simple(main, params)
# modelo_complejo(main, params)
# eff_0(main, params)
eff_2(main, params)
# eff_5(main, params)
# eff_7(main, params)

params['cube_shape_x'] = 500
# modelo_simple(main, params)
# modelo_complejo(main, params)
# eff_0(main, params)
eff_2(main, params)
# eff_5(main, params)
# # eff_7(main, params)
