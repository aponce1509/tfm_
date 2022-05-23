import sys
sys.path.append('./scripts')
import torch
import torchvision.transforms as T
# setting path
from exp_p_y.params import params
from utils_test import *
from model_run import fix_seed, basic_pipeline, optuna_pipeline
# exp_name = "Pruebas modelo complejo"
# exp_name = "Pruebas tamaño/resolución"
exp_name = "BORRAR"

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

params['seeds'] = (3, )
params['seed_'] = 3

params['cube_shape_x'] = 1500
params['win_shape'] = (62, 62, 128)
# params['batch_size'] = 
# modelo_complejo(main, params)
bs = 64
step = 3
gamma = 1.
lr = 0.0001
params['batch_size'] = bs
params['scheduler_gamma'] = gamma
params['scheduler_step_size'] = step

params['cube_shape_x'] = 1000
# params['win_shape'] = (62, 62, 128)
params['batch_size'] = 48
params['scheduler_gamma'] = 0.7
params['scheduler_step_size'] = 3
modelo_simple(main, params)

params['batch_size'] = bs
params['scheduler_gamma'] = gamma
params['scheduler_step_size'] = step
modelo_complejo(main, params)

params['cube_shape_x'] = 1500
# params['win_shape'] = (62, 62, 128)
params['batch_size'] = 48
params['scheduler_gamma'] = 0.7
params['scheduler_step_size'] = 3
modelo_simple(main, params)
params['batch_size'] = bs
params['scheduler_gamma'] = gamma
params['scheduler_step_size'] = step
modelo_complejo(main, params)


# params['cube_shape_x'] = 250
# # params['seeds'] = (19, )
# # params['seed_'] = 19
# # params['win_shape'] = (62, 62, 128)
# # modelo_simple(main, params)
# modelo_complejo(main, params)

# params['cube_shape_x'] = 500
# # params['win_shape'] = (62, 62, 128)
# # modelo_simple(main, params)
# modelo_complejo(main, params)



# params['cube_shape_x'] = 2000
# # params['win_shape'] = (62, 62, 128)
# # modelo_simple(main, params)
# modelo_complejo(main, params)

# params['win_shape'] = (224, 62, 224)
# eff_0(main, params)
# eff_2(main, params)
# eff_5(main, params)
# # eff_7(main, params)

params['cube_shape_x'] = 2000
# modelo_simple(main, params)
# modelo_complejo(main, params)
# params['win_shape'] = (224, 62, 224)
# eff_0(main, params)
# eff_2(main, params)
# eff_5(main, params)
# # eff_7(main, params)

params['cube_shape_x'] = 250
# modelo_simple(main, params)
# modelo_complejo(main, params)
# params['win_shape'] = (224, 62, 224)
# eff_0(main, params)
# eff_2(main, params)
# eff_5(main, params)
# eff_7(main, params)

# params['cube_shape_x'] = 500
# modelo_simple(main, params)
# modelo_complejo(main, params)
# params['win_shape'] = (224, 62, 224)
# eff_0(main, params)
# eff_2(main, params)
# eff_5(main, params)
# # eff_7(main, params)
