import sys
sys.path.append('./scripts')
from model_run import fix_seed, basic_pipeline, optuna_pipeline
from torchvision import transforms as T
from exp_p_y.params import params
exp_name = "Eff pruebas optim"

def main_optuna(n_trials, torch_rand=True):
    for seed in params['seeds']:
        params["seed_"] = seed
        fix_seed(params, torch_rand)
        optuna_pipeline(params, exp_name, n_trials=n_trials)

params['grid'] = {
    "optimizer_grid":
    {
        "name": "optim_name",
        "choices": ["adam", 'SGD']
    },
    "lr_grid": 
    {
        "name": "lr",
        "low": 2,
        "high": 5
    }
}

params['unfreeze_layers'] = 20
params['cube_shape_x'] = 2000
params["model_name"] = "effb0"
params['clf_neurons'] = (100, 50, 2)
params['transform_des'] = 'Normalización eff'
params['transform'] = T.Compose([
    T.Lambda(lambda x: x.repeat(3, 1, 1)),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
main_optuna(8)

# params['grid'] ={
#     "unfreeze_layers":
#     {
#         "name": "unfreeze",
#         "choices": [5, 10, 15, 20, 50, 100, 1000]
#     },
#     "lr_grid": 
#     {
#         "name": "lr",
#         "low": 3,
#         "high": 4
#     }
# }

exp_name = "Eff pruebas clf y layers"
params["model_name"] = "effb0"
# params['clf_neurons'] = (32, 8, 2)
# main_optuna(14)

# params['clf_neurons'] = (100, 50, 2)
# main_optuna(14)

# params['clf_neurons'] = (256, 2)
# main_optuna(14)

# params['clf_neurons'] = (50, 2)
# main_optuna(14)

params["model_name"] = "effb2"
# params['clf_neurons'] = (32, 8, 2)
# main_optuna(14)

# params['clf_neurons'] = (100, 50, 2)
# main_optuna(14)

# params['clf_neurons'] = (256, 2)
# main_optuna(14)

# params['clf_neurons'] = (50, 2)
# main_optuna(14)
params["model_name"] = "effb5"
# params['clf_neurons'] = (32, 8, 2)
# main_optuna(14)

# params['clf_neurons'] = (100, 50, 2)
# main_optuna(14)

# params['clf_neurons'] = (256, 2)
# main_optuna(14)

# params['clf_neurons'] = (50, 2)
# main_optuna(14)
params["model_name"] = "effb7"
# params['clf_neurons'] = (32, 8, 2)
# main_optuna(14)

# params['clf_neurons'] = (100, 50, 2)
# main_optuna(14)

# params['clf_neurons'] = (256, 2)
# main_optuna(14)

# params['clf_neurons'] = (50, 2)
# main_optuna(14)