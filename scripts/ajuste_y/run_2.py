from model_run import fix_seed, basic_pipeline, optuna_pipeline
from exp_p_y.params import params
exp_name = "Pruebas modelo simple"

def main_optuna(n_trials, torch_rand=True):
    for seed in params['seeds']:
        params["seed_"] = seed
        fix_seed(params, torch_rand)
        optuna_pipeline(params, exp_name, n_trials=n_trials)

params['grid'] = {"grid": {
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
}
params["model_name"] = "effb2"
main_optuna(8)

params['grid'] = {"grid": {
    "unfreeze_layers":
    {
        "name": "unfreeze",
        "choices": [5, 10, 15, 20, 50, 100, 1000]
    },
    "lr_grid": 
    {
        "name": "lr",
        "low": 3,
        "high": 4
    }
}
}

params['clf_neurons'] = (32, 8, 2)
main_optuna(14)

params['clf_neurons'] = (100, 50, 2)
main_optuna(14)

params['clf_neurons'] = (256, 2)
main_optuna(14)

params['clf_neurons'] = (50, 2)
main_optuna(14)

