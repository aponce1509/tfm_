import sys
sys.path.append('./scripts')
from model_run import fix_seed, basic_pipeline, optuna_pipeline
from exp_p_y.params import params
exp_name = "Pruebas kernel simple"

def main_optuna(n_trials, exp_name, torch_rand=True):
    for seed in params['seeds']:
        params["seed_"] = seed
        fix_seed(params, torch_rand)
        optuna_pipeline(params, exp_name, n_trials=n_trials)

params['grid'] = {"optimizer_grid":
    {
        "name": "optim_name",
        "choices": ["adam"]
    },
    # "lr_grid": 
    # {
    #     "name": "lr",
    #     "low": 3,
    #     "high": 4
    # },
    'kernel_size':
    {
        'name': 'kernel_size',
        'low': 3,
        'high': 5
    },
    'kernel_stride':
    {
        'name': 'kernel_stride',
        'low': 1,
        'high': 4
    },
    'scheduler_gamma':
    {
        'name': 'gamma',
        'low': 6,
        'high': 10
    }
}

params['n_epochs'] = 10
params["model_name"] = "convnet"
params['conv_filters'] = (16, 24)
params['conv_sizes'] = (3, 3)
params['conv_strides'] = (1, 1)
params['pool_sizes'] = (2, 2)
params['pool_strides'] = (2, 2)
params['clf_n_layers'] = (3)
params['clf_neurons'] = (32, 8, 2)
params['seed'] = (19, )

main_optuna(30, exp_name)

# params['n_epochs'] = 10
# params['seed'] = (2, 3, 19)
# params["model_name"] = "convnet"
# params['conv_filters'] = (32, 64, 64, 128)
# params['conv_sizes'] = (3, 3, 3, 3)
# params['conv_strides'] = (1, 1, 1, 1)
# params['pool_sizes'] = (1, 2, 1, 2)
# params['pool_strides'] = (1, 2, 1, 4)
# params['clf_n_layers'] = (4)
# params['clf_neurons'] = (50, 100, 50, 2)

# main_optuna(50)