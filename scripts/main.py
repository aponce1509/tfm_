import sys
from model_run import basic_pipeline, optuna_pipeline, fix_seed
from ensemble import ensemble_pipeline
# Cambiar carpeta desde donde se importan los parámetros y el nombre del 
# experimento

if sys.argv[1] == "Projection_y":
    from exp_p_y.params import params
    exp_name = "Projection_y"
elif sys.argv[1] == "Projection_y_optim_test":
    from exp_p_y.params import params
    exp_name = "Projection_y optim test"
elif sys.argv[1] == "ColorConvnet":
    from exp_p_y_color.params import params
    exp_name = "ColorConvnet"
elif sys.argv[1] == "ResNet":
    from exp_p_resnet.params import params
    exp_name = "ResNet"
elif sys.argv[1] == "EffNet":
    from exp_p_eff.params import params
    exp_name = "EffNet"
elif sys.argv[1] == "3D":
    from exp_3d.params import params
    exp_name = "3D"
elif sys.argv[1] == "def":
    from exp_def.params import params
elif sys.argv[1] == "ensemble_projections":
    exp_name = "Ensemble Projections"
    from exp_ensemble_pro.params import runs_id, model_name
else:
    raise Exception("Not valid experiment")


def main():
    fix_seed(params)
    basic_pipeline(params, experiment_name=exp_name)

def main_optuna(n_trials):
    for seed in params['seeds']:
        params["seed_"] = seed
        fix_seed(params)
        optuna_pipeline(params, exp_name, n_trials=n_trials)
        
def main_ensemble():
    ensemble_pipeline(runs_id, exp_name, model_name)

# TODO system var para que mire si optuna o main
if __name__ == "__main__":
    # main_optuna(10)
    if sys.argv[1] == "ensemble_projections":
        main_ensemble()
    elif sys.argv[2] == "optuna":
        main_optuna(sys.argv[3])
    else:
        main()