import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from model_run import basic_pipeline, optuna_pipeline, test_model, final_test_pipeline, fix_seed
from params import params
exp_name = "Projection_y"

def main():
    fix_seed(params)
    basic_pipeline(params, experiment_name=exp_name)
def main_optuna(n_trials):
    optuna_pipeline(params, exp_name, n_trials=n_trials)

if __name__ == "__main__":
    main()
    # main_optuna(n_trials=2)
    # run = 'runs:/b75ac16ef590427abb3e2c1d2caa45ea/convnet_20220308_002130'
    # test_model("b75ac16ef590427abb3e2c1d2caa45ea", run)
    # final_test_pipeline(params)
