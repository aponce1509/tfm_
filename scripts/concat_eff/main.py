import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from model_run import basic_pipeline, fix_seed
from utils_study_res import get_model
from params import params
import torchvision.transforms as T

def main():
    fix_seed(params)
    basic_pipeline(params, experiment_name=exp_name)

if __name__ == "__main__":
    params['runs_id'] = ['dfsdfsdf', '']
    params['params_multi'] = [[get_model(run_id)[1] for run_id in params['runs_id']]]
    params['transform_multi'] = T.Compose([
        T.Lambda(lambda x: x.repeat(3, 1, 1)),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    params['clf_neurons'] = (256, 2)
    params['optim_lr'] = 0.001
    exp_name = "Concat Eff"
    main()



