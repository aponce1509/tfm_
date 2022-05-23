import sys
sys.path.append('./scripts')
from model_run import fix_seed, basic_pipeline, optuna_pipeline
from torchvision import transforms as T
from exp_p_y.params import params

def main(params, torch_rand=True):
    fix_seed(params, torch_rand)
    basic_pipeline(params, experiment_name=exp_name)

exp_name = "Eff pruebas clf y layers"
# General params
params['cube_shape_x'] = 3000
params['win_shape'] = (224, 224, 224)
params['projection'] = 'y'
params['transform_des'] =   'Normalización efficientnet'
params['transform'] = T.Compose([
    T.Lambda(lambda x: x.repeat(3, 1, 1)),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# grid 1
# params['model_name'] = 'effb2'
# params['clf_neurons'] = (256, 2)
# params['unfreeze_layers'] = 50
# lrs = [0.01, 0.001, 0.0001]
# steps = [0.5, 0.8, 1.0]
# for lr in lrs:
#     for step in steps:
#         params['optim_lr'] = lr
#         params['scheduler_gamma'] = step
#         main(params)

# grid 2
# clf_neurons = [(256, 2), (32, 8, 2), (100, 50, 2), (50, 2)]
clf_neurons = [(512, 2), ]
params['n_epochs'] = 10
models_name = ['effb5', ]
params['scheduler_gamma'] = 0.5
params['dropout'] = 0.3
unfreezes = [100, ]
lrs = [0.01, ]
for clf_neuron in clf_neurons:
    for model_name in models_name:
        for lr in lrs:
            for unfreeze in unfreezes:
                params['optim_lr'] = lr
                params['unfreeze_layers'] = unfreeze
                params['model_name'] = model_name
                params['clf_neurons'] = clf_neuron
                main(params)