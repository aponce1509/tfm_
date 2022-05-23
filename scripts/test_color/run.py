import sys
import torch
import torchvision.transforms as T
# setting path
sys.path.append('./scripts')
from exp_p_y_color.params import params
from model_run import fix_seed, basic_pipeline
exp_name = "Comparación repre dat"

def main(torch_rand=True):
    fix_seed(params, torch_rand)
    basic_pipeline(params, experiment_name=exp_name)


# modelo inicial
print("Modelo inicial")
params['n_epochs'] = 10
params["model_name"] = "convnet"
params['conv_filters'] = (16, 24)
params['conv_sizes'] = (3, 3)
params['conv_strides'] = (1, 1)
params['pool_sizes'] = (2, 2)
params['pool_strides'] = (2, 2)
params['clf_n_layers'] = (3)
params['clf_neurons'] = (32, 8, 2)
params['avg_layer'] = "identity"
params['avg_layer_size'] = None
params['optim_lr'] = 0.0001
params['batch_size'] = 48
params['dropout'] = 0.2
for seed in params['seeds']:
    params['seed_'] = 19
    main()

# modelo denso
print("Modelo denso")
params["model_name"] = "convnet"
params['conv_filters'] = (32, 64, 64, 128)
params['conv_sizes'] = (3, 3, 3, 3)
params['conv_strides'] = (1, 1, 1, 1)
params['pool_sizes'] = (1, 2, 1, 2)
params['pool_strides'] = (1, 2, 1, 2)
params['clf_n_layers'] = (4)
params['clf_neurons'] = (50, 100, 50, 2)
params['optim_lr'] = 0.0001
params['batch_size'] = 72
for seed in params['seeds']:
    params['seed_'] = 19
    main()


params['win_shape'] = (224, 62, 224)
params['batch_size'] = 48
params['scheduler_step_size'] = 3
params['clf_n_layers'] = (4)
params['unfreeze_layers'] = 10
params['clf_neurons'] = (32, 8, 2)
params['optim_lr'] = 0.001
print('Eff b0')
params["model_name"] = "effb0"
for seed in params['seeds']:
    params['seed_'] = 19
    main()

print('Eff b2')
params["model_name"] = "effb2"
for seed in params['seeds']:
    params['seed_'] = 19
    main()

print('Eff b5')
params["model_name"] = "effb5"
for seed in params['seeds']:
    params['seed_'] = 19
    main()

print('Eff b7')
params["model_name"] = "effb7"
for seed in params['seeds']:
    params['seed_'] = 19
    main()

print('ResNet')
params["model_name"] = "resnet50"
params['dropout'] = 0.2
for seed in params['seeds']:
    params['seed_'] = 19
    main()

print('GoogleNet')
params['dropout'] = 0.2
params["model_name"] = "google"
for seed in params['seeds']:
    params['seed_'] = 19
    main()