import sys
sys.path.append('./scripts')
from model_run import fix_seed, basic_pipeline, optuna_pipeline
from exp_p_y.params import params
# exp_name = "Pruebas kernel simple"
exp_name = 'Pruebas tamaño/resolución'

def main(params, torch_rand=True):
    fix_seed(params, torch_rand)
    basic_pipeline(params, experiment_name=exp_name)

# params['n_epochs'] = 10
# params["model_name"] = "convnet"
# params['conv_filters'] = (16, 24)
# params['cube_shape_x'] = 1500
# params['win_shape'] = (62, 62, 128)
# params['pool_sizes'] = (2, 2)
# params['pool_strides'] = (2, 2)
# params['clf_n_layers'] = (3)
# params['clf_neurons'] = (32, 8, 2)
# params['seed'] = (19, )

# kss = [5]
# kds = [1, 2]
# lrs = [0.001, 0.0001]
# steps = [0.7, 0.8, 1]
# for ks in kss:
#     for kd in kds:
#         for lr in lrs:
#             for step in steps:
#                 params['conv_sizes'] = (ks, ks)
#                 params['conv_strides'] = (kd, kd)
#                 params['optim_lr'] = lr
#                 params['scheduler_gamma'] = step
#                 main(params)


params['n_epochs'] = 10
params["model_name"] = "convnet "
params['cube_shape_x'] = 1500
params['win_shape'] = (62, 62, 128)
params['conv_filters'] = (32, 64, 64, 128)
params['pool_sizes'] = (1, 2, 1, 2)
params['pool_strides'] = (1, 2, 1, 2)
params['clf_neurons'] = (50, 100, 50, 2)
params['batch_size'] = 74
params['scheduler_step_size'] = 3
params['seed'] = (19, )

kss = [3, ]
kds = [1, ]
lrs = [0.0001]
steps = [0.7, ]
c_shapes = [250, 500, 1000, 1500, 2000]
for ks in kss:
    for kd in kds:
        for lr in lrs:
            for step in steps: 
                for c_shape in c_shapes:
                    if ks == 5:
                        params['batch_size'] = 48
                    elif ks == 3:
                        params['batch_size'] = 74
                    params['conv_sizes'] = (ks, ks, ks, ks)
                    params['conv_strides'] = (kd, kd, kd, kd)
                    params['optim_lr'] = lr
                    params['cube_shape_x'] = c_shape
                    params['scheduler_gamma'] = step
                    main(params)

# main_optuna(50)