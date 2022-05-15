import sys
# setting path
sys.path.append('./scripts')
from exp_p_y_color.params import params
import torchvision.transforms as T
from model_run import fix_seed, basic_pipeline
exp_name = "Comparación repre"

def main(torch_rand=True):
    fix_seed(params, torch_rand)
    basic_pipeline(params, experiment_name=exp_name)

# # modelo simple
# print("Modelo simple")
# params["model_name"] = "convnet"
# params['conv_filters'] = (32, )
# params['conv_sizes'] = (3, )
# params['conv_strides'] = (1, )
# params['pool_sizes'] = (2, )
# params['pool_strides'] = (2, )
# params['clf_n_layers'] = (2, )
# params['clf_n_layers'] = (2, )
# params['avg_layer'] = "adp"
# params['avg_layer_size'] = (14, 30)

# params['clf_neurons'] = (25, 50, 2)
# for seed in params['seeds']:
#     params['seed_'] = seed
#     main(False)

# modelo inicial
# print("Modelo inicial")
# params['n_epochs'] = 10
# params["model_name"] = "convnet"
# params['conv_filters'] = (16, 24)
# params['conv_sizes'] = (3, 3)
# params['conv_strides'] = (1, 1)
# params['pool_sizes'] = (2, 2)
# params['pool_strides'] = (2, 2)
# params['clf_n_layers'] = (3)
# params['clf_neurons'] = (32, 8, 2)
# params['avg_layer'] = "identity"
# params['avg_layer_size'] = None
# params['optim_lr'] = 0.0001
# params['batch_size'] = 64
# params['scheduler_step_size'] = 5
# params['dropout'] = 0.2
# for seed in params['seeds']:
#     params['seed_'] = seed
#     main()
# # for i in (0.0001, 0.01, 0.1):
# #     params['optim_lr'] = i
# #     main()

# # modelo denso
# print("Modelo denso")
# params["model_name"] = "convnet"
# params['conv_filters'] = (32, 64, 64, 128)
# params['conv_sizes'] = (3, 3, 3, 3)
# params['conv_strides'] = (1, 1, 1, 1)
# params['pool_sizes'] = (1, 2, 1, 2)
# params['pool_strides'] = (1, 2, 1, 4)
# params['clf_n_layers'] = (4)
# params['clf_neurons'] = (50, 100, 50, 2)
# params['optim_lr'] = 0.0001
# for seed in params['seeds']:
#     params['seed_'] = seed
#     main()

# print('Eff b0')
# params["model_name"] = "effb0"
# params['clf_n_layers'] = (4)
# params['unfreeze_layers'] = 10
# params['clf_neurons'] = (32, 8, 2)
# params['optim_lr'] = 0.001
# for seed in params['seeds']:
#     params['seed_'] = seed
#     main()

print('Eff b2')
params["model_name"] = "effb2"
params['clf_n_layers'] = (4)
params['unfreeze_layers'] = 10
params['clf_neurons'] = (32, 8, 2)
params['optim_lr'] = 0.001
for seed in params['seeds']:
    params['seed_'] = seed
    main()

print('Eff b5')
params["model_name"] = "effb5"
params['clf_n_layers'] = (4)
params['unfreeze_layers'] = 10
params['clf_neurons'] = (32, 8, 2)
params['optim_lr'] = 0.001
for seed in params['seeds']:
    params['seed_'] = seed
    main()

# print('Eff b7')
# params["model_name"] = "effb7"
# params['clf_n_layers'] = (4)
# params['unfreeze_layers'] = 10
# params['clf_neurons'] = (32, 8, 2)
# params['optim_lr'] = 0.001
# for seed in params['seeds']:
#     params['seed_'] = seed
#     main()

# print('ResNet')
# params["model_name"] = "resnet50"
# params['clf_n_layers'] = (4)
# params['unfreeze_layers'] = 10
# params['clf_neurons'] = (32, 8, 2)
# params['optim_lr'] = 0.001
# for seed in params['seeds']:
#     params['seed_'] = seed
#     main()

print('GoogleNet')
params["model_name"] = "google"
params['clf_n_layers'] = (4)
params['unfreeze_layers'] = 10
params['clf_neurons'] = (32, 8, 2)
params['optim_lr'] = 0.001
for seed in params['seeds']:
    params['seed_'] = seed
    main()
# from _models import ConvNet
# from torchsummary import summary
# model = ConvNet(params)
# model.to('cuda')
# summary(model, (1, 62,  128))
