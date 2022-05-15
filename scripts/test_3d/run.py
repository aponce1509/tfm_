import sys
import torch
import torchvision.transforms as T
# setting path
sys.path.append('./scripts')
from exp_3d.params import params
from model_run import fix_seed, basic_pipeline
exp_name = "Comparación repre"

def main(torch_rand=True):
    fix_seed(params, torch_rand)
    basic_pipeline(params, experiment_name=exp_name)


# modelo inicial
print("Modelo inicial")
params['n_epochs'] = 10
params["model_name"] = "convnet3d"
params['conv_filters'] = (16, 24)
params['conv_sizes'] = (3, 3)
params['conv_strides'] = (1, 1)
params['pool_sizes'] = (2, 2)
params['pool_strides'] = (2, 2)
params['clf_n_layers'] = (3)
params['clf_neurons'] = (32, 8, 2)
params['avg_layer'] = "identity"
params['avg_layer_size'] = None
params['scheduler_step'] = 1
params['optim_lr'] = 0.001
for seed in params['seeds']:
    params['seed_'] = 123
    main(False)

# modelo denso
print("Modelo denso")
params["model_name"] = "convnet3d"
params['conv_filters'] = (32, 64, 64, 128)
params['conv_sizes'] = (3, 3, 3, 3)
params['conv_strides'] = (1, 1, 1, 1)
params['pool_sizes'] = (1, 2, 1, 2)
params['pool_strides'] = (1, 2, 1, 4)
params['clf_n_layers'] = (4)
params['clf_neurons'] = (50, 100, 50, 2)
params['optim_lr'] = 0.0001
for seed in params['seeds']:
    params['seed_'] = seed
    main(False)

# print('Eff b0')
# params["model_name"] = "effb0"
# params['clf_n_layers'] = (4)
# params['unfreeze_layers'] = 10
# params['clf_neurons'] = (32, 8, 2)
# params['transform'] = T.Compose([
#     T.Lambda(lambda x: x.repeat(3, 1, 1)),
#     # transforms.RandomResizedCrop((224, 224)),
#     # transforms.RandomHorizontalFlip(),
#     T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])
# params['optim_lr'] = 0.001
# for seed in params['seeds']:
#     params['seed_'] = seed
#     main()

# print('Eff b2')
# params["model_name"] = "effb0"
# params['clf_n_layers'] = (4)
# params['unfreeze_layers'] = 10
# params['clf_neurons'] = (32, 8, 2)
# params['optim_lr'] = 0.001
# params['transform'] = T.Compose([
#     T.Lambda(lambda x: x.repeat(3, 1, 1)),
#     # transforms.RandomResizedCrop((224, 224)),
#     # transforms.RandomHorizontalFlip(),
#     T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])
# for seed in params['seeds']:
#     params['seed_'] = seed
#     main()

# print('Eff b5')
# params["model_name"] = "effb0"
# params['clf_n_layers'] = (4)
# params['unfreeze_layers'] = 10
# params['clf_neurons'] = (32, 8, 2)
# params['optim_lr'] = 0.001
# params['transform'] = T.Compose([
#     T.Lambda(lambda x: x.repeat(3, 1, 1)),
#     # transforms.RandomResizedCrop((224, 224)),
#     # transforms.RandomHorizontalFlip(),
#     T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])
# for seed in params['seeds']:
#     params['seed_'] = seed
#     main()

# print('Eff b7')
# params["model_name"] = "effb7"
# params['clf_n_layers'] = (4)
# params['unfreeze_layers'] = 10
# params['clf_neurons'] = (32, 8, 2)
# params['optim_lr'] = 0.001
# params['transform'] = T.Compose([
#     T.Lambda(lambda x: x.repeat(3, 1, 1)),
#     # transforms.RandomResizedCrop((224, 224)),
#     # transforms.RandomHorizontalFlip(),
#     T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])
# for seed in params['seeds']:
#     params['seed_'] = seed
#     main()

# print('ResNet')
# params["model_name"] = "resnet50"
# params['clf_n_layers'] = (4)
# params['unfreeze_layers'] = 10
# params['clf_neurons'] = (32, 8, 2)
# params['optim_lr'] = 0.001
# params['transform'] = T.Compose([
#     T.Lambda(lambda x: x.repeat(3, 1, 1)),
#     # transforms.RandomResizedCrop((224, 224)),
#     # transforms.RandomHorizontalFlip(),
#     T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])
# for seed in params['seeds']:
#     params['seed_'] = seed
#     main()

# print('GoogleNet')
# params["model_name"] = "google"
# params['clf_n_layers'] = (4)
# params['unfreeze_layers'] = 10
# params['clf_neurons'] = (32, 8, 2)
# params['optim_lr'] = 0.001
# params['transform'] = T.Compose([
#     T.Lambda(lambda x: x.repeat(3, 1, 1)),
#     # transforms.RandomResizedCrop((224, 224)),
#     # transforms.RandomHorizontalFlip(),
#     T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])
# for seed in params['seeds']:
#     params['seed_'] = seed
#     main()
# # from _models import ConvNet
# # from torchsummary import summary
# # model = ConvNet(params)
# # model.to('cuda')
# # summary(model, (1, 62,  128))
