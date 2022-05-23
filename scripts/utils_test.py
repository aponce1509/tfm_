import torchvision.transforms as T

def modelo_simple(main, params: dict):
    params_copy = params.copy()
    params_copy['n_epochs'] = 10
    params_copy["model_name"] = "convnet"
    params_copy['conv_filters'] = (16, 24)
    params_copy['conv_sizes'] = (3, 3)
    params_copy['conv_strides'] = (1, 1)
    params_copy['pool_sizes'] = (2, 2)
    params_copy['pool_strides'] = (2, 2)
    params_copy['clf_n_layers'] = (3)
    params_copy['clf_neurons'] = (32, 8, 2)
    for seed in params_copy['seeds']:
        params_copy['seed_'] = seed
        main(params_copy)

def modelo_complejo(main, params: dict, lr=0.0001):
    params_copy = params.copy()
    params_copy["model_name"] = "convnet"
    params_copy['conv_filters'] = (32, 64, 64, 128)
    params_copy['conv_sizes'] = (3, 3, 3, 3)
    params_copy['conv_strides'] = (1, 1, 1, 1)
    params_copy['pool_sizes'] = (1, 2, 1, 2)
    params_copy['pool_strides'] = (1, 2, 1, 2)
    params_copy['clf_n_layers'] = (4)
    params_copy['clf_neurons'] = (50, 100, 50, 2)
    params_copy['optim_lr'] = lr
    for seed in params_copy['seeds']:
        params_copy['seed_'] = seed
        main(params_copy)

def eff_0(main, params: dict, lr=0.001):
    params_copy = params.copy()
    params_copy["model_name"] = "effb0"
    params_copy['clf_n_layers'] = (4)
    params_copy['unfreeze_layers'] = 10
    params_copy['clf_neurons'] = (32, 8, 2)
    params_copy['transform'] = T.Compose([
        T.Lambda(lambda x: x.repeat(3, 1, 1)),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    params_copy['optim_lr'] = lr
    for seed in params_copy['seeds']:
        params_copy['seed_'] = seed
        main(params_copy)

def eff_2(main, params: dict, lr=0.001):
    params_copy = params.copy()
    params_copy["model_name"] = "effb2"
    params_copy['clf_n_layers'] = (4)
    params_copy['unfreeze_layers'] = 10
    params_copy['clf_neurons'] = (32, 8, 2)
    params_copy['transform'] = T.Compose([
        T.Lambda(lambda x: x.repeat(3, 1, 1)),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    params_copy['optim_lr'] = lr
    for seed in params_copy['seeds']:
        params_copy['seed_'] = seed
        main(params_copy)

def eff_5(main, params: dict, lr=0.001):
    params_copy = params.copy()
    params_copy["model_name"] = "effb5"
    params_copy['clf_n_layers'] = (4)
    params_copy['unfreeze_layers'] = 10
    params_copy['clf_neurons'] = (32, 8, 2)
    params_copy['transform'] = T.Compose([
        T.Lambda(lambda x: x.repeat(3, 1, 1)),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    params_copy['optim_lr'] = lr
    for seed in params_copy['seeds']:
        params_copy['seed_'] = seed
        main(params_copy)

def eff_7(main, params: dict, lr=0.001):
    params_copy = params.copy()
    params_copy["model_name"] = "effb7"
    params_copy['clf_n_layers'] = (4)
    params_copy['unfreeze_layers'] = 10
    params_copy['clf_neurons'] = (32, 8, 2)
    params_copy['transform'] = T.Compose([
        T.Lambda(lambda x: x.repeat(3, 1, 1)),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    params_copy['optim_lr'] = lr
    for seed in params_copy['seeds']:
        params_copy['seed_'] = seed
        main(params_copy)
