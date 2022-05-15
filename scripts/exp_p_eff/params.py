#%%
import torchvision.transforms as transforms
from torchvision import models
# Model
model_params = {
    "model_name": "effb0",
    'bn': False,
    "input_shape": (224, 224),
    "clf_neurons": (32, 8, 2),
    "clf_no_linear_fun": "relu",
    "dropout": 0.0,
    "optim_name": "adam",
    "optim_lr": 0.001,
    "scheduler_name": "steplr",
    "scheduler_step_size": 3,
    "scheduler_gamma": 0.7,
    "criterion_name": "nllloss",
    "unfreeze_layers": 5,
    'log_trans': False
}
# base_model = models.resnet152(pretrained=True)
params = {
    "seeds": (3, ),
    "seed_": 1,
    # parámetros de datos
    "cube_shape_x": 1250,
    "win_shape": (224, 224, 224),
    "projection": "y",
    "projection_pool": "max",
    "cube_pool": "mean",
    "transform": transforms.Compose([
                transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                # transforms.RandomResizedCrop((224, 224)),
                # transforms.RandomHorizontalFlip(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
                 ]),
    "is_fast": True,
    # parámetros de entrenamiento de la red
    "batch_size": 48,
    "n_epochs": 7,
    "is_def": False,
}
grid = {"grid": {
    "optimizer_grid": {
        "name": "optim_name",
        "choices": ["adam", "adadelta"]
    },
    "lr_grid": {
        "name": "lr",
        "low": 1e-4,
        "high": 1e-1
    }
}
}

params = {**params, **model_params, **grid}