import torchvision.transforms as T

model_params = {
    "model_name": "concat_eff",
    'bn': False,
    "clf_neurons": (32, 8, 2),
    "clf_no_linear_fun": "relu",
    "dropout": 0.0,
    "optim_name": "adam",
    "optim_lr": 0.001,
    "scheduler_name": "steplr",
    "scheduler_step_size": 3,  # 3
    "scheduler_gamma": 0.7,
    "criterion_name": "nllloss",
    'avg_layer': "identity",
    'avg_layer_size': None,  # (14, 30)
    "batch_size": 48,  # 48
    "n_epochs": 10,
}
params = {
    "seed_": 19,
    # parámetros de datos
}

params = {**params, **model_params}