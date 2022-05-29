# %%
import torch
from torchsummary import summary
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import mlflow
import pickle
from paths import *
import os

from torch import optim

torch.use_deterministic_algorithms(True)

def get_model_(run_id, is_enesemble=False):
    """
    Función que dado el id de la run a estudiar nos devuelve el modelo y los
    parámetros de dicho modelo
    Parameters
    ----------
    Return
    ----------
    params: dicicionario con los parámetros del modelo
    model: modelo listo para entrenar en nuevos datos
    """
    client = mlflow.tracking.MlflowClient(MODEL_PATH)
    # cargamos los parámetros
    local_dir = "temp/downloads"  # carpeta donde dejar el artifact
    if not os.path.exists(local_dir):
        os.mkdir(local_dir)

    mlflow.set_tracking_uri(MODEL_PATH)
    # obtenemos el modelo
    if is_enesemble:
        # cargamos el artifact
        local_path = client.download_artifacts(run_id, "runs_id", local_dir)
        with open(local_path, "rb") as file:
            runs_id = pickle.load(file)
        model_log = f"runs:/{run_id}/model"
        model = mlflow.sklearn.load_model(model_log)
        return runs_id, model
    else:
        # cargamos el artifact
        local_path = client.download_artifacts(run_id, "params", local_dir)
        with open(local_path, "rb") as file:
            params = pickle.load(file)
        model_log = f"runs:/{run_id}/{params['model_name']}_{params['_id']}"
        model = mlflow.pytorch.load_model(model_log)
        return params, model

def get_criterion(model_params):
    """
    Función que a partir de los parámetros del experimento devuelva el criterion.
    Parameters
    ----------
    model_params: dict. Diccionario con las siguientes claves:
        * criterion_name, con el nombre del criterion empleado. Están 
        "crossentropyloss".
    Return
    ----------
    criterion
    """
    if model_params["criterion_name"] == "crossentropyloss":
        criterion = nn.CrossEntropyLoss()
    elif model_params["criterion_name"] == "nllloss":
        criterion = nn.NLLLoss()
    else:
        raise Exception("Crieterion not defined")
    return criterion

def get_gradient_elements(model_params: dict):
    """
    Funcion para obtener los elementos necesarios para el entrenamiento:
    Modelo, device (cpu o cuda), optimizador, criterion y scheduler.
    Parameters
    ----------
    model_params: dict. Dicionario con las siguientes claves:
        * model_name, con el nombre de la red. Estan "convnet" para una letnet
        simple y "convnet3d" que es una letnet simple pero que admite imágenes
        3d.
        * optim_name, con el nombre del optimizador. Estan "adam" y "adadelta"
        * optim_lr, con el learning rate del optimizador
        * scheduler_name, con el nombre del scheduler. Estan "steplr". Los 
        parámetros tienen la estructura de "schedules_{param_name}" por si 
        distintos scheduler tienen distintos parámetros.
        * criterion_name, con el nombre del criterion empleado. Estan 
        "crossentropyloss".
    Return
    ----------
    tuple: model, device, optimizer, scheduler, criterion
    """
    # Modelo
    if model_params["model_name"] == "convnet":
        model = ConvNet(model_params)
    elif model_params["model_name"] == "convnet_padd":
        model = ConvNetPadd(model_params)
    elif model_params["model_name"] == "convnet3d":
        model = ConvNet3D(model_params)
    elif model_params["model_name"] == "resnet50":
        model = ResNet50(model_params)
    elif model_params["model_name"] == "effb0":
        model = EffNetB0(model_params)
    elif model_params["model_name"] == "effb2":
        model = EffNetB2(model_params)
    elif model_params["model_name"] == "effb5":
        model = EffNetB5(model_params)
    elif model_params["model_name"] == "effb7":
        model = EffNetB7(model_params)
    elif model_params["model_name"] == "google":
        model = GoogleNet(model_params)
    elif model_params['model_name'] == 'concat':
        model = ConvNetConcat(model_params)
    elif model_params['model_name'] == 'concat_eff':
        model = ConcatEffModels(model_params)
    else:
        raise Exception("That model doesn't exist")
    # obtenemos device y cargamos el modelo al device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    # optimizador
    if model_params["optim_name"] == "adam":
        print("adam")
        optimizer = optim.Adam(model.parameters(), lr=model_params["optim_lr"])
    elif model_params["optim_name"] == "adadelta":
        optimizer = optim.Adadelta(
            model.parameters(), lr=model_params["optim_lr"]
        )
    elif model_params["optim_name"] == "SGD":
        print("SGD")
        optimizer = optim.SGD(
            model.parameters(), lr=model_params["optim_lr"]
        )
    else:
        raise Exception("Optimizer not defined")
    # Scheduler
    if model_params["scheduler_name"] == "steplr":
        scheduler = StepLR(
            optimizer,
            step_size=model_params["scheduler_step_size"],
            gamma=model_params["scheduler_gamma"]
        )
    elif model_params["scheduler_name"] == "None":
        scheduler = None
    else:
        raise Exception("Scheduler not defined")
    # criterion
    criterion = get_criterion(model_params)
    return model, device, optimizer, scheduler, criterion

class ConvNet(nn.Module):

    def __init__(self, params: dict) -> None:
        super(ConvNet, self).__init__()
        self.bn = params['bn']
        conv_block = self.make_conv_block(params)
        self.features = nn.Sequential(*conv_block[:-2])
        self.end_conv = nn.Sequential(*conv_block[-2:])
        if params['avg_layer'] == "identity":
            self.avgpool = nn.Identity()
        elif params['avg_layer'] == "adp":
            self.avgpool = nn.AdaptiveAvgPool2d(params['avg_layer_size'])
        else:
            raise Exception("not valid avg_layer")

        clf_block = self.make_clf_block(params)
        self.classifier = nn.Sequential(*clf_block)

    def make_conv_block(self, params: dict):
        conv_sizes = params['conv_sizes']
        conv_strides = params['conv_strides']
        pool_sizes = params['pool_sizes']
        pool_strides = params['pool_strides']
        channels_in = (params['n_channels'], *params['conv_filters'][:-1])
        channels_out = params['conv_filters']
        no_linear_fun = params["clf_no_linear_fun"]
        n_convs = len(conv_sizes)
        dropout_ratios = (params['dropout'], ) * (n_convs - 1) + (0, ) 
        conv_params = (conv_sizes, conv_strides, pool_sizes, pool_strides,
                       channels_in, channels_out, dropout_ratios)
        conv_block = []
        for c_size, c_stride, p_size, p_stride, chan_in, chan_out, d_r in zip(*conv_params):
            block = self.get_block(chan_in, chan_out, c_size, c_stride, p_size, 
                           p_stride, no_linear_fun, d_r)
            conv_block.extend(block)
        return tuple(conv_block)

    def get_block(self, channels_in, channels_out, conv_size, conv_stride, 
                  pool_size, pool_sride, no_linear_fun, dropout_ratio):
        conv = nn.Conv2d(channels_in, channels_out, conv_size, conv_stride)
        pool = nn.MaxPool2d(pool_size, pool_sride)
        dropout_ = nn.Dropout(dropout_ratio)
        if no_linear_fun == "relu":
            no_linear = nn.ReLU(inplace=True)
        elif no_linear_fun == "none":
            no_linear = nn.Identity()
        else:
            raise Exception("Not valid no_linear_fun")
        if self.bn:
            bn = nn.BatchNorm2d(channels_out)
            block = conv, no_linear, bn, pool, dropout_
        else:
            block = conv, no_linear, pool, dropout_
        return block

    def make_clf_block(self, params: dict):
        clf_neurons = params['clf_neurons']
        input_shape = (1, params['n_channels']) + params["input_shape"]
        n_layers = len(clf_neurons)
        # obtenemos la forma de la capa aplanada para obtener el nº de inputs 
        # para la primera capa del clasificador
        pre_cls_shape = self.convolutions(
            torch.rand(input_shape)
        ).data.shape[1]
        layers_in = (pre_cls_shape, *clf_neurons[:-1])
        layers_out = clf_neurons
        # creamos una tupla (len = número de capas) con el dropout tq el todos 
        # son iguales (dado por params) menos el último que lo hacemos 0
        # dropout_ratios = (0, ) + (params['dropout'], ) * (n_layers - 2) + (0, )  
        dropout_ratios = (params['dropout'], ) * (n_layers - 1) + (0, )  
        no_linear_fun = params["clf_no_linear_fun"]
        no_linear_funs = (no_linear_fun, ) * (n_layers - 1) + ("log_softmax", )
        clf_params = (layers_in, layers_out, dropout_ratios, no_linear_funs)
        clf_block = []
        
        for layer_in, layer_out, d_r, nl_fun in zip(*clf_params):
            block = self.get_layer_clf(layer_in, layer_out, d_r, nl_fun)
            clf_block.extend(block)
        return tuple(clf_block)

    def get_layer_clf(self, layers_in, layers_out, dropout_ratio,
                      no_linear_fun="relu"):
        layer = nn.Linear(layers_in, layers_out)
        dropout_ = nn.Dropout(dropout_ratio)
        # seleccion de no lineal
        if no_linear_fun == "relu":
            no_linear = nn.ReLU(inplace=True)
        elif no_linear_fun == "sigmoid":
            no_linear = nn.Sigmoid()
        elif no_linear_fun == "log_softmax":
            no_linear = nn.LogSoftmax(1)
        elif no_linear_fun == "none":
            no_linear = nn.Identity()
        else:
            raise Exception("Not valid no_linear_fun")
        if self.bn and no_linear_fun != "log_softmax" :
            bn = nn.BatchNorm1d(layers_out)
            fc = dropout_, layer, no_linear, bn
        else:
            fc = dropout_, layer, no_linear
        return fc        

    def convolutions(self, x):
        x = self.features(x)
        x = self.end_conv(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convolutions(x)
        x = self.classifier(x)
        return x

class ConvNetPadd(nn.Module):

    def __init__(self, params: dict) -> None:
        super(ConvNetPadd, self).__init__()
        self.bn = params['bn']
        conv_block = self.make_conv_block(params)
        self.features = nn.Sequential(*conv_block[:-2])
        self.end_conv = nn.Sequential(*conv_block[-2:])
        if params['avg_layer'] == "identity":
            self.avgpool = nn.Identity()
        elif params['avg_layer'] == "adp":
            self.avgpool = nn.AdaptiveAvgPool2d(params['avg_layer_size'])
        else:
            raise Exception("not valid avg_layer")

        clf_block = self.make_clf_block(params)
        self.classifier = nn.Sequential(*clf_block)

    def make_conv_block(self, params: dict):
        conv_sizes = params['conv_sizes']
        conv_strides = params['conv_strides']
        pool_sizes = params['pool_sizes']
        pool_strides = params['pool_strides']
        channels_in = (params['n_channels'], *params['conv_filters'][:-1])
        channels_out = params['conv_filters']
        no_linear_fun = params["clf_no_linear_fun"]
        n_convs = len(conv_sizes)
        dropout_ratios = (params['dropout'], ) * (n_convs - 1) + (0, ) 
        conv_params = (conv_sizes, conv_strides, pool_sizes, pool_strides,
                       channels_in, channels_out, dropout_ratios)
        conv_block = []
        for c_size, c_stride, p_size, p_stride, chan_in, chan_out, d_r in zip(*conv_params):
            block = self.get_block(chan_in, chan_out, c_size, c_stride, p_size, 
                           p_stride, no_linear_fun, d_r)
            conv_block.extend(block)
        return tuple(conv_block)

    def get_block(self, channels_in, channels_out, conv_size, conv_stride, 
                  pool_size, pool_sride, no_linear_fun, dropout_ratio):
        conv = nn.Conv2d(channels_in, channels_out, conv_size, conv_stride,
                         padding=5)
        pool = nn.MaxPool2d(pool_size, pool_sride)
        dropout_ = nn.Dropout(dropout_ratio)
        if no_linear_fun == "relu":
            no_linear = nn.ReLU(inplace=True)
        elif no_linear_fun == "none":
            no_linear = nn.Identity()
        else:
            raise Exception("Not valid no_linear_fun")
        if self.bn:
            bn = nn.BatchNorm2d(channels_out)
            block = conv, no_linear, bn, pool, dropout_
        else:
            block = conv, no_linear, pool, dropout_
        return block

    def make_clf_block(self, params: dict):
        clf_neurons = params['clf_neurons']
        input_shape = (1, params['n_channels']) + params["input_shape"]
        n_layers = len(clf_neurons)
        # obtenemos la forma de la capa aplanada para obtener el nº de inputs 
        # para la primera capa del clasificador
        pre_cls_shape = self.convolutions(
            torch.rand(input_shape)
        ).data.shape[1]
        layers_in = (pre_cls_shape, *clf_neurons[:-1])
        layers_out = clf_neurons
        # creamos una tupla (len = número de capas) con el dropout tq el todos 
        # son iguales (dado por params) menos el último que lo hacemos 0
        # dropout_ratios = (0, ) + (params['dropout'], ) * (n_layers - 2) + (0, )  
        dropout_ratios = (params['dropout'], ) * (n_layers - 1) + (0, )  
        no_linear_fun = params["clf_no_linear_fun"]
        no_linear_funs = (no_linear_fun, ) * (n_layers - 1) + ("log_softmax", )
        clf_params = (layers_in, layers_out, dropout_ratios, no_linear_funs)
        clf_block = []
        
        for layer_in, layer_out, d_r, nl_fun in zip(*clf_params):
            block = self.get_layer_clf(layer_in, layer_out, d_r, nl_fun)
            clf_block.extend(block)
        return tuple(clf_block)

    def get_layer_clf(self, layers_in, layers_out, dropout_ratio,
                      no_linear_fun="relu"):
        layer = nn.Linear(layers_in, layers_out)
        dropout_ = nn.Dropout(dropout_ratio)
        # seleccion de no lineal
        if no_linear_fun == "relu":
            no_linear = nn.ReLU(inplace=True)
        elif no_linear_fun == "sigmoid":
            no_linear = nn.Sigmoid()
        elif no_linear_fun == "log_softmax":
            no_linear = nn.LogSoftmax(1)
        elif no_linear_fun == "none":
            no_linear = nn.Identity()
        else:
            raise Exception("Not valid no_linear_fun")
        if self.bn and no_linear_fun != "log_softmax" :
            bn = nn.BatchNorm1d(layers_out)
            fc = dropout_, layer, no_linear, bn
        else:
            fc = dropout_, layer, no_linear
        return fc        

    def convolutions(self, x):
        x = self.features(x)
        x = self.end_conv(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convolutions(x)
        x = self.classifier(x)
        return x

class ConvNetConcat(nn.Module):

    def __init__(self, params: dict) -> None:
        super(ConvNetConcat, self).__init__()
        self.bn = params['bn']
        conv_block_x = self.make_conv_block(params)
        conv_block_y = self.make_conv_block(params)
        conv_block_z = self.make_conv_block(params)
        self.features_x = nn.Sequential(*conv_block_x[:-2])
        self.features_y = nn.Sequential(*conv_block_y[:-2])
        self.features_z = nn.Sequential(*conv_block_z[:-2])
        self.end_conv_x = nn.Sequential(*conv_block_x[-2:])
        self.end_conv_y = nn.Sequential(*conv_block_y[-2:])
        self.end_conv_z = nn.Sequential(*conv_block_z[-2:])
        if params['avg_layer'] == "identity":
            self.avgpool_x = nn.Identity()
            self.avgpool_y = nn.Identity()
            self.avgpool_z = nn.Identity()
        elif params['avg_layer'] == "adp":
            self.avgpool_x = nn.AdaptiveAvgPool2d(params['avg_layer_size'])
            self.avgpool_y = nn.AdaptiveAvgPool2d(params['avg_layer_size'])
            self.avgpool_z = nn.AdaptiveAvgPool2d(params['avg_layer_size'])
        else:
            raise Exception("not valid avg_layer")

        clf_block = self.make_clf_block(params)
        self.classifier = nn.Sequential(*clf_block)

    def make_conv_block(self, params: dict):
        conv_sizes = params['conv_sizes']
        conv_strides = params['conv_strides']
        pool_sizes = params['pool_sizes']
        pool_strides = params['pool_strides']
        channels_in = (1, *params['conv_filters'][:-1])
        channels_out = params['conv_filters']
        no_linear_fun = params["clf_no_linear_fun"]
        n_convs = len(conv_sizes)
        dropout_ratios = (params['dropout'], ) * (n_convs - 1) + (0, ) 
        conv_params = (conv_sizes, conv_strides, pool_sizes, pool_strides,
                       channels_in, channels_out, dropout_ratios)
        conv_block = []
        for c_size, c_stride, p_size, p_stride, chan_in, chan_out, d_r in zip(*conv_params):
            block = self.get_block(chan_in, chan_out, c_size, c_stride, p_size, 
                           p_stride, no_linear_fun, d_r)
            conv_block.extend(block)
        return tuple(conv_block)

    def get_block(self, channels_in, channels_out, conv_size, conv_stride, 
                  pool_size, pool_sride, no_linear_fun, dropout_ratio):
        conv = nn.Conv2d(channels_in, channels_out, conv_size, conv_stride)
        pool = nn.MaxPool2d(pool_size, pool_sride)
        dropout_ = nn.Dropout(dropout_ratio)
        if no_linear_fun == "relu":
            no_linear = nn.ReLU(inplace=True)
        elif no_linear_fun == "none":
            no_linear = nn.Identity()
        else:
            raise Exception("Not valid no_linear_fun")
        if self.bn:
            bn = nn.BatchNorm2d(channels_out)
            block = conv, no_linear, bn, pool, dropout_
        else:
            block = conv, no_linear, pool, dropout_
        return block

    def make_clf_block(self, params: dict):
        clf_neurons = params['clf_neurons']
        input_shape = params["input_shape"]
        n_layers = len(clf_neurons)
        # obtenemos la forma de la capa aplanada para obtener el nº de inputs 
        # para la primera capa del clasificador
        x = torch.rand(((1, 1) + input_shape[0]))
        y = torch.rand(((1, 1) + input_shape[1]))
        z = torch.rand(((1, 1) + input_shape[2]))

        pre_cls_shape = self.convolutions(
            x, y, z
        ).data.shape[1]
        layers_in = (pre_cls_shape, *clf_neurons[:-1])
        layers_out = clf_neurons
        # creamos una tupla (len = número de capas) con el dropout tq el todos 
        # son iguales (dado por params) menos el último que lo hacemos 0
        # dropout_ratios = (0, ) + (params['dropout'], ) * (n_layers - 2) + (0, )  
        dropout_ratios = (params['dropout'], ) * (n_layers - 1) + (0, )  
        no_linear_fun = params["clf_no_linear_fun"]
        no_linear_funs = (no_linear_fun, ) * (n_layers - 1) + ("log_softmax", )
        clf_params = (layers_in, layers_out, dropout_ratios, no_linear_funs)
        clf_block = []
        
        for layer_in, layer_out, d_r, nl_fun in zip(*clf_params):
            block = self.get_layer_clf(layer_in, layer_out, d_r, nl_fun)
            clf_block.extend(block)
        return tuple(clf_block)

    def get_layer_clf(self, layers_in, layers_out, dropout_ratio,
                      no_linear_fun="relu"):
        layer = nn.Linear(layers_in, layers_out)
        dropout_ = nn.Dropout(dropout_ratio)
        # seleccion de no lineal
        if no_linear_fun == "relu":
            no_linear = nn.ReLU(inplace=True)
        elif no_linear_fun == "sigmoid":
            no_linear = nn.Sigmoid()
        elif no_linear_fun == "log_softmax":
            no_linear = nn.LogSoftmax(1)
        elif no_linear_fun == "none":
            no_linear = nn.Identity()
        else:
            raise Exception("Not valid no_linear_fun")
        if self.bn and no_linear_fun != "log_softmax" :
            bn = nn.BatchNorm1d(layers_out)
            fc = dropout_, layer, no_linear, bn
        else:
            fc = dropout_, layer, no_linear
        return fc        

    def convolutions(self, x, y, z):
        x = self.features_x(x)
        x = self.end_conv_x(x)
        x = self.avgpool_x(x)
        x = torch.flatten(x, 1)
        y = self.features_y(y)
        y = self.end_conv_y(y)
        y = self.avgpool_y(y)
        y = torch.flatten(y, 1)
        z = self.features_z(z)
        z = self.end_conv_z(z)
        z = self.avgpool_z(z)
        z = torch.flatten(z, 1)
        r = torch.cat([x, y, z], dim=1)
        return r

    def forward(self, x, y, z) -> torch.Tensor:
        r = self.convolutions(x, y, z)
        r = self.classifier(r)
        return r

class ConcatEffModels(nn.Module):

    def __init__(self, params: dict) -> None:
        super(ConcatEffModels, self).__init__()
        self.bn = params['bn']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.base_models = [get_model_(run_id)[1] for run_id in params['runs_id']]
        for base_model in self.base_models:
            base_model.classifier = nn.Identity
            for index, param in enumerate(base_model.parameters()):
                param.require_grad = False


        clf_block = self.make_clf_block(params)
        self.classifier = nn.Sequential(*clf_block)

    def make_clf_block(self, params: dict):
        clf_neurons = params['clf_neurons']
        n_layers = len(clf_neurons)
        # obtenemos la forma de la capa aplanada para obtener el nº de inputs 
        # para la primera capa del clasificador
        n_models = len(self.base_models)
        x = [torch.rand(((1, 3) + (224, 224))) for i in range(n_models)]
        x = [i.to(self.device) for i in x]

        pre_cls_shape = self.convolutions(*x).data.shape[1]
        layers_in = (pre_cls_shape, *clf_neurons[:-1])
        layers_out = clf_neurons
        # creamos una tupla (len = número de capas) con el dropout tq el todos 
        # son iguales (dado por params) menos el último que lo hacemos 0
        # dropout_ratios = (0, ) + (params['dropout'], ) * (n_layers - 2) + (0, )  
        dropout_ratios = (params['dropout'], ) * (n_layers - 1) + (0, )  
        no_linear_fun = params["clf_no_linear_fun"]
        no_linear_funs = (no_linear_fun, ) * (n_layers - 1) + ("log_softmax", )
        clf_params = (layers_in, layers_out, dropout_ratios, no_linear_funs)
        clf_block = []
        
        for layer_in, layer_out, d_r, nl_fun in zip(*clf_params):
            block = self.get_layer_clf(layer_in, layer_out, d_r, nl_fun)
            clf_block.extend(block)
        return tuple(clf_block)

    def get_layer_clf(self, layers_in, layers_out, dropout_ratio,
                      no_linear_fun="relu"):
        layer = nn.Linear(layers_in, layers_out)
        dropout_ = nn.Dropout(dropout_ratio)
        # seleccion de no lineal
        if no_linear_fun == "relu":
            no_linear = nn.ReLU(inplace=True)
        elif no_linear_fun == "sigmoid":
            no_linear = nn.Sigmoid()
        elif no_linear_fun == "log_softmax":
            no_linear = nn.LogSoftmax(1)
        elif no_linear_fun == "none":
            no_linear = nn.Identity()
        else:
            raise Exception("Not valid no_linear_fun")
        if self.bn and no_linear_fun != "log_softmax" :
            bn = nn.BatchNorm1d(layers_out)
            fc = dropout_, layer, no_linear, bn
        else:
            fc = dropout_, layer, no_linear
        return fc        

    def convolutions(self, *args):
        if not len(args) == len(self.base_models):
            raise Exception('El nº de modelos tiene que ser igual al nº de img') 
        
        all_features = [model(x) for x, model in zip(args, self.base_models)]
        all_features = torch.cat(all_features, dim=1)
        return all_features

    def forward(self, *args) -> torch.Tensor:
        r = self.convolutions(*args)
        r = self.classifier(r)
        return r

class ConvNet3D(nn.Module):

    def __init__(self, params: dict) -> None:
        super(ConvNet3D, self).__init__()
        self.bn = params['bn']
        conv_block = self.make_conv_block(params)
        self.features = nn.Sequential(*conv_block[:-2])
        self.end_conv = nn.Sequential(*conv_block[-2:])
        if params['avg_layer'] == "identity":
            self.avgpool = nn.Identity()
        elif params['avg_layer'] == "adp":
            self.avgpool = nn.AdaptiveAvgPool3d(params['avg_layer_size'])
        else:
            raise Exception("not valid avg_layer")

        clf_block = self.make_clf_block(params)
        self.classifier = nn.Sequential(*clf_block)

    def make_conv_block(self, params: dict):
        conv_sizes = params['conv_sizes']
        conv_strides = params['conv_strides']
        pool_sizes = params['pool_sizes']
        pool_strides = params['pool_strides']
        channels_in = (params['n_channels'], *params['conv_filters'][:-1])
        channels_out = params['conv_filters']
        no_linear_fun = params["clf_no_linear_fun"]
        n_convs = len(conv_sizes)
        dropout_ratios = (params['dropout'], ) * (n_convs - 1) + (0, ) 
        conv_params = (conv_sizes, conv_strides, pool_sizes, pool_strides,
                       channels_in, channels_out, dropout_ratios)
        conv_block = []
        for c_size, c_stride, p_size, p_stride, chan_in, chan_out, d_r in zip(*conv_params):
            block = self.get_block(chan_in, chan_out, c_size, c_stride, p_size, 
                           p_stride, no_linear_fun, d_r)
            conv_block.extend(block)
        return tuple(conv_block)

    def get_block(self, channels_in, channels_out, conv_size, conv_stride, 
                  pool_size, pool_sride, no_linear_fun, dropout_ratio):
        conv = nn.Conv3d(channels_in, channels_out, conv_size, conv_stride)
        pool = nn.MaxPool3d(pool_size, pool_sride)
        dropout_ = nn.Dropout(dropout_ratio)
        if no_linear_fun == "relu":
            no_linear = nn.ReLU(inplace=True)
        elif no_linear_fun == "none":
            no_linear = nn.Identity()
        else:
            raise Exception("Not valid no_linear_fun")
        if self.bn:
            bn = nn.BatchNorm3d(channels_out)
            block = conv, no_linear, bn, pool, dropout_
        else:
            block = conv, no_linear, pool, dropout_
        return block

    def make_clf_block(self, params: dict):
        clf_neurons = params['clf_neurons']
        input_shape = (1, params['n_channels']) + params["input_shape"]
        n_layers = len(clf_neurons)
        # obtenemos la forma de la capa aplanada para obtener el nº de inputs 
        # para la primera capa del clasificador
        pre_cls_shape = self.convolutions(
            torch.rand(input_shape)
        ).data.shape[1]
        layers_in = (pre_cls_shape, *clf_neurons[:-1])
        layers_out = clf_neurons
        # creamos una tupla (len = número de capas) con el dropout tq el todos 
        # son iguales (dado por params) menos el último que lo hacemos 0
        # dropout_ratios = (0, ) + (params['dropout'], ) * (n_layers - 2) + (0, )  
        dropout_ratios = (params['dropout'], ) * (n_layers - 1) + (0, )  
        no_linear_fun = params["clf_no_linear_fun"]
        no_linear_funs = (no_linear_fun, ) * (n_layers - 1) + ("log_softmax", )
        clf_params = (layers_in, layers_out, dropout_ratios, no_linear_funs)
        clf_block = []
        
        for layer_in, layer_out, d_r, nl_fun in zip(*clf_params):
            block = self.get_layer_clf(layer_in, layer_out, d_r, nl_fun)
            clf_block.extend(block)
        return tuple(clf_block)

    def get_layer_clf(self, layers_in, layers_out, dropout_ratio,
                      no_linear_fun="relu"):
        layer = nn.Linear(layers_in, layers_out)
        dropout_ = nn.Dropout(dropout_ratio)
        # seleccion de no lineal
        if no_linear_fun == "relu":
            no_linear = nn.ReLU(inplace=True)
        elif no_linear_fun == "sigmoid":
            no_linear = nn.Sigmoid()
        elif no_linear_fun == "log_softmax":
            no_linear = nn.LogSoftmax(1)
        elif no_linear_fun == "none":
            no_linear = nn.Identity()
        else:
            raise Exception("Not valid no_linear_fun")
        if self.bn and no_linear_fun != "log_softmax" :
            bn = nn.BatchNorm1d(layers_out)
            fc = dropout_, layer, no_linear, bn
        else:
            fc = dropout_, layer, no_linear
        return fc        

    def convolutions(self, x):
        x = self.features(x)
        x = self.end_conv(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convolutions(x)
        x = self.classifier(x)
        return x

class GoogleNet(nn.Module):
    def __init__(self, params: dict) -> None:
        super(GoogleNet, self).__init__()
        self.bn = params['bn']
        self.base_model = models.googlenet(True)
        n_layers = 173
        for index, param in enumerate(self.base_model.parameters()):
            if index <= n_layers - 1 - params["unfreeze_layers"]:
                param.requires_grad = False
            else:
                param.requires_grad = True
        # aquí puedes cambiar el avgpool
        clf_block = self.make_clf_block(params)
        self.base_model.fc = nn.Sequential(*clf_block)

    def make_clf_block(self, params: dict):
        clf_neurons = params['clf_neurons']
        input_shape = (1, 1) + params["input_shape"]
        n_layers = len(clf_neurons)
        # obtenemos la forma de la capa aplanada para obtener el nº de inputs 
        # para la primera capa del clasificador
        pre_cls_shape = 1024
        layers_in = (pre_cls_shape, *clf_neurons[:-1])
        layers_out = clf_neurons
        # creamos una tupla (len = número de capas) con el dropout tq el todos 
        # son iguales (dado por params) menos el último que lo hacemos 0
        # dropout_ratios = (0, ) + (params['dropout'], ) * (n_layers - 2) + (0, )  
        dropout_ratios = (params['dropout'], ) * (n_layers - 1) + (0, )  
        no_linear_fun = params["clf_no_linear_fun"]
        no_linear_funs = (no_linear_fun, ) * (n_layers - 1) + ("log_softmax", )
        clf_params = (layers_in, layers_out, dropout_ratios, no_linear_funs)
        clf_block = []
        
        for layer_in, layer_out, d_r, nl_fun in zip(*clf_params):
            block = self.get_layer_clf(layer_in, layer_out, d_r, nl_fun)
            clf_block.extend(block)
        return tuple(clf_block)

    def get_layer_clf(self, layers_in, layers_out, dropout_ratio,
                      no_linear_fun="relu"):
        layer = nn.Linear(layers_in, layers_out)
        dropout_ = nn.Dropout(dropout_ratio)
        # seleccion de no lineal
        if no_linear_fun == "relu":
            no_linear = nn.ReLU(inplace=True)
        elif no_linear_fun == "sigmoid":
            no_linear = nn.Sigmoid()
        elif no_linear_fun == "log_softmax":
            no_linear = nn.LogSoftmax(1)
        elif no_linear_fun == "none":
            no_linear = nn.Identity()
        else:
            raise Exception("Not valid no_linear_fun")
        if self.bn and no_linear_fun != "log_softmax" :
            bn = nn.BatchNorm1d(layers_out)
            fc = dropout_, layer, no_linear, bn
        else:
            fc = dropout_, layer, no_linear
        return fc        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.base_model(x)
        return x

class EffNetB0(nn.Module):
    def __init__(self, params: dict) -> None:
        super(EffNetB0, self).__init__()
        self.bn = params['bn']
        self.base_model = models.efficientnet_b0(True)
        n_layers = 213
        for index, param in enumerate(self.base_model.parameters()):
            if index <= n_layers - 1 - params["unfreeze_layers"]:
                param.requires_grad = False
            else:
                param.requires_grad = True
        # aquí puedes cambiar el avgpool
        clf_block = self.make_clf_block(params)
        self.base_model.classifier = nn.Sequential(*clf_block)

    def make_clf_block(self, params: dict):
        clf_neurons = params['clf_neurons']
        input_shape = (1, 1) + params["input_shape"]
        n_layers = len(clf_neurons)
        # obtenemos la forma de la capa aplanada para obtener el nº de inputs 
        # para la primera capa del clasificador
        pre_cls_shape = 1280
        layers_in = (pre_cls_shape, *clf_neurons[:-1])
        layers_out = clf_neurons
        # creamos una tupla (len = número de capas) con el dropout tq el todos 
        # son iguales (dado por params) menos el último que lo hacemos 0
        # dropout_ratios = (0, ) + (params['dropout'], ) * (n_layers - 2) + (0, )  
        dropout_ratios = (params['dropout'], ) * (n_layers - 1) + (0, )  
        no_linear_fun = params["clf_no_linear_fun"]
        no_linear_funs = (no_linear_fun, ) * (n_layers - 1) + ("log_softmax", )
        clf_params = (layers_in, layers_out, dropout_ratios, no_linear_funs)
        clf_block = []
        
        for layer_in, layer_out, d_r, nl_fun in zip(*clf_params):
            block = self.get_layer_clf(layer_in, layer_out, d_r, nl_fun)
            clf_block.extend(block)
        return tuple(clf_block)

    def get_layer_clf(self, layers_in, layers_out, dropout_ratio,
                      no_linear_fun="relu"):
        layer = nn.Linear(layers_in, layers_out)
        dropout_ = nn.Dropout(dropout_ratio)
        # seleccion de no lineal
        if no_linear_fun == "relu":
            no_linear = nn.ReLU(inplace=True)
        elif no_linear_fun == "sigmoid":
            no_linear = nn.Sigmoid()
        elif no_linear_fun == "log_softmax":
            no_linear = nn.LogSoftmax(1)
        elif no_linear_fun == "none":
            no_linear = nn.Identity()
        else:
            raise Exception("Not valid no_linear_fun")
        if self.bn and no_linear_fun != "log_softmax" :
            bn = nn.BatchNorm1d(layers_out)
            fc = dropout_, layer, no_linear, bn
        else:
            fc = dropout_, layer, no_linear
        return fc        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.base_model(x)
        return x

class EffNetB2(nn.Module):
    def __init__(self, params: dict) -> None:
        super(EffNetB2, self).__init__()
        self.bn = params['bn']
        self.base_model = models.efficientnet_b2(True)
        n_layers = 301
        for index, param in enumerate(self.base_model.parameters()):
            if index <= n_layers - 1 - params["unfreeze_layers"]:
                param.requires_grad = False
            else:
                param.requires_grad = True
        # aquí puedes cambiar el avgpool
        clf_block = self.make_clf_block(params)
        self.base_model.classifier = nn.Sequential(*clf_block)

    def make_clf_block(self, params: dict):
        clf_neurons = params['clf_neurons']
        input_shape = (1, 1) + params["input_shape"]
        n_layers = len(clf_neurons)
        # obtenemos la forma de la capa aplanada para obtener el nº de inputs 
        # para la primera capa del clasificador
        pre_cls_shape = 1408
        layers_in = (pre_cls_shape, *clf_neurons[:-1])
        layers_out = clf_neurons
        # creamos una tupla (len = número de capas) con el dropout tq el todos 
        # son iguales (dado por params) menos el último que lo hacemos 0
        # dropout_ratios = (0, ) + (params['dropout'], ) * (n_layers - 2) + (0, )  
        dropout_ratios = (params['dropout'], ) * (n_layers - 1) + (0, )  
        no_linear_fun = params["clf_no_linear_fun"]
        no_linear_funs = (no_linear_fun, ) * (n_layers - 1) + ("log_softmax", )
        clf_params = (layers_in, layers_out, dropout_ratios, no_linear_funs)
        clf_block = []
        
        for layer_in, layer_out, d_r, nl_fun in zip(*clf_params):
            block = self.get_layer_clf(layer_in, layer_out, d_r, nl_fun)
            clf_block.extend(block)
        return tuple(clf_block)

    def get_layer_clf(self, layers_in, layers_out, dropout_ratio,
                      no_linear_fun="relu"):
        layer = nn.Linear(layers_in, layers_out)
        dropout_ = nn.Dropout(dropout_ratio)
        # seleccion de no lineal
        if no_linear_fun == "relu":
            no_linear = nn.ReLU(inplace=True)
        elif no_linear_fun == "sigmoid":
            no_linear = nn.Sigmoid()
        elif no_linear_fun == "log_softmax":
            no_linear = nn.LogSoftmax(1)
        elif no_linear_fun == "none":
            no_linear = nn.Identity()
        else:
            raise Exception("Not valid no_linear_fun")
        if self.bn and no_linear_fun != "log_softmax" :
            bn = nn.BatchNorm1d(layers_out)
            fc = dropout_, layer, no_linear, bn
        else:
            fc = dropout_, layer, no_linear
        return fc        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.base_model(x)
        return x

class EffNetB5(nn.Module):
    def __init__(self, params: dict) -> None:
        super(EffNetB5, self).__init__()
        self.bn = params['bn']
        self.base_model = models.efficientnet_b5(True)
        n_layers = 506
        for index, param in enumerate(self.base_model.parameters()):
            if index <= n_layers - 1 - params["unfreeze_layers"]:
                param.requires_grad = False
            else:
                param.requires_grad = True
        # aquí puedes cambiar el avgpool
        clf_block = self.make_clf_block(params)
        self.base_model.classifier = nn.Sequential(*clf_block)

    def make_clf_block(self, params: dict):
        clf_neurons = params['clf_neurons']
        input_shape = (1, 1) + params["input_shape"]
        n_layers = len(clf_neurons)
        # obtenemos la forma de la capa aplanada para obtener el nº de inputs 
        # para la primera capa del clasificador
        pre_cls_shape = 2048
        layers_in = (pre_cls_shape, *clf_neurons[:-1])
        layers_out = clf_neurons
        # creamos una tupla (len = número de capas) con el dropout tq el todos 
        # son iguales (dado por params) menos el último que lo hacemos 0
        # dropout_ratios = (0, ) + (params['dropout'], ) * (n_layers - 2) + (0, )  
        dropout_ratios = (params['dropout'], ) * (n_layers - 1) + (0, )  
        no_linear_fun = params["clf_no_linear_fun"]
        no_linear_funs = (no_linear_fun, ) * (n_layers - 1) + ("log_softmax", )
        clf_params = (layers_in, layers_out, dropout_ratios, no_linear_funs)
        clf_block = []
        
        for layer_in, layer_out, d_r, nl_fun in zip(*clf_params):
            block = self.get_layer_clf(layer_in, layer_out, d_r, nl_fun)
            clf_block.extend(block)
        return tuple(clf_block)

    def get_layer_clf(self, layers_in, layers_out, dropout_ratio,
                      no_linear_fun="relu"):
        layer = nn.Linear(layers_in, layers_out)
        dropout_ = nn.Dropout(dropout_ratio)
        # seleccion de no lineal
        if no_linear_fun == "relu":
            no_linear = nn.ReLU(inplace=True)
        elif no_linear_fun == "sigmoid":
            no_linear = nn.Sigmoid()
        elif no_linear_fun == "log_softmax":
            no_linear = nn.LogSoftmax(1)
        elif no_linear_fun == "none":
            no_linear = nn.Identity()
        else:
            raise Exception("Not valid no_linear_fun")
        if self.bn and no_linear_fun != "log_softmax" :
            bn = nn.BatchNorm1d(layers_out)
            fc = dropout_, layer, no_linear, bn
        else:
            fc = dropout_, layer, no_linear
        return fc        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.base_model(x)
        return x

class ViT(nn.Module):
    def __init__(self, params: dict) -> None:
        super(EffNetB5, self).__init__()
        self.bn = params['bn']
        self.base_model = models.vit_b_16(True)
        n_layers = 152
        for index, param in enumerate(self.base_model.parameters()):
            if index <= n_layers - 1 - params["unfreeze_layers"]:
                param.requires_grad = False
            else:
                param.requires_grad = True
        # aquí puedes cambiar el avgpool
        clf_block = self.make_clf_block(params)
        self.base_model.heads = nn.Sequential(*clf_block)

    def make_clf_block(self, params: dict):
        clf_neurons = params['clf_neurons']
        input_shape = (1, 1) + params["input_shape"]
        n_layers = len(clf_neurons)
        # obtenemos la forma de la capa aplanada para obtener el nº de inputs 
        # para la primera capa del clasificador
        pre_cls_shape = 768
        layers_in = (pre_cls_shape, *clf_neurons[:-1])
        layers_out = clf_neurons
        # creamos una tupla (len = número de capas) con el dropout tq el todos 
        # son iguales (dado por params) menos el último que lo hacemos 0
        # dropout_ratios = (0, ) + (params['dropout'], ) * (n_layers - 2) + (0, )  
        dropout_ratios = (params['dropout'], ) * (n_layers - 1) + (0, )  
        no_linear_fun = params["clf_no_linear_fun"]
        no_linear_funs = (no_linear_fun, ) * (n_layers - 1) + ("log_softmax", )
        clf_params = (layers_in, layers_out, dropout_ratios, no_linear_funs)
        clf_block = []
        
        for layer_in, layer_out, d_r, nl_fun in zip(*clf_params):
            block = self.get_layer_clf(layer_in, layer_out, d_r, nl_fun)
            clf_block.extend(block)
        return tuple(clf_block)

    def get_layer_clf(self, layers_in, layers_out, dropout_ratio,
                      no_linear_fun="relu"):
        layer = nn.Linear(layers_in, layers_out)
        dropout_ = nn.Dropout(dropout_ratio)
        # seleccion de no lineal
        if no_linear_fun == "relu":
            no_linear = nn.ReLU(inplace=True)
        elif no_linear_fun == "sigmoid":
            no_linear = nn.Sigmoid()
        elif no_linear_fun == "log_softmax":
            no_linear = nn.LogSoftmax(1)
        elif no_linear_fun == "none":
            no_linear = nn.Identity()
        else:
            raise Exception("Not valid no_linear_fun")
        if self.bn and no_linear_fun != "log_softmax" :
            bn = nn.BatchNorm1d(layers_out)
            fc = dropout_, layer, no_linear, bn
        else:
            fc = dropout_, layer, no_linear
        return fc        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.base_model(x)
        return x

class EffNetB7(nn.Module):
    def __init__(self, params: dict) -> None:
        super(EffNetB7, self).__init__()
        self.bn = params['bn']
        self.base_model = models.efficientnet_b7(True)
        n_layers = 711
        for index, param in enumerate(self.base_model.parameters()):
            if index <= n_layers - 1 - params["unfreeze_layers"]:
                param.requires_grad = False
            else:
                param.requires_grad = True
        # aquí puedes cambiar el avgpool
        clf_block = self.make_clf_block(params)
        self.base_model.classifier = nn.Sequential(*clf_block)

    def make_clf_block(self, params: dict):
        clf_neurons = params['clf_neurons']
        input_shape = (1, 1) + params["input_shape"]
        n_layers = len(clf_neurons)
        # obtenemos la forma de la capa aplanada para obtener el nº de inputs 
        # para la primera capa del clasificador
        pre_cls_shape = 2560
        layers_in = (pre_cls_shape, *clf_neurons[:-1])
        layers_out = clf_neurons
        # creamos una tupla (len = número de capas) con el dropout tq el todos 
        # son iguales (dado por params) menos el último que lo hacemos 0
        # dropout_ratios = (0, ) + (params['dropout'], ) * (n_layers - 2) + (0, )  
        dropout_ratios = (params['dropout'], ) * (n_layers - 1) + (0, )  
        no_linear_fun = params["clf_no_linear_fun"]
        no_linear_funs = (no_linear_fun, ) * (n_layers - 1) + ("log_softmax", )
        clf_params = (layers_in, layers_out, dropout_ratios, no_linear_funs)
        clf_block = []
        
        for layer_in, layer_out, d_r, nl_fun in zip(*clf_params):
            block = self.get_layer_clf(layer_in, layer_out, d_r, nl_fun)
            clf_block.extend(block)
        return tuple(clf_block)

    def get_layer_clf(self, layers_in, layers_out, dropout_ratio,
                      no_linear_fun="relu"):
        layer = nn.Linear(layers_in, layers_out)
        dropout_ = nn.Dropout(dropout_ratio)
        # seleccion de no lineal
        if no_linear_fun == "relu":
            no_linear = nn.ReLU(inplace=True)
        elif no_linear_fun == "sigmoid":
            no_linear = nn.Sigmoid()
        elif no_linear_fun == "log_softmax":
            no_linear = nn.LogSoftmax(1)
        elif no_linear_fun == "none":
            no_linear = nn.Identity()
        else:
            raise Exception("Not valid no_linear_fun")
        if self.bn and no_linear_fun != "log_softmax" :
            bn = nn.BatchNorm1d(layers_out)
            fc = dropout_, layer, no_linear, bn
        else:
            fc = dropout_, layer, no_linear
        return fc        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.base_model(x)
        return x

class ResNet50(nn.Module):
    def __init__(self, params: dict) -> None:
        super(ResNet50, self).__init__()
        self.bn = params['bn']
        self.base_model = models.resnet50(True)
        n_layers = 161
        for index, param in enumerate(self.base_model.parameters()):
            if index <= n_layers - 1 - params["unfreeze_layers"]:
                param.requires_grad = False
            else:
                param.requires_grad = True
        # aquí puedes cambiar el avgpool
        clf_block = self.make_clf_block(params)
        self.base_model.fc = nn.Sequential(*clf_block)

    def make_clf_block(self, params: dict):
        clf_neurons = params['clf_neurons']
        input_shape = (1, 1) + params["input_shape"]
        n_layers = len(clf_neurons)
        # obtenemos la forma de la capa aplanada para obtener el nº de inputs 
        # para la primera capa del clasificador
        pre_cls_shape = 2048
        layers_in = (pre_cls_shape, *clf_neurons[:-1])
        layers_out = clf_neurons
        # creamos una tupla (len = número de capas) con el dropout tq el todos 
        # son iguales (dado por params) menos el último que lo hacemos 0
        # dropout_ratios = (0, ) + (params['dropout'], ) * (n_layers - 2) + (0, )  
        dropout_ratios = (params['dropout'], ) * (n_layers - 1) + (0, )  
        no_linear_fun = params["clf_no_linear_fun"]
        no_linear_funs = (no_linear_fun, ) * (n_layers - 1) + ("log_softmax", )
        clf_params = (layers_in, layers_out, dropout_ratios, no_linear_funs)
        clf_block = []
        
        for layer_in, layer_out, d_r, nl_fun in zip(*clf_params):
            block = self.get_layer_clf(layer_in, layer_out, d_r, nl_fun)
            clf_block.extend(block)
        return tuple(clf_block)

    def get_layer_clf(self, layers_in, layers_out, dropout_ratio,
                      no_linear_fun="relu"):
        layer = nn.Linear(layers_in, layers_out)
        dropout_ = nn.Dropout(dropout_ratio)
        # seleccion de no lineal
        if no_linear_fun == "relu":
            no_linear = nn.ReLU(inplace=True)
        elif no_linear_fun == "sigmoid":
            no_linear = nn.Sigmoid()
        elif no_linear_fun == "log_softmax":
            no_linear = nn.LogSoftmax(1)
        elif no_linear_fun == "none":
            no_linear = nn.Identity()
        else:
            raise Exception("Not valid no_linear_fun")
        if self.bn and no_linear_fun != "log_softmax" :
            bn = nn.BatchNorm1d(layers_out)
            fc = dropout_, layer, no_linear, bn
        else:
            fc = dropout_, layer, no_linear
        return fc        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.base_model(x)
        return x
# https://discuss.pytorch.org/t/how-to-delete-layer-in-pretrained-model/17648

class ConvNetOld(nn.Module):
    def __init__(self, params: dict):
        # , input_size, kernel_size, n_layers, n_filter, 
        #          linear_out_features, dropout=0.2
        super().__init__()
        self.input_shape = (1, 1) + params["input_shape"]
        
        self.conv1 = nn.Conv2d(
            1,
            params["n_filters"][0],
            params["kernel_size"][0],
            params["kernel_stride"][0]
            )
        self.pool = nn.MaxPool2d(params["pool_size"], 2)
        self.conv2 = nn.Conv2d(
            params["n_filters"][0],
            params["n_filters"][1],
            params["kernel_size"][1],
            params["kernel_stride"][1]
            )
        self.fc2 = nn.Linear(32, 8)
        self.fc3 = nn.Linear(8, 2)
        self.dropout = nn.Dropout(params["dropout"])
        # obtenemos la forma de la capa aplanada para obtener el nº de inputs 
        # para la primera capa del clasificador
        pre_cls_shape = self.convolutions(
            torch.rand(self.input_shape)
        ).data.shape[1]
        self.fc1 = nn.Linear(pre_cls_shape, 32)    

    def convolutions(self, x):
        """
        Capas de convolición donde
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        return x

    def clasificator(self, x):
        """
        capas del clasificador
        """
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        x = F.log_softmax(self.fc3(x), dim=1)
        return x

    def forward(self, x):
        """
        red completa
        """
        x = self.convolutions(x)
        x = self.clasificator(x)
        return x

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

# %%
if __name__ == "__main__":
    from exp_p_y.params import params
    params['input_shape'] = ((62, 128), (128, 128), (62, 128))
    model = ConvNetConcat(params)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    summary(model, [(1, 62, 128), (1, 128, 128), (1, 62, 128)])
# %%
    
    from exp_p_resnet.params import params
    # model = ResNet(params)
    # model = models.vit_b_16(pretrained=True)
    params['unfreeze_layers'] = 10
    model = EffNetB2(params)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    summary(model, (3, 224,  224))
    
# %%
    # from exp_p_resnet.params import params
    # # model = ResNet(params)
    # # model = models.resnet50(pretrained=True)
    # params['unfreeze_layers'] = 7
    # model = ResNet50(params)
    # # model = EffNetB0(params)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)
    # summary(model, (3, 224,  224))
    # models.vit_b_16(True)
    model = models.vit_b_16(pretrained=True)
    for i, j in enumerate(model.parameters()):
        print(i)