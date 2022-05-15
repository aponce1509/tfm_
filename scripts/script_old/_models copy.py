# %%
import torch
from torchsummary import summary
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch import optim
torch.use_deterministic_algorithms(True)

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
    elif model_params["model_name"] == "convnet3d":
        model = ConvNet3D(model_params)
    elif model_params["model_name"] == "convnet_big":
        model = ConvNetBig(model_params)
    elif model_params["model_name"] == "resnet":
        model = Resnet(model_params)
    elif model_params["model_name"] == "color":
        model = Resnet(model_params)
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
    else:
        raise Exception("Optimizer not defined")
    # Scheduler
    if model_params["scheduler_name"] == "steplr":
        scheduler = StepLR(
            optimizer,
            step_size=model_params["scheduler_step_size"],
            gamma=model_params["scheduler_gamma"]
        )
    else:
        raise Exception("Scheduler not defined")
    # criterion
    criterion = get_criterion(model_params)
    return model, device, optimizer, scheduler, criterion

class ConvNet(nn.Module):

    def __init__(self, params: dict) -> None:
        super(ConvNet, self).__init__()
        
        conv_block = self.make_conv_block(params)
        self.features = nn.Sequential(*conv_block)
        self.avgpool = nn.Identity()

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
        print(dropout_ratios)
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
        block = conv, no_linear, pool, dropout_
        return block

    def make_clf_block(self, params: dict):
        clf_neurons = params['clf_neurons']
        input_shape = (1, 1) + params["input_shape"]
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
        dropout_ratios = (0, ) + (params['dropout'], ) * (n_layers - 2) + (0, )  
        print(dropout_ratios)
        # dropout_ratios = (params['dropout'], ) * (n_layers - 1) + (0, )  
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
        fc = dropout_, layer, no_linear
        return fc        

    def convolutions(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convolutions(x)
        x = self.classifier(x)
        return x

class ConvNet(nn.Module):
    def __init__(self, params: dict):
        # , input_size, kernel_size, n_layers, conv_filter, 
        #          linear_out_features, dropout=0.2
        super().__init__()
        self.input_shape = (1, 1) + params["input_shape"]
        
        self.conv1 = nn.Conv2d(
            1,
            params["conv_filters"][0],
            params["conv_sizes"][0],
            params["conv_strides"][0]
            )
        self.pool = nn.MaxPool2d(params["pool_sizes"][0], 2)
        self.conv2 = nn.Conv2d(
            params["conv_filters"][0],
            params["conv_filters"][1],
            params["conv_sizes"][1],
            params["conv_strides"][1]
            )

        # obtenemos la forma de la capa aplanada para obtener el nº de inputs 
        # para la primera capa del clasificador
        self.dropout = nn.Dropout(params["dropout"])
        pre_cls_shape = self.convolutions(
            torch.rand(self.input_shape)
        ).data.shape[1]
        self.fc1 = nn.Linear(pre_cls_shape, 32)    
        self.fc2 = nn.Linear(32, 8)
        self.fc3 = nn.Linear(8, 2)

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
# https://discuss.pytorch.org/t/how-to-delete-layer-in-pretrained-model/17648

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class Resnet(nn.Module):
    def __init__(self, params: dict):
        # , input_size, kernel_size, n_layers, n_filter, 
        #          linear_out_features, dropout=0.2
        super().__init__()
        self.base_model = models.resnet152(pretrained=True)
        num_ftrs = self.base_model.fc.in_features
        self.base_model.fc = Identity()
        # n_layers = len(self.base_model.parameters())
        for index, param in enumerate(self.base_model.parameters()):
            # if index <= n_layers - 1 - params["unfreeze_layers"]:
            param.requires_grad = False
        self.fc1 = nn.Linear(num_ftrs, 32)    
        self.fc2 = nn.Linear(32, 8)
        self.fc3 = nn.Linear(8, 2)
        self.dropout = nn.Dropout(params["dropout"])

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
        x = self.base_model(x)
        x = self.clasificator(x)
        return x

class ConvNetBig(nn.Module):
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
        x = F.log_softmax(self.fc3(x), dim=1)
        return x

    def forward(self, x):
        """
        red completa
        """
        x = self.convolutions(x)
        x = self.clasificator(x)
        return x

class ConvNetColor(nn.Module):
    def __init__(self, params: dict):
        # , input_size, kernel_size, n_layers, n_filter, 
        #          linear_out_features, dropout=0.2
        super().__init__()
        self.input_shape = (1, 3) + params["input_shape"]
        
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

class ConvNet3D(nn.Module):
    def __init__(self, params: dict):
        # , input_size, kernel_size, n_layers, n_filter, 
        #          linear_out_features, dropout=0.2
        super().__init__()
        self.input_shape = (1, 1) + params["input_shape"]

        self.conv1 = nn.Conv3d(
            1,
            params["n_filters"][0],
            params["kernel_size"][0]
        )
        self.pool = nn.MaxPool3d(params["pool_size"], 2)
        self.conv2 = nn.Conv3d(
            params["n_filters"][0],
            params["n_filters"][1],
            params["kernel_size"][1]
        )
        # Aquí entra el tamaño de la ventana
        self.dropout = nn.Dropout(params["dropout"])
        pre_cls_shape = self.convolutions(
            torch.rand(self.input_shape)
        ).data.shape[1]
        self.fc1 = nn.Linear(pre_cls_shape, 32)
        self.fc2 = nn.Linear(32, 8)
        self.fc3 = nn.Linear(8, 2)

    def convolutions(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        return x
        
    def clasficator(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)
        return x


    def forward(self, x):
        x = self.convolutions(x)
        x = self.clasficator(x)
        return x
# %%
if __name__ == "__main__":
    # from utils_win_cube_copy import CascadasFast
    # cascada = CascadasFast(cube_shape_x=1000, projection="y")
    # cascada.plot_simple(8)
    from exp_p_y.params import params
    model = ConvNet(params)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # a = summary(model, (1, 62, 62, 128))
    summary(model, (1, 62, 128))
