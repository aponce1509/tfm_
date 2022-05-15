# %%
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import optuna
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import mlflow
from utils_win_cube import Cascadas
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
n_epochs = 5

# Fuentes:
# https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
# https://deeplizard.com/learn/video/XfYmia3q2Ow
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py
# https://medium.com/swlh/pytorch-mlflow-optuna-experiment-tracking-and-hyperparameter-optimization-132778d6defc

# Definimos la arquitecturra de la red

class ConvNet(nn.Module):
    def __init__(self):
        # , input_size, kernel_size, n_layers, n_filter, 
        #          linear_out_features, dropout=0.2
        super().__init__()
        # suponemos 62 x 128
        self.conv1 = nn.Conv2d(1, 16, 3)  # 60 x 126
        self.pool = nn.MaxPool2d(2, 2)  # 30 x 63
        self.conv2 = nn.Conv2d(16, 24, 3)  # 28 x 61 -> 14 x 30
        # Aquí entra el tamaño de la ventana
        self.fc1 = nn.Linear(24 * 14 * 30, 32)
        self.fc2 = nn.Linear(32, 8)
        self.fc3 = nn.Linear(8, 2)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train(model: nn.Module, optimizer: optim.Optimizer, device: str, 
          train_loader: DataLoader, writer: SummaryWriter,
          criterion, epoch: int):
    model.train()
    running_loss = 0.
    total_loss = 0.
    total_correct = 0
    n_batches = len(train_loader)
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device, dtype=torch.float), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        total_loss += loss.item()

        aux = int(n_batches / 10)
        predicted = torch.max(outputs.data, 1)[1]
        total_correct += sum(predicted == labels).item()

        if batch_idx % aux == aux - 1:
            last_loss = running_loss / aux  # loss per batch
            print(f'  batch {batch_idx + 1} loss: {last_loss:.3g}')
            tb_x = epoch * len(train_loader) + batch_idx + 1  # eje x
            writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.
    
    train_size = len(train_loader.dataset)
    train_acc = total_correct / train_size
    mean_loss_train = total_loss / (batch_idx + 1)
    return mean_loss_train, train_acc

def validation(model: nn.Module, device: str, 
               validation_loader: DataLoader, writer: SummaryWriter,
               criterion):
    model.eval()
    val_size = len(validation_loader.dataset)
    val_loss = 0.
    total_correct_val = 0
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(validation_loader):
            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss
            predicted = torch.max(outputs.data, 1)[1]
            total_correct_val += sum((predicted == labels)).item()

    mean_val_loss = val_loss / (batch_idx + 1)
    val_acc = total_correct_val / val_size
    return mean_val_loss, val_acc

def hyperparameter_suggest(trial: optuna.Trial):
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    optimizer_id = trial.suggest_categorical(
        "optimizer_id",
        ["Adam", "Adadelta"]
    )
    print(trial.params)  # PONER BONITO
    return lr, optimizer_id

def objective(trial: optuna.Trial):
    with mlflow.start_run():
        lr, optimizer_id = hyperparameter_suggest(trial)
        mlflow.log_param("Params", trial.params)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mlflow.log_param("Device", device)

        model = ConvNet()
        model = model.to(device)

        if optimizer_id == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=lr)
        if optimizer_id == "Adadelta":
            optimizer = optim.Adadelta(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
        # IDEA: a la hora de obtener la clase DataSet (cascada) 
        # comprueba que los parametros que la definen hayan cambiado
        # antes de calcularla
        # DEFINIR LOS DATASET Y LOADERS
        print("Cargando datos:")
        transform = transforms.ToTensor()
        train_data = Cascadas(win_shape=(62, 62, 128), transform=transform)
        val_data = Cascadas(win_shape=(62, 62, 128), train=False)
        train_loader = DataLoader(train_data, 32, shuffle=True)
        val_loader = DataLoader(val_data, 32, shuffle=True)
        EPOCHS = n_epochs
        mlflow.log_param("Épocas", EPOCHS)
        # PONER IDENTIFICADOR DE LOS PARÁMETROS
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        writer = SummaryWriter('runs/temp_{}'.format(timestamp))
        scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
        best_val_loss = 1_000_000
        for epoch in range(EPOCHS):
            print(f"EPOCH {epoch + 1}:")
            # Entrenamiento
            mean_train_loss, acc_train = train(
                model=model,
                optimizer=optimizer,
                device=device,
                train_loader=train_loader,
                criterion=criterion,
                writer=writer,
                epoch=epoch            
            )
            mean_val_loss, acc_val = validation(
                model=model,
                device=device,
                validation_loader=val_loader,
                writer=writer,
                criterion=criterion
            )
            mlflow.log_metric(
                "avg_train_losses", mean_train_loss, step=epoch
            )
            mlflow.log_metric("avg_val_losses", mean_val_loss, step=epoch)

            mlflow.log_metric("Train Acc", acc_train, step=epoch)
            mlflow.log_metric("Validation Acc", acc_val, step=epoch)
            print(f'Acc train: {mean_train_loss:.3g} ' +\
                'validation {mean_val_loss:.3g}')
            print(f'Acc train {acc_train:.3g} validation: {acc_val:.3g}')
            writer.add_scalars(
                'Training vs. Validation Loss',
                {'Training' : mean_train_loss, 'Validation' : mean_val_loss},
                epoch + 1
            )
            writer.add_scalars(
                'Training vs. Validation Accuracy',
                {'Training' : acc_train, 'Validation' : acc_val},
                epoch + 1
            )
            # Crear gráfico entre pruebas
            writer.flush()
            if mean_val_loss < best_val_loss:
                best_val_loss = mean_val_loss
                # CAMBIAR FORMATO
                model_path = f"models/model_{timestamp}_{epoch}"
                torch.save(model.state_dict(), model_path)
            scheduler.step()
    error = best_val_loss
    return error  # An objective value linked with the Trial object.

def tuning():
    study = optuna.create_study(direction="minimize")  # Create a new study.
    # Invoke optimization of the objective function.
    study.optimize(objective, n_trials=5)
    # Print optuna study statistics
    print("\n++++++++++++++++++++++++++++++++++\n")
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Trial number: ", trial.number)
    print("  Loss (trial value): ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

def basic_run():
    print("Cargando datos:")
    transform = transforms.ToTensor()
    train_data = Cascadas(win_shape=(62, 62, 128), transform=transform)
    val_data = Cascadas(
        win_shape=(62, 62, 128), train=False, transform=transform
    )
    train_loader = DataLoader(train_data, 32, shuffle=True)
    val_loader = DataLoader(val_data, 32, shuffle=True)

    print(f'Training set has {len(train_data)} instances')
    print(f'Validation set has {len(val_data)} instances')
    # Definición elementos de la red neuronal
    model = ConvNet()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    model = model.to(device)
    print("Device:", device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/temp_{}'.format(timestamp))
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
    print("training:")
    print(f"timestamp: {timestamp}")
    best_val_loss = 1_000_000
    for epoch in range(n_epochs):
        print(f"EPOCH {epoch + 1}:")
        mean_train_loss, acc_train = train(
            model=model,
            optimizer=optimizer,
            device=device,
            train_loader=train_loader,
            criterion=criterion,
            writer=writer,
            epoch=epoch            
        )
        mean_val_loss, acc_val = validation(
            model=model,
            device=device,
            validation_loader=val_loader,
            writer=writer,
            criterion=criterion
        )
        print(f'Loss train: {mean_train_loss:.3g} validation {mean_val_loss:.3g}')
        print(f'Acc train {acc_train:.3g} validation: {acc_val:.3g}')
        writer.add_scalars(
            'Training vs. Validation Loss',
            {'Training' : mean_train_loss, 'Validation' : mean_val_loss},
            epoch + 1
        )
        writer.add_scalars(
            'Training vs. Validation Accuracy',
            {'Training' : acc_train, 'Validation' : acc_val},
            epoch + 1
        )
        writer.flush()
        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            # CAMBIAR FORMATO
            model_path = f'models/model_test_1_{timestamp}_{epoch}'
            torch.save(model.state_dict(), model_path)
        scheduler.step()
    error = best_val_loss
    return error
# %%
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(device)
    basic_run()
    # transform = transforms.ToTensor()
    # train_data = Cascadas(win_shape=(62, 62, 128), transform=transform)
    # val_data = Cascadas(win_shape=(62, 62, 128), train=False)
    # train_loader = DataLoader(train_data, 32, shuffle=True)
    # val_loader = DataLoader(val_data, 32, shuffle=True)
    # print(f'Training set has {len(train_data)} instances')
    # print(f'Validation set has {len(val_data)} instances')
