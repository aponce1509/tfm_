# %%
import os

os.chdir("..")
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
from utis import Cascadas

EPOCH = 5

class ConvNet(nn.Module):
    def __init__(self, input_shape=(64, 64)):
        super().__init__()
        # suponemos 62 x 62
        self.conv1 = nn.Conv2d(1, 3, 3)  # 60 x 60
        self.pool = nn.MaxPool2d(2, 2)  # 30 x 30
        self.conv2 = nn.Conv2d(3, 16, 3)  # 28 x 28 -> 14 x 14
        # Aquí entra el tamaño de la ventana
        self.fc1 = nn.Linear(16 * 14 * 14, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train(model: nn.Module, optimizer: optim.Optimizer, device: str,
          train_loader: DataLoader, criterion):
    model.train()
    n_batches = len(train_loader)
    training_loss = 0.
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        training_loss += loss.item()
    mean_training_loss = training_loss / n_batches
    return mean_training_loss

def validation(model: nn.Module, validation_loader: DataLoader,
               device: str, criterion):
    model.eval()
    validation_size = len(validation_loader.dataset)
    validation_loss = 0.
    correct = 0
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(validation_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            validation_loss += loss
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            acc = 100 * correct / validation_size
            print(acc)
    validation_loss = validation_loss / validation_size
    return validation_loss, acc

def hyperparameter_suggest(trial: optuna.Trial):
    lr = trial.suggest_float("lr", 1e-4, 1e-1)
    optimizer_id = trial.suggest_categorical(
        "optimizer_id",
        ["Adam", "Adadelta"]
    )
    print(trial.params)
    return lr, optimizer_id

def objective(trial: optuna.Trial):
    with mlflow.start_run():
        lr, optimizer_id = hyperparameter_suggest(trial)
        mlflow.log_param(trial.params)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mlflow.log_param(device)

        model = ConvNet()
        model = model.to(device)

        if optimizer_id == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=lr)
        if optimizer_id == "Adadelta":
            optimizer = optim.Adadelta(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
        train_loader, validation_loader = 0, 0
        EPOCHS = 5
        mlflow.log_param(EPOCHS)

        for epoch in EPOCHS:
            mean_training_loss = train(
                model, optimizer, device, train_loader, criterion
            )
            validation_loss, acc = validation(model, validation_loader, device)
            mlflow.log_metric("avg_train_losses", mean_training_loss, step=epoch)
            mlflow.log_metric("Validation Acc", acc, step=epoch)
        scheduler.step()
    error = acc
    return error  # An objective value linked with the Trial object.

def tuning():
    study = optuna.create_study(direction="maximize")  # Create a new study.
    study.optimize(objective, n_trials=5)  # Invoke optimization of the objective function.
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
        print("    {}: {}".format(key, value))


def basic_run():
    batch_size = 32
    transform = transforms.ToTensor()
    train_data = Cascadas(transform=transform)
    train_loader = DataLoader(train_data, batch_size, True)
    model = ConvNet()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lr = 0.001
    EPOCHS = 3

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(EPOCHS):
        training_loss = 0.
        n_batches = len(train_loader)
        for batch_idx, (inputs, labels) in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            training_loss += loss.item()
            if batch_idx
            mean_training_loss = training_loss / n_batches
    print(mean_training_loss)

