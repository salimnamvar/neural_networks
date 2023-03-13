"""PyTorch MLP - Titanic

    This file contains an implementation of an MLP in PyTorch to do a classification on Titanic dataset.
"""

# region Imported Dependencies------------------------------------------------------------------------------------------
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm
# endregion Imported Dependencies


# region Load Data------------------------------------------------------------------------------------------------------
data = pd.read_csv(
    'E:\\LEARNING\\Priciple Neural_Networks Engineering\\Codes\\neural_networks\\data\\titanic\\train.csv')
# endregion Load Data

# region Preprocessing--------------------------------------------------------------------------------------------------
# Data Cleaning
label_encoder = LabelEncoder()
data['Sex'] = label_encoder.fit_transform(data['Sex'])
data['Embarked'] = label_encoder.fit_transform(data['Embarked'].astype(str))
data['Age'].fillna(data['Age'].median(), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
data['Survived'] = data['Survived'].astype(int)
data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
print(data.head())

# Normalization
scaler = StandardScaler()
data.iloc[:, :] = scaler.fit_transform(data.values)

# Data Sampling
train_inputs, validation_inputs, train_targets, validation_targets = train_test_split(data.drop('Survived', axis=1),
                                                                                      data['Survived'], test_size=0.2,
                                                                                      random_state=42)
train_inputs = torch.tensor(train_inputs.values, dtype=torch.float32)
train_targets = torch.tensor(train_targets.values, dtype=torch.long)
validation_inputs = torch.tensor(validation_inputs.values, dtype=torch.float32)
validation_targets = torch.tensor(validation_targets.values, dtype=torch.long)
# endregion Preprocessing

# region Figures--------------------------------------------------------------------------------------------------------
plt.ion()
fig_loss, (ax1_loss, ax2_loss) = plt.subplots(1, 2, figsize=(10, 5))
ax1_loss.set_title('Training LOSS')
ax2_loss.set_title('Evaluation LOSS')

fig_acc, (ax1_acc, ax2_acc) = plt.subplots(1, 2, figsize=(10, 5))
ax1_acc.set_title('Training Accuracy')
ax2_acc.set_title('Evaluation Accuracy')
# endregion Figures


# region Neural Network Hyper-Parameters--------------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(7, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Parameters
epochs = 100
model = MLP()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
# endregion Neural Network Hyper-Parameters

# region Training-------------------------------------------------------------------------------------------------------
train_losses = np.zeros((1, epochs))
validation_losses = np.zeros((1, epochs))
train_accuracies = np.zeros((1, epochs))
validation_accuracies = np.zeros((1, epochs))
for iteration in tqdm(range(epochs)):
    # Training----------------------------------------------------------------------------------------------------------
    model.train()
    optimizer.zero_grad()
    train_predictions = model(train_inputs)
    train_loss = criterion(train_predictions, train_targets)
    train_losses[0, iteration] = train_loss
    train_loss.backward()
    optimizer.step()
    _, train_predictions = torch.max(train_predictions, dim=1)
    train_accuracy = torch.sum(train_predictions == train_targets).item() / len(train_targets)
    train_accuracies[0, iteration] = train_accuracy
    #  print(f"Training Accuracy: {train_accuracy:.4f}")
    #  print(f"Epoch {iteration}, Training Loss: {train_loss.item():.4f}")

    # Evaluate----------------------------------------------------------------------------------------------------------
    model.eval()
    with torch.no_grad():
        validation_predictions = model(validation_inputs)
        validation_loss = criterion(validation_predictions, validation_targets)
        validation_losses[0, iteration] = validation_loss
        _, validation_predictions = torch.max(validation_predictions, dim=1)
        validation_accuracy = torch.sum(validation_predictions == validation_targets).item() / len(validation_targets)
        validation_accuracies[0, iteration] = validation_accuracy
        #  print(f"Validation Accuracy: {validation_accuracy:.4f}")

    # region Plots------------------------------------------------------------------------------------------------------
    ax1_loss.cla()
    ax1_loss.plot(range(epochs)[:iteration], train_losses[0, :iteration], marker='o', linestyle='-', markersize=5,
                  color='b', label='Training Loss[' + str(train_losses[0, iteration]) + ']')
    ax1_loss.legend()
    ax2_loss.cla()
    ax2_loss.plot(range(epochs)[:iteration], validation_losses[0, :iteration], marker='o', linestyle='-', markersize=5,
                  color='g', label='Evaluation Loss[' + str(validation_losses[0, iteration]) + ']')
    ax2_loss.legend()

    fig_loss.canvas.draw()
    fig_loss.canvas.flush_events()

    ax1_acc.cla()
    ax1_acc.plot(range(epochs)[:iteration], train_accuracies[0, :iteration], marker='o', linestyle='-', markersize=5,
                 color='b', label='Training Accuracy[' + str(train_accuracies[0, iteration]) + ']')
    ax1_acc.legend()

    ax2_acc.cla()
    ax2_acc.plot(range(epochs)[:iteration], validation_accuracies[0, :iteration], marker='o', linestyle='-',
                 markersize=5, color='b', label='Evaluation Accuracy[' + str(validation_accuracies[0, iteration]) + ']')
    ax2_acc.legend()

    fig_acc.canvas.draw()
    fig_acc.canvas.flush_events()

    plt.draw()
    plt.pause(0.05)
    # endregion Plots

plt.ioff()
plt.show()
# endregion Training
