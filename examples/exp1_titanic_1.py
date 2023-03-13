"""PyTorch MLP - Titanic

    This file contains an implementation of an MLP in PyTorch to do a classification on Titanic dataset.
"""

# region Imported Dependencies------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm
# endregion Imported Dependencies

# region Load Data------------------------------------------------------------------------------------------------------
data = pd.read_csv('../data/titanic/train.csv')
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
for iteration in tqdm(range(epochs)):
    model.train()
    optimizer.zero_grad()
    train_predictions = model(train_inputs)
    loss = criterion(train_predictions, train_targets)
    loss.backward()
    optimizer.step()

    if iteration % 1 == 0:
        print(f"Epoch {iteration}, Training Loss: {loss.item():.4f}")

# Evaluate
model.eval()
with torch.no_grad():
    validation_predictions = model(validation_inputs)
    _, validation_predictions = torch.max(validation_predictions, dim=1)
    accuracy = torch.sum(validation_predictions == validation_targets).item() / len(validation_targets)
    print(f"Validation Accuracy: {accuracy:.4f}")
# endregion Training
