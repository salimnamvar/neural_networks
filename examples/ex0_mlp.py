"""Multi-Layer Perceptron Neural Networks [MLP]

    This file contains an implementation of an MLP from scratch with 3 active layers that utilizes the Backpropagation,
    Gradient Descent, and Delta Rule(LMS) to update the active weights in the hidden layers. The problem is a regression
    case on :ref:`Mackey-Glass Time Series <https://ieee-dataport.org/keywords/mackey-glass-time-series>` dataset.
"""

# region Imported Dependencies------------------------------------------------------------------------------------------
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
# endregion Imported Dependencies

# region Load Data------------------------------------------------------------------------------------------------------
data = np.loadtxt('../data/mgdata.dat')
data = data[:, 1]
# endregion Load Data

# region Preprocessing--------------------------------------------------------------------------------------------------
# Normalization
normalized_data = (data-np.min(data))/(np.max(data)-np.min(data))

# Feature Extraction
f0 = np.array([normalized_data]).transpose()
f1 = np.vstack((f0[1:], [0]))
f2 = np.vstack((f1[1:], [0]))
f3 = np.vstack((f2[1:], [0]))
f4 = np.vstack((f3[1:], [0]))
extracted_features_data = np.concatenate((f0, f1, f2, f3, f4), axis=1)

# Data Sampling
train_size = 0.75
num_train_data = int(np.round(len(data) * train_size))
num_eval_data = int(len(data) - num_train_data) - 4
train_inputs = extracted_features_data[:num_train_data, :-1]
train_targets = extracted_features_data[:num_train_data, -1]
eval_inputs = extracted_features_data[num_train_data:-4, :-1]
eval_targets = extracted_features_data[num_train_data:-4, -1]
# endregion Preprocessing

# region Neural Network Hyper-Parameters--------------------------------------------------------------------------------
# The number of neurons in layers
n0 = 4
n1 = 5
n2 = 3
n3 = 1

# Training Settings
eta = 0.4
epochs = 100
initial_weights_range = (-1, 1)

# Network Initialization
mse_train = np.zeros((1, epochs))
mse_eval = np.zeros((1, epochs))

w0 = np.random.uniform(low=initial_weights_range[0], high=initial_weights_range[1], size=(n1, n0))
net0 = np.zeros((n1, 1))
o0 = np.zeros((n1, 1))

w1 = np.random.uniform(low=initial_weights_range[0], high=initial_weights_range[1], size=(n2, n1))
net1 = np.zeros((n2, 1))
o1 = np.zeros((n2, 1))

w2 = np.random.uniform(low=initial_weights_range[0], high=initial_weights_range[1], size=(n3, n2))
net2 = np.zeros((n3, 1))
o2 = np.zeros((n3, 1))
# endregion Neural Network Hyper-Parameters


# region Sub-Functions--------------------------------------------------------------------------------------------------
# Activation Function
def sigmoid(x):
    return 1/(1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)
# endregion Sub-Functions


# region Figures--------------------------------------------------------------------------------------------------------
plt.ion()
fig_mse, (ax1_mse, ax2_mse) = plt.subplots(1, 2, figsize=(10, 5))
ax1_mse.set_title('Training MSE')
ax2_mse.set_title('Evaluation MSE')

fig_reg, (ax1_reg, ax2_reg) = plt.subplots(1, 2, figsize=(10, 5))
ax1_reg.set_title('Training Data Learning')
ax2_reg.set_title('Evaluation Data Learning')
# endregion Figures

# region Training-Evaluation--------------------------------------------------------------------------------------------
for iteration in tqdm(range(epochs)):
    # region Training---------------------------------------------------------------------------------------------------
    errors = np.zeros((1, num_train_data))
    for i, (x, d) in enumerate(zip(train_inputs, train_targets)):
        # Feedforward---------------------------------------------------------------------------------------------------
        x = np.expand_dims(x, axis=0)
        net0 = w0 @ x.transpose()
        o0 = sigmoid(net0)
        net1 = w1 @ o0
        o1 = sigmoid(net1)
        net2 = w2 @ o1
        o2 = net2

        # Backpropagation-----------------------------------------------------------------------------------------------
        errors[0, i] = d - o2
        w0 = w0 - eta * errors[0, i] * -1 * (w2 @ np.diag(sigmoid_derivative(o1.squeeze())) @ w1 @ np.diag(sigmoid_derivative(o0.squeeze()))).transpose() @ x
        w1 = w1 - eta * errors[0, i] * -1 * (w2 @ np.diag(sigmoid_derivative(o1.squeeze()))).transpose() @ o0.transpose()
        w2 = w2 - eta * errors[0, i] * -1 * 1 * o1.transpose()
    # endregion Training

    # region Evaluation-------------------------------------------------------------------------------------------------
    # Train Error
    train_errors = np.zeros((1, num_train_data))
    train_outputs = np.zeros((1, num_train_data))
    for i, (x, d) in enumerate(zip(train_inputs, train_targets)):
        # Feedforward---------------------------------------------------------------------------------------------------
        x = np.expand_dims(x, axis=0)
        net0 = w0 @ x.transpose()
        o0 = sigmoid(net0)
        net1 = w1 @ o0
        o1 = sigmoid(net1)
        net2 = w2 @ o1
        o2 = net2
        train_outputs[0, i] = o2
        train_errors[0, i] = d - o2

    # Total train error
    mse_train[0, iteration] = np.square(np.subtract(train_targets,train_outputs)).mean()

    # Evaluation Error
    eval_errors = np.zeros((1, num_eval_data))
    eval_outputs = np.zeros((1, num_eval_data))
    for i, (x, d) in enumerate(zip(eval_inputs, eval_targets)):
        # Feedforward---------------------------------------------------------------------------------------------------
        x = np.expand_dims(x, axis=0)
        net0 = w0 @ x.transpose()
        o0 = sigmoid(net0)
        net1 = w1 @ o0
        o1 = sigmoid(net1)
        net2 = w2 @ o1
        o2 = net2
        eval_outputs[0, i] = o2
        eval_errors[0, i] = d - o2

    # Total train error - MSE
    mse_eval[0, iteration] = np.square(np.subtract(eval_targets, eval_outputs)).mean()
    # endregion Evaluation

    # region Plots------------------------------------------------------------------------------------------------------
    # MSE
    ax1_mse.cla()
    ax1_mse.plot(range(epochs)[:iteration], mse_train[0, :iteration], marker='o', linestyle='-', markersize=5,
                 color='b', label='Training MSE[' + str(mse_train[0, iteration]) + ']')
    ax1_mse.legend()
    ax2_mse.cla()
    ax2_mse.plot(range(epochs)[:iteration], mse_eval[0, :iteration], marker='o', linestyle='-', markersize=5,
                 color='g', label='Evaluation MSE[' + str(mse_eval[0, iteration]) + ']')
    ax2_mse.legend()

    fig_mse.canvas.draw()
    fig_mse.canvas.flush_events()

    # Train Data
    ax1_reg.cla()
    ax1_reg.plot(range(num_train_data)[0:], train_targets, linestyle='-', color='g', label='Training Target')
    ax1_reg.plot(range(num_train_data)[0:], train_outputs[0, :], linestyle='-', color='r', label='Training Prediction')
    ax1_reg.legend()

    # Evaluation Data
    ax2_reg.cla()
    ax2_reg.plot(range(num_eval_data)[0:], eval_targets, linestyle='-', color='g', label='Evaluation Target')
    ax2_reg.plot(range(num_eval_data)[0:], eval_outputs[0, :], linestyle='-', color='r', label='Evaluation Prediction')
    ax2_reg.legend()

    fig_reg.canvas.draw()
    fig_reg.canvas.flush_events()

    plt.draw()
    plt.pause(0.05)
    # endregion Plots

plt.ioff()
plt.show()
# endregion Training-Evaluation
