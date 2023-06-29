import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.metrics import MeanSquaredError

# Load the input and output data from text files
alpha = np.loadtxt('alpha.txt')
delta_e = np.loadtxt('deltae.txt')
q = np.loadtxt('q.txt')
Cm = np.loadtxt('cm.txt')
Cl = np.loadtxt('cl.txt')
Cd = np.loadtxt('cd.txt')

# Combine the input and output data
X = np.column_stack((alpha, delta_e, q))
Y = np.column_stack((Cm, Cl, Cd))

# Split the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Normalize the input and output data
scaler_X = MinMaxScaler()
X_train_norm = scaler_X.fit_transform(X_train)
X_test_norm = scaler_X.transform(X_test)

scaler_Y = MinMaxScaler()
Y_train_norm = scaler_Y.fit_transform(Y_train)
Y_test_norm = scaler_Y.transform(Y_test)

# Define the model architecture
model = Sequential()
model.add(Dense(64, input_dim=3, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(3))

# Define the optimizer (RMSprop)
optimizer = RMSprop(learning_rate=0.001)

# Compile the model
model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=[MeanSquaredError()])

# Train the model
history = model.fit(X_train_norm, Y_train_norm, epochs=100, batch_size=32, verbose=1)

# Make predictions on test data
Y_pred_norm = model.predict(X_test_norm)

# Denormalize the predictions and test data
Y_pred = scaler_Y.inverse_transform(Y_pred_norm)
Y_test = scaler_Y.inverse_transform(Y_test_norm)

# Calculate residuals for each output variable
residuals = Y_test - Y_pred

# Calculate loss for each output variable
loss_per_variable = np.mean(np.square(residuals), axis=0)

# Create a table of loss values
loss_table = pd.DataFrame({'Output Variable': ['Cm', 'Cl', 'Cd'], 'Loss': loss_per_variable})

# Print the loss table
print("\nLoss Table:")
print(loss_table)

# Plot input signals
fig, axs = plt.subplots(3, 1, figsize=(12, 12))
input_variables = ['Alpha', 'Delta_E', 'Q']

for i in range(3):
    axs[i].plot(X_test[:, i], label=input_variables[i], color='blue')
    axs[i].set_xlabel('Time (sec)')
    axs[i].set_ylabel(input_variables[i])
    axs[i].legend()
    axs[i].grid(True)

plt.tight_layout()

# Plot the target and predicted values for each output variable
fig, axs = plt.subplots(3, 1, figsize=(12,12))
output_variables = ['Cm', 'Cl', 'Cd']

for i in range(3):
    axs[i].plot(Y_test[:, i], label='Measured', color='blue', linewidth=1)
    axs[i].plot(Y_pred[:, i], label='Neural Network Estimate', color='orange')
    axs[i].set_xlabel('Time (sec)')
    axs[i].set_ylabel(output_variables[i])
    axs[i].legend()
    axs[i].grid(True)

plt.tight_layout()

# Plot histograms of Cd, Cm, Cl
fig, axs = plt.subplots(3, 1, figsize=(12, 12))

for i in range(3):
    axs[i].hist(Y_test[:, i], bins=25, alpha=0.5, label='Measured', color='blue')
    axs[i].hist(Y_pred[:, i], bins=25, alpha=0.5, label='Neural Network Estimate', color='orange')
    axs[i].axvline(np.median(Y_pred[:, i]), color='red', linestyle='dotted', linewidth=1)
    axs[i].set_xlabel(output_variables[i])
    axs[i].set_ylabel('Frequency')
    axs[i].legend()
    axs[i].grid(True)

plt.tight_layout()

# Plot residuals
fig, axs = plt.subplots(3, 1, figsize=(12, 12))

for i in range(3):
    axs[i].plot(residuals[:, i], label='Residuals', color='blue')
    axs[i].axhline(0, color='red', linestyle='dotted', linewidth=1)
    axs[i].set_xlabel('Time (sec)')
    axs[i].set_ylabel('Residuals')
    axs[i].legend()
    axs[i].grid(True)

plt.tight_layout()

# Display all figure windows simultaneously
plt.show()
