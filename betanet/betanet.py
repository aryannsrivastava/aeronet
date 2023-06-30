import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanSquaredError

# Load the input and output data from text files
time = np.loadtxt('time.txt')
alpha_rad = np.loadtxt('alpha.txt')
delta_e_rad = np.loadtxt('deltae.txt')
q = np.loadtxt('q.txt')
Cm = np.loadtxt('cm.txt')
Cl = np.loadtxt('cl.txt')
Cd = np.loadtxt('cd.txt')

# Convert alpha and delta_e from radians to degrees
alpha_deg = np.degrees(alpha_rad)
delta_e_deg = np.degrees(delta_e_rad)

# Combine the input and output data
X = np.column_stack((alpha_deg, delta_e_deg, q)) 
Y = np.column_stack((Cm, Cl, Cd))

# Normalize the input and output data
scaler_X = MinMaxScaler()
X_norm = scaler_X.fit_transform(X)

scaler_Y = MinMaxScaler()
Y_norm = scaler_Y.fit_transform(Y)

# Define the model architecture
model = Sequential()
model.add(Dense(256, input_dim=3, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(3))

# Define the optimizer (Adam)
optimizer = 'adam'

# Compile the model
model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=[MeanSquaredError()])

# Train the model
history = model.fit(X_norm, Y_norm, epochs=100, batch_size=32, verbose=1, validation_split=0.2)

# Make predictions on all data
Y_pred_norm = model.predict(X_norm)

# Denormalize the predictions
Y_pred = scaler_Y.inverse_transform(Y_pred_norm)

# Calculate residuals for each output variable
residuals = Y - Y_pred

# Calculate loss for each output variable
loss_per_variable = np.mean(np.square(residuals), axis=0)

# Create a table of loss values
loss_table = pd.DataFrame({'Output Variable': ['Cm', 'Cl', 'Cd'], 'Loss': loss_per_variable})

# Print the loss table
print("\nLoss Table:")
print(loss_table)

# Plot input signals
fig, axs = plt.subplots(3, 1, figsize=(12, 12))
input_variables = ['Alpha (deg)', 'Delta_E (deg)', 'Q']

for i in range(3):
    axs[i].plot(time, X[:, i], label=input_variables[i], color='blue')
    axs[i].set_xlabel('Time (sec)')
    axs[i].set_ylabel(input_variables[i])
    axs[i].legend()
    axs[i].grid(True)

plt.tight_layout()

# Plot the target and predicted values for each output variable
fig, axs = plt.subplots(3, 1, figsize=(12, 12))
output_variables = ['Cm', 'Cl', 'Cd']

for i in range(3):
    axs[i].plot(time, Y[:, i], label='Measured', color='blue', linewidth=1)
    axs[i].plot(time, Y_pred[:, i], label='Neural Network Estimate', color='orange')
    axs[i].set_xlabel('Time (sec)')
    axs[i].set_ylabel(output_variables[i])
    axs[i].legend()
    axs[i].grid(True)

plt.tight_layout()

# Plot histograms of Cd, Cm, Cl
fig, axs = plt.subplots(3, 1, figsize=(12, 12))

for i in range(3):
    axs[i].hist(Y[:, i], bins=25, alpha=0.5, label='Measured', color='blue')
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
    axs[i].plot(time, residuals[:, i], label='Residuals', color='blue')
    axs[i].axhline(0, color='red', linestyle='dotted', linewidth=1)
    axs[i].set_xlabel('Time (sec)')
    axs[i].set_ylabel('Residuals')
    axs[i].legend()
    axs[i].grid(True)

plt.tight_layout()

# Display all figure windows simultaneously
plt.show()
