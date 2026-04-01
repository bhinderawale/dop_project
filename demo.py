from dataHelpers import generate_data
import numpy as np
import matplotlib.pyplot as plt
from ASATVN import ASATVN
from evaluate import evaluate
from train_model import train
from evaluate import format_input
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import torch 

#Use a fixed seed for repreducible results
np.random.seed(1)

data = pd.read_csv('input.csv', index_col=0)
data.index = pd.to_datetime(data.index, format='%d-%m-%Y')
data = data.ffill().bfill()
data = data[['PM2.5']]
scaler = MinMaxScaler()
data[:] = scaler.fit_transform(data)
train_x, train_y, test_x, test_y, valid_x, valid_y, period, mm = generate_data(data.values, period=64)

# Model parameters
print(torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device count:", torch.cuda.device_count())
print("Using device:", device)

test_x = torch.from_numpy(test_x).float().to(device)
test_y = torch.from_numpy(test_y).float().to(device)

in_seq_length = 2 * period
out_seq_length = period
hidden_dim = 75
head = 4
M = 2
input_dim = 1
output_dim = 1
learning_rate = 0.001
batch_size = 4
change_dim = 64
dis_alpha = 0.1
d_model = 64

# Initialise model
ASA = ASATVN(in_seq_length=in_seq_length, out_seq_length=out_seq_length, d_model=d_model,input_dim=input_dim,
                        hidden_dim=hidden_dim, h=head, M=M, output_dim=output_dim, batch_size = batch_size,
                        period=period, n_epochs = 1, learning_rate = learning_rate, train_x=train_x, save_file = './asatvn.pt')

# Train the model
training_costs, training_d_costs, validation_costs = train(ASA, train_x, train_y, valid_x, valid_y, restore_session=False)

# Plot the training curves
# plt.figure()
# plt.plot(training_costs)
# plt.plot(validation_costs)

# Evaluate the model
mase,smape, nrmse = evaluate(ASA, test_x, test_y, return_lists=False)

print('MASE:', mase)
print('SMAPE:', smape)
print('NRMSE:', nrmse)
# Generate and plot forecasts for various samples from the test dataset
samples = [0, 12, 23]
predict_start = 24 
y_pred = ASA.forecast(test_x[:predict_start, samples, :], predict_start)
for i in range(len(samples)):
    pred= mm.inverse_transform(y_pred[:, i, 0][:,np.newaxis])
    true = mm.inverse_transform(test_y[predict_start:, :, :][:, samples[i], 0][:,np.newaxis])
    plt.figure()
    plt.plot(np.arange(ASA.in_seq_length, ASA.in_seq_length + ASA.out_seq_length),
             test_y[predict_start:, :, :][:, samples[i], 0],
             '-')
    plt.plot(np.arange(ASA.in_seq_length, ASA.in_seq_length + ASA.out_seq_length),
             y_pred[:, i, 0],
             '-', linewidth=0.7, label='mean')
plt.show()