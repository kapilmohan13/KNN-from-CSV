import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Create a simple time series dataset
dfLoad = pd.read_csv('..//data//GOLDBEES.csv')
data = np.sin(np.linspace(0, 100, 1000))  # Sine wave data
dfLoad['time'] = pd.to_datetime(dfLoad['time'])
dfLoad.sort_values(by='time', inplace=True)

df = dfLoad[['intc']]
# df = dfLoad['intc']
print(df)
df.rename(columns={'intc': 'value'}, inplace=True)
df.reset_index(drop=True, inplace=True)
# df.columns = ['intc', 'value']
plt.plot(df['value'])
plt.show()

from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import (DataLoader, TensorDataset)

# Normalize the data
scaler = MinMaxScaler()
df['value'] = scaler.fit_transform(df['value'].values.reshape(-1, 1))

# Create sequences
def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 50
X, y = create_sequences(df['value'].values, seq_length)
# Convert to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# Split into train/test sets
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

train_data = TensorDataset(X_train, y_train)
test_data = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32)


import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

input_size = 1
hidden_size = 50
output_size = 1
num_layers = 2

model = LSTM(input_size, hidden_size, output_size, num_layers)


import torch.optim as optim

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    for inputs, targets in train_loader:
        inputs = inputs.unsqueeze(-1)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


model.eval()
predictions = []
with torch.no_grad():
    for inputs, _ in test_loader:
        inputs = inputs.unsqueeze(-1)
        outputs = model(inputs)
        predictions.append(outputs.cpu().numpy())

predictions = np.concatenate(predictions)
predictions = scaler.inverse_transform(predictions)

y_test = y_test.cpu().numpy()
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# Plot the results
plt.plot(y_test, label='True')
plt.plot(predictions, label='Predicted')
plt.legend()
plt.show()
