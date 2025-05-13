import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean\_squared\_error
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Set random seed for reproducibility

torch.manual\_seed(42)
np.random.seed(42)

# Check if GPU is available

device = torch.device('cuda' if torch.cuda.is\_available() else 'cpu')
print(f"Using device: {device}")

# Generate synthetic time series data

def generate\_sine\_wave\_data(n\_samples=1000):
time = np.arange(0, n\_samples, 1)
\# Create a sine wave with some noise
sine\_wave = np.sin(0.1 \* time) + 0.1 \* np.random.randn(n\_samples)
\# Add trend
trend = 0.001 \* time
\# Combine signals
series = sine\_wave + trend
return series

# Function to prepare data for LSTM (create time windows)

def create\_dataset(dataset, time\_step=1):
X, y = \[], \[]
for i in range(len(dataset) - time\_step - 1):
a = dataset\[i:(i + time\_step), 0]
X.append(a)
y.append(dataset\[i + time\_step, 0])
return np.array(X), np.array(y)

# Generate data

series = generate\_sine\_wave\_data(n\_samples=1000)
df = pd.DataFrame(series, columns=\['value'])

# Plot the data

plt.figure(figsize=(12, 6))
plt.plot(df\['value'])
plt.title('Synthetic Time Series Data')
plt.xlabel('Time Steps')
plt.ylabel('Value')
plt.grid(True)
plt.savefig('time\_series\_data.png')
plt.close()

# Normalize the data

scaler = MinMaxScaler(feature\_range=(0, 1))
scaled\_data = scaler.fit\_transform(df.values)

# Split into train and test sets (80-20 split)

train\_size = int(len(scaled\_data) \* 0.8)
test\_size = len(scaled\_data) - train\_size
train\_data = scaled\_data\[0\:train\_size, :]
test\_data = scaled\_data\[train\_size\:len(scaled\_data), :]

# Define time steps for LSTM

time\_step = 60  # Look back 60 time steps

# Prepare training data

X\_train, y\_train = create\_dataset(train\_data, time\_step)

# Prepare test data

X\_test, y\_test = create\_dataset(test\_data, time\_step)

# Convert to PyTorch tensors

X\_train\_tensor = torch.FloatTensor(X\_train).view(-1, time\_step, 1).to(device)
y\_train\_tensor = torch.FloatTensor(y\_train).to(device)
X\_test\_tensor = torch.FloatTensor(X\_test).view(-1, time\_step, 1).to(device)
y\_test\_tensor = torch.FloatTensor(y\_test).to(device)

# Create DataLoader for batch processing

batch\_size = 32
train\_dataset = TensorDataset(X\_train\_tensor, y\_train\_tensor)
train\_loader = DataLoader(dataset=train\_dataset, batch\_size=batch\_size, shuffle=True)

# Define LSTM model

class LSTMModel(nn.Module):
def **init**(self, input\_dim=1, hidden\_dim=50, num\_layers=2, output\_dim=1, dropout=0.2):
super(LSTMModel, self).**init**()
self.hidden\_dim = hidden\_dim
self.num\_layers = num\_layers

```
    # LSTM layers
    self.lstm = nn.LSTM(
        input_size=input_dim,
        hidden_size=hidden_dim,
        num_layers=num_layers,
        batch_first=True,
        dropout=dropout
    )

    # Fully connected output layer
    self.fc = nn.Linear(hidden_dim, output_dim)

def forward(self, x):
    # Initialize hidden state with zeros
    h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
    # Initialize cell state
    c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

    # Forward propagate LSTM
    # out: tensor of shape (batch_size, seq_length, hidden_size)
    out, (hn, cn) = self.lstm(x, (h0, c0))

    # Decode the hidden state of the last time step
    out = self.fc(out[:, -1, :])

    return out
```

# Instantiate the model

input\_dim = 1  # One feature (univariate time series)
hidden\_dim = 50  # Number of LSTM units
num\_layers = 2  # Number of LSTM layers
output\_dim = 1  # One output
dropout = 0.2  # Dropout rate

model = LSTMModel(input\_dim, hidden\_dim, num\_layers, output\_dim, dropout).to(device)
print(model)

# Loss function and optimizer

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function

def train\_model(model, train\_loader, criterion, optimizer, num\_epochs=20):
model.train()
train\_losses = \[]

```
for epoch in range(num_epochs):
    running_loss = 0.0

    for batch_idx, (data, targets) in enumerate(train_loader):
        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, targets.unsqueeze(1))

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    train_losses.append(epoch_loss)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')

return train_losses
```

# Train the model

num\_epochs = 20
train\_losses = train\_model(model, train\_loader, criterion, optimizer, num\_epochs)

# Plot the training loss

plt.figure(figsize=(12, 6))
plt.plot(train\_losses, label='Training Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('model\_loss.png')
plt.close()

# Evaluation function

def evaluate\_model(model, X, y):
model.eval()
with torch.no\_grad():
predictions = model(X)
return predictions

# Make predictions

model.eval()
train\_predict = evaluate\_model(model, X\_train\_tensor, y\_train\_tensor).cpu().numpy()
test\_predict = evaluate\_model(model, X\_test\_tensor, y\_test\_tensor).cpu().numpy()

# Inverse transform predictions to original scale

train\_predict = scaler.inverse\_transform(train\_predict)
y\_train\_inv = scaler.inverse\_transform(y\_train.reshape(-1, 1))
test\_predict = scaler.inverse\_transform(test\_predict)
y\_test\_inv = scaler.inverse\_transform(y\_test.reshape(-1, 1))

# Calculate RMSE

train\_rmse = math.sqrt(mean\_squared\_error(y\_train\_inv, train\_predict))
test\_rmse = math.sqrt(mean\_squared\_error(y\_test\_inv, test\_predict))
print(f'Train RMSE: {train\_rmse:.4f}')
print(f'Test RMSE: {test\_rmse:.4f}')

# Visualize the predictions

# Prepare data for plotting

train\_predict\_plot = np.empty\_like(scaled\_data)
train\_predict\_plot\[:, :] = np.nan
train\_predict\_plot\[time\_step\:len(train\_predict) + time\_step, :] = train\_predict

# Shift test predictions for plotting

test\_predict\_plot = np.empty\_like(scaled\_data)
test\_predict\_plot\[:, :] = np.nan

# Fix for the shape mismatch issue

start\_idx = len(train\_predict) + (time\_step \* 2) - 1
end\_idx = start\_idx + len(test\_predict)

# Make sure we don't exceed the array bounds

end\_idx = min(end\_idx, len(scaled\_data))
test\_predict\_plot\[start\_idx\:end\_idx, :] = test\_predict\[:end\_idx - start\_idx]

# Plot baseline and predictions

plt.figure(figsize=(12, 6))
plt.plot(scaler.inverse\_transform(scaled\_data), label='Actual Data')
plt.plot(train\_predict\_plot, label='Training Predictions')
plt.plot(test\_predict\_plot, label='Testing Predictions')
plt.title('Time Series Prediction')
plt.xlabel('Time Steps')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.savefig('predictions.png')
plt.show()

# Function to make future predictions

def predict\_future(model, last\_sequence, n\_future=100):
model.eval()
future\_predictions = \[]

```
# Convert input to tensor and reshape to (1, time_step, 1)
current_sequence = torch.FloatTensor(last_sequence).view(1, time_step, 1).to(device)

with torch.no_grad():
    for _ in range(n_future):
        # Get prediction for next step
        prediction = model(current_sequence)

        # Store prediction
        future_predictions.append(prediction.cpu().numpy())

        # Update sequence: remove oldest, add new prediction
        # Make sure prediction has the right shape for concatenation
        if len(prediction.shape) == 1:
            # If prediction is [batch_size]
            prediction_reshaped = prediction.view(1, 1, 1)
        elif len(prediction.shape) == 2:
            # If prediction is [batch_size, 1]
            prediction_reshaped = prediction.view(1, 1, 1)
        else:
            # Already has the right shape
            prediction_reshaped = prediction.view(1, 1, 1)

        # Update the sequence by removing oldest timestep and adding newest prediction
        current_sequence = torch.cat((current_sequence[:, 1:, :], prediction_reshaped), dim=1)

return np.array(future_predictions).reshape(-1, 1)
```

# Predict future values

last\_batch = scaled\_data\[-time\_step:].reshape(1, time\_step, 1)
future\_preds = predict\_future(model, last\_batch, n\_future=200)
future\_preds = scaler.inverse\_transform(future\_preds)

# Plot the future predictions

plt.figure(figsize=(12, 6))
plt.plot(range(len(df)), scaler.inverse\_transform(scaled\_data), label='Historical Data')
plt.plot(range(len(df), len(df) + len(future\_preds)), future\_preds, label='Future Predictions', color='red')
plt.title('Future Time Series Prediction')
plt.xlabel('Time Steps')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.savefig('future\_predictions.png')
plt.show()

# Save the model

torch.save(model.state\_dict(), 'lstm\_model.pth')
print("LSTM Time Series Prediction with PyTorch Completed!")
