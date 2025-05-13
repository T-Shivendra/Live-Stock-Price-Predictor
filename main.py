import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# Generate synthetic time series data
def generate_sine_wave_data(n_samples=1000):
    time = np.arange(0, n_samples, 1)
    # Create a sine wave with some noise
    sine_wave = np.sin(0.1 * time) + 0.1 * np.random.randn(n_samples)
    # Add trend
    trend = 0.001 * time
    # Combine signals
    series = sine_wave + trend
    return series


# Function to prepare data for LSTM (create time windows)
def create_dataset(dataset, time_step=1):
    X, y = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        X.append(a)
        y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(y)


# Generate data
series = generate_sine_wave_data(n_samples=1000)
df = pd.DataFrame(series, columns=['value'])

# Plot the data
plt.figure(figsize=(12, 6))
plt.plot(df['value'])
plt.title('Synthetic Time Series Data')
plt.xlabel('Time Steps')
plt.ylabel('Value')
plt.grid(True)
plt.savefig('time_series_data.png')
plt.close()

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df.values)

# Split into train and test sets (80-20 split)
train_size = int(len(scaled_data) * 0.8)
test_size = len(scaled_data) - train_size
train_data = scaled_data[0:train_size, :]
test_data = scaled_data[train_size:len(scaled_data), :]

# Define time steps for LSTM
time_step = 60  # Look back 60 time steps

# Prepare training data
X_train, y_train = create_dataset(train_data, time_step)
# Prepare test data
X_test, y_test = create_dataset(test_data, time_step)

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train).view(-1, time_step, 1).to(device)
y_train_tensor = torch.FloatTensor(y_train).to(device)
X_test_tensor = torch.FloatTensor(X_test).view(-1, time_step, 1).to(device)
y_test_tensor = torch.FloatTensor(y_test).to(device)

# Create DataLoader for batch processing
batch_size = 32
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)


# Define LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=50, num_layers=2, output_dim=1, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

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


# Instantiate the model
input_dim = 1  # One feature (univariate time series)
hidden_dim = 50  # Number of LSTM units
num_layers = 2  # Number of LSTM layers
output_dim = 1  # One output
dropout = 0.2  # Dropout rate

model = LSTMModel(input_dim, hidden_dim, num_layers, output_dim, dropout).to(device)
print(model)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Training function
def train_model(model, train_loader, criterion, optimizer, num_epochs=20):
    model.train()
    train_losses = []

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


# Train the model
num_epochs = 20
train_losses = train_model(model, train_loader, criterion, optimizer, num_epochs)

# Plot the training loss
plt.figure(figsize=(12, 6))
plt.plot(train_losses, label='Training Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('model_loss.png')
plt.close()


# Evaluation function
def evaluate_model(model, X, y):
    model.eval()
    with torch.no_grad():
        predictions = model(X)
    return predictions


# Make predictions
model.eval()
train_predict = evaluate_model(model, X_train_tensor, y_train_tensor).cpu().numpy()
test_predict = evaluate_model(model, X_test_tensor, y_test_tensor).cpu().numpy()

# Inverse transform predictions to original scale
train_predict = scaler.inverse_transform(train_predict)
y_train_inv = scaler.inverse_transform(y_train.reshape(-1, 1))
test_predict = scaler.inverse_transform(test_predict)
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate RMSE
train_rmse = math.sqrt(mean_squared_error(y_train_inv, train_predict))
test_rmse = math.sqrt(mean_squared_error(y_test_inv, test_predict))
print(f'Train RMSE: {train_rmse:.4f}')
print(f'Test RMSE: {test_rmse:.4f}')

# Visualize the predictions
# Prepare data for plotting
train_predict_plot = np.empty_like(scaled_data)
train_predict_plot[:, :] = np.nan
train_predict_plot[time_step:len(train_predict) + time_step, :] = train_predict

# Shift test predictions for plotting
test_predict_plot = np.empty_like(scaled_data)
test_predict_plot[:, :] = np.nan

# Fix for the shape mismatch issue
start_idx = len(train_predict) + (time_step * 2) - 1
end_idx = start_idx + len(test_predict)
# Make sure we don't exceed the array bounds
end_idx = min(end_idx, len(scaled_data))
test_predict_plot[start_idx:end_idx, :] = test_predict[:end_idx - start_idx]

# Plot baseline and predictions
plt.figure(figsize=(12, 6))
plt.plot(scaler.inverse_transform(scaled_data), label='Actual Data')
plt.plot(train_predict_plot, label='Training Predictions')
plt.plot(test_predict_plot, label='Testing Predictions')
plt.title('Time Series Prediction')
plt.xlabel('Time Steps')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.savefig('predictions.png')
plt.show()


# Function to make future predictions
def predict_future(model, last_sequence, n_future=100):
    model.eval()
    future_predictions = []

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


# Predict future values
last_batch = scaled_data[-time_step:].reshape(1, time_step, 1)
future_preds = predict_future(model, last_batch, n_future=200)
future_preds = scaler.inverse_transform(future_preds)

# Plot the future predictions
plt.figure(figsize=(12, 6))
plt.plot(range(len(df)), scaler.inverse_transform(scaled_data), label='Historical Data')
plt.plot(range(len(df), len(df) + len(future_preds)), future_preds, label='Future Predictions', color='red')
plt.title('Future Time Series Prediction')
plt.xlabel('Time Steps')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.savefig('future_predictions.png')
plt.show()

# Save the model
torch.save(model.state_dict(), 'lstm_model.pth')
print("LSTM Time Series Prediction with PyTorch Completed!")