import yfinance as yf
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# Load stock data from Yahoo Finance API
stock_data = yf.download("NVDA", start="2021-01-01", end="2023-09-07")
prices = torch.tensor(stock_data["Adj Close"].values, dtype=torch.float32, device="mps")

# Define a model to predict stock price range
class PriceRangeModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PriceRangeModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.softmax(out)
        return out

# Initialize the model
model = PriceRangeModel(input_size=1, hidden_size=35, output_size=3)
model.to("mps")

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001)

# Train the model
num_epochs = 10000
losses = []
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(prices[:-1].unsqueeze(dim=1))
    labels = torch.zeros(len(prices) - 1, dtype=torch.long, device="mps")
    labels[(prices[1:] - prices[:-1]) > 0] = 1
    labels[(prices[1:] - prices[:-1]) < 0] = 2
    loss = criterion(outputs, labels)
    losses.append(loss.item())

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Plot the loss curve
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.show()

# Test the model
with torch.no_grad():
    test_input = prices[-1].unsqueeze(dim=0)
    test_output = model(test_input)
    test_pred = torch.argmax(test_output).item()
    if test_pred == 0:
        print("Next day's stock price is likely to decrease")
    elif test_pred == 1:
        print("Next day's stock price is likely to stay the same")
    else:
        print("Next day's stock price is likely to increase")

# Calculate additional metrics
from sklearn.metrics import confusion_matrix, mean_squared_error
from math import sqrt

# Convert predictions to numpy arrays
predictions = model(prices[:-1].unsqueeze(dim=1)).detach().cpu().numpy()
predicted_labels = np.argmax(predictions, axis=1)


# Calculate confusion matrix
conf_matrix = confusion_matrix(labels.cpu().numpy(), predicted_labels)

# Calculate Mean Squared Error (MSE) and Root Mean Squared Error (RMSE)
mse = mean_squared_error(prices[1:].cpu().numpy(), prices[:-1].cpu().numpy())
rmse = sqrt(mse)

# Calculate Directional Accuracy
directional_accuracy = np.mean(np.sign(predictions[:, 2] - predictions[:, 0]) == np.sign(prices[1:].cpu().numpy() - prices[:-1].cpu().numpy()))

# Implement backtesting (simple strategy: buy when predicted to increase, sell when predicted to decrease)
initial_balance = 10000
balance = initial_balance
shares = 0

for i in range(len(predictions)):
    if predicted_labels[i] == 2:  # Predicted to increase, buy
        shares_to_buy = balance / prices[i]
        shares += shares_to_buy
        balance -= shares_to_buy * prices[i]
    elif predicted_labels[i] == 0:  # Predicted to decrease, sell
        balance += shares * prices[i]
        shares = 0

# Calculate Sharpe Ratio
returns = (balance + shares * prices[-1]) - initial_balance
daily_returns = np.diff(prices.cpu().numpy())
sharpe_ratio = (returns / initial_balance) / (np.std(daily_returns) * np.sqrt(252))  # Assuming 252 trading days in a year

print("Accuracy:", np.sum(np.diag(conf_matrix)) / np.sum(conf_matrix))
print("Confusion Matrix:")
print(conf_matrix)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("Directional Accuracy:", directional_accuracy)
print("Sharpe Ratio:", sharpe_ratio)
print("Backtesting Final Balance:", balance + shares * prices[-1])

# Plot the actual stock prices and the predicted price range
fig, ax = plt.subplots()
ax.plot(stock_data.index, stock_data["Adj Close"], label="Actual Price")
ax.plot(stock_data.index[-1], prices[-1].item(), "go", label="Last Price")
ax.plot(stock_data.index[-1], test_input.item(), "ro", label="Test Price")
ax.plot(stock_data.index[-1], test_input.item() - 1, "bx", label="Lower Bound")
ax.plot(stock_data.index[-1], test_input.item() + 1, "bx", label="Upper Bound")
ax.legend()
ax.set_xlabel("Date")
ax.set_ylabel("Price")
plt.show()


# Plot the actual and predicted stock prices
actual_prices = stock_data["Adj Close"].values
predicted_prices = []
with torch.no_grad():
    for i in range(len(prices)):
        input = prices[i].unsqueeze(dim=0)
        output = model(input)
        pred = torch.argmax(output).item()
        if pred == 0:
            predicted_prices.append(actual_prices[i] * 0.9)
        elif pred == 1:
            predicted_prices.append(actual_prices[i])
        else:
            predicted_prices.append(actual_prices[i] * 1.1)

plt.plot(actual_prices, label="Actual Prices")
plt.plot(predicted_prices, label="Predicted Prices")
plt.xlabel("Time (days)")
plt.ylabel("Stock Price ($)")
plt