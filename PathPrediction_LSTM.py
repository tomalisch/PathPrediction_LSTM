import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Subset

# Seeds
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# Config
timesteps = 10
batch_size = 64
hidden_size = 64
num_layers = 1
epochs = 20
lr = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
data = pd.read_csv("TrackingData.csv")
xy = data.iloc[:, :2].dropna().to_numpy(dtype=np.float32)

# Scale x and y to [0, 1]
scaler = MinMaxScaler()
xy_scaled = scaler.fit_transform(xy).astype(np.float32)

# Sliding window
X, y = [], []
for i in range(len(xy_scaled) - timesteps):
    X.append(xy_scaled[i:i + timesteps])   # (timesteps, 2)
    y.append(xy_scaled[i + timesteps])     # (2,)

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)

X_tensor = torch.tensor(X)
y_tensor = torch.tensor(y)
dataset = TensorDataset(X_tensor, y_tensor)

# Training/validation split
n_total = len(dataset)
n_train = int(0.8 * n_total)

train_indices = list(range(n_train))
val_indices = list(range(n_train, n_total))

train_ds = Subset(dataset, train_indices)
val_ds = Subset(dataset, val_indices)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

# LSTM model
class XYLSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        """ x: (batch, seq_len, 2)"""
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]   # last timestep
        pred = self.fc(last_hidden)   # (batch, 2)
        return pred

model = XYLSTM(input_size=2, hidden_size=hidden_size, num_layers=num_layers).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()

# Training loop
train_losses = []
val_losses = []
best_val_loss = float("inf")
best_state = None

for epoch in range(epochs):
    model.train()
    running_train = 0.0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()

        running_train += loss.item() * xb.size(0)

    train_loss = running_train / len(train_ds)
    train_losses.append(train_loss)

    model.eval()
    running_val = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            running_val += loss.item() * xb.size(0)

    val_loss = running_val / len(val_ds)
    val_losses.append(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print(f"Epoch {epoch+1:02d} | train loss {train_loss:.6f} | val loss {val_loss:.6f}")

# Restore best model
if best_state is not None:
    model.load_state_dict(best_state)

# Plot training vs validation loss
plt.figure()
plt.plot(train_losses, label="train")
plt.plot(val_losses, label="val")
plt.xlabel("Epoch")
plt.ylabel("MSE loss")
plt.legend()
plt.title("LSTM training and validation loss")
plt.show()

# Quick one-step prediction example from validation set
model.eval()

val_example_idx = random.choice(val_indices)
test_seq = X_tensor[val_example_idx:val_example_idx + 1].to(device)
true_next = y[val_example_idx]

with torch.no_grad():
    pred_next = model(test_seq).cpu().numpy()[0]

test_seq_unscaled = scaler.inverse_transform(X[val_example_idx])
true_next_unscaled = scaler.inverse_transform(true_next.reshape(1, -1))[0]
pred_next_unscaled = scaler.inverse_transform(pred_next.reshape(1, -1))[0]

plt.figure()
plt.scatter(test_seq_unscaled[:, 0], test_seq_unscaled[:, 1], label="history")
plt.scatter(true_next_unscaled[0], true_next_unscaled[1], label="true next")
plt.scatter(pred_next_unscaled[0], pred_next_unscaled[1], label="pred next")
plt.legend()
plt.title("One-step prediction on held-out validation window")
plt.axis("equal")
plt.show()

# Autoregressive rollout
length_frames = 30 * 60  # 1 minute at 30 fps

# Seed from validation region
seed_idx = val_indices[0]
seed = X[seed_idx].copy()
moving_window = torch.tensor(seed, dtype=torch.float32, device=device)

simulated = []

model.eval()
with torch.no_grad():
    for _ in range(length_frames):
        pred = model(moving_window.unsqueeze(0)).cpu().numpy()[0]
        simulated.append(pred)

        # Slide window and append prediction
        moving_window = torch.roll(moving_window, shifts=-1, dims=0)
        moving_window[-1] = torch.tensor(pred, dtype=torch.float32, device=device)

simulated = np.array(simulated, dtype=np.float32)
simulated_unscaled = scaler.inverse_transform(simulated)
seed_unscaled = scaler.inverse_transform(seed)

plt.figure(figsize=(6, 6))
plt.plot(seed_unscaled[:, 0], seed_unscaled[:, 1], label="seed history", linewidth=2)
plt.plot(simulated_unscaled[:, 0], simulated_unscaled[:, 1], label="simulated rollout", linewidth=1)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Autoregressively simulated fly path")
plt.legend()
plt.axis("equal")
plt.show()