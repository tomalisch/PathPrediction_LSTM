import random
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Subset

# Seeds & CUDA
seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed_value)

# Config
timesteps = 10
batch_size = 64
hidden_size = 64
num_layers = 1
epochs = 20
lr = 1e-3
generate_steps = 15 * 60  # generate N steps for visualization

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
repo_root = Path.cwd()

print(f"Using device: {device}")
print(f"Repository root: {repo_root}")

# Load data
data_path = repo_root / "TrackingData.csv"
if not data_path.exists():
    raise FileNotFoundError(f"Could not find input file: {data_path}")

print("Loading data...")
data = pd.read_csv(data_path)
xy = data.iloc[:, :2].dropna().to_numpy(dtype=np.float32)

if xy.shape[0] <= timesteps:
    raise ValueError(
        f"Not enough frames ({xy.shape[0]}) for timesteps={timesteps}. "
        f"You need more than {timesteps} rows."
    )

print(f"Loaded {xy.shape[0]} frames with {xy.shape[1]} features (X, Y).")

# Scale x and y to [0, 1]
scaler = MinMaxScaler()
xy_scaled = scaler.fit_transform(xy).astype(np.float32)

# Sliding window
print("Building sliding windows...")
X, y = [], []
for i in tqdm(range(len(xy_scaled) - timesteps), desc="Sliding windows", dynamic_ncols=True):
    X.append(xy_scaled[i:i + timesteps])   # shape: (timesteps, 2)
    y.append(xy_scaled[i + timesteps])     # shape: (2,)

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)

X_tensor = torch.tensor(X)
y_tensor = torch.tensor(y)
dataset = TensorDataset(X_tensor, y_tensor)

print(f"Constructed {len(dataset)} training examples.")

# Training/validation split
n_total = len(dataset)
n_train = int(0.8 * n_total)
n_val = n_total - n_train

train_indices = list(range(n_train))
val_indices = list(range(n_train, n_total))

train_ds = Subset(dataset, train_indices)
val_ds = Subset(dataset, val_indices)

pin_memory = torch.cuda.is_available()

train_loader = DataLoader(
    train_ds,
    batch_size=batch_size,
    shuffle=True,
    pin_memory=pin_memory
)
val_loader = DataLoader(
    val_ds,
    batch_size=batch_size,
    shuffle=False,
    pin_memory=pin_memory
)

print(f"Train samples: {len(train_ds)} | Validation samples: {len(val_ds)}")

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
        # x: (batch, seq_len, 2)
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

print("Starting training...")
epoch_bar = tqdm(range(1, epochs + 1), desc="Epochs", dynamic_ncols=True)

for epoch in epoch_bar:
    # ---- Training ----
    model.train()
    running_train = 0.0

    train_bar = tqdm(
        train_loader,
        desc=f"Train {epoch:02d}/{epochs}",
        leave=False,
        dynamic_ncols=True
    )
    for xb, yb in train_bar:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()

        running_train += loss.item() * xb.size(0)
        train_bar.set_postfix(loss=f"{loss.item():.6f}")

    train_loss = running_train / len(train_ds)
    train_losses.append(train_loss)

    # ---- Validation ----
    model.eval()
    running_val = 0.0

    val_bar = tqdm(
        val_loader,
        desc=f"Val   {epoch:02d}/{epochs}",
        leave=False,
        dynamic_ncols=True
    )
    with torch.no_grad():
        for xb, yb in val_bar:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            pred = model(xb)
            loss = criterion(pred, yb)
            running_val += loss.item() * xb.size(0)
            val_bar.set_postfix(loss=f"{loss.item():.6f}")

    val_loss = running_val / len(val_ds)
    val_losses.append(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    epoch_bar.set_postfix(train=f"{train_loss:.6f}", val=f"{val_loss:.6f}", best=f"{best_val_loss:.6f}")

print("Training complete.")

# Restore best model
if best_state is not None:
    model.load_state_dict(best_state)
    model_path = repo_root / "lstm_best_model.pt"
    torch.save(best_state, model_path)
    print(f"Saved best model to: {model_path}")

# Plot training vs validation loss
loss_plot_path = repo_root / "training_validation_loss.png"
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="train")
plt.plot(val_losses, label="val")
plt.xlabel("Epoch")
plt.ylabel("MSE loss")
plt.legend()
plt.title("LSTM training and validation loss")
plt.tight_layout()
plt.savefig(loss_plot_path, dpi=300)
plt.close()
print(f"Saved loss plot to: {loss_plot_path}")

# Quick one-step prediction example from validation set
print("Generating one-step validation example...")
model.eval()

val_example_idx = random.choice(val_indices)
test_seq = X_tensor[val_example_idx:val_example_idx + 1].to(device)
true_next = y[val_example_idx]

with torch.no_grad():
    pred_next = model(test_seq).cpu().numpy()[0]

test_seq_unscaled = scaler.inverse_transform(X[val_example_idx])
true_next_unscaled = scaler.inverse_transform(true_next.reshape(1, -1))[0]
pred_next_unscaled = scaler.inverse_transform(pred_next.reshape(1, -1))[0]

one_step_plot_path = repo_root / "one_step_validation_example.png"
plt.figure(figsize=(6, 6))
plt.scatter(test_seq_unscaled[:, 0], test_seq_unscaled[:, 1], label="history")
plt.scatter(true_next_unscaled[0], true_next_unscaled[1], label="true next")
plt.scatter(pred_next_unscaled[0], pred_next_unscaled[1], label="pred next")
plt.legend()
plt.title("One-step prediction on held-out validation window")
plt.axis("equal")
plt.tight_layout()
plt.savefig(one_step_plot_path, dpi=300)
plt.close()
print(f"Saved one-step validation plot to: {one_step_plot_path}")

# Autoregressive rollout
print("Generating autoregressive rollout...")

# Seed from validation region
seed_idx = val_indices[0]
seed_window = X[seed_idx].copy()  # shape: (timesteps, 2)
moving_window = torch.tensor(seed_window, dtype=torch.float32, device=device)

simulated = []

model.eval()
with torch.no_grad():
    rollout_bar = tqdm(range(generate_steps), desc="Rollout", dynamic_ncols=True)
    for _ in rollout_bar:
        pred = model(moving_window.unsqueeze(0)).cpu().numpy()[0]
        simulated.append(pred)

        # Slide window and append prediction
        moving_window = torch.roll(moving_window, shifts=-1, dims=0)
        moving_window[-1] = torch.tensor(pred, dtype=torch.float32, device=device)

simulated = np.array(simulated, dtype=np.float32)
simulated_unscaled = scaler.inverse_transform(simulated)
seed_unscaled = scaler.inverse_transform(seed_window)

# Save simulated coordinates as CSV in repo root
sim_csv_path = repo_root / "simulated_fly_path.csv"
sim_df = pd.DataFrame(simulated_unscaled, columns=["x", "y"])
sim_df.insert(0, "frame", np.arange(len(sim_df)))
sim_df.to_csv(sim_csv_path, index=False)
print(f"Saved simulated coordinates to: {sim_csv_path}")

# Save simulated path plot in repo root
sim_plot_path = repo_root / "simulated_fly_path.png"
plt.figure(figsize=(6, 6))
plt.plot(seed_unscaled[:, 0], seed_unscaled[:, 1], label="seed history", linewidth=2)
plt.plot(simulated_unscaled[:, 0], simulated_unscaled[:, 1], label="simulated rollout", linewidth=1)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Autoregressively simulated fly path")
plt.legend()
plt.axis("equal")
plt.tight_layout()
plt.savefig(sim_plot_path, dpi=300)
plt.close()
print(f"Saved simulated path plot to: {sim_plot_path}")

print("Done.")