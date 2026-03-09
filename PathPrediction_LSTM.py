import random
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Seeds & CUDA
seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("CUDA not available, using CPU")
    print(f"torch.version.cuda = {torch.version.cuda}")

# Config
interpolate_missing_data = False
max_nan_fraction = 0.10
timesteps = 60
batch_size = 512
hidden_size = 512
num_layers = 2
epochs = 100
lr = 1e-3
rollout_steps = 5
huber_delta = 0.01
train_samples_per_epoch = 300_000
val_samples_per_epoch = 20_000
generate_steps = 30 * 60  # generate N steps for visualization

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
repo_root = Path.cwd()

print(f"Using device: {device}")
print(f"Repository root: {repo_root}")

# Load wide multi-fly data
data = pd.read_csv("TrackingData.csv", dtype=np.float32)

# Drop the first column ("frame"), keep only Fly_i_X / Fly_i_Y columns
coords = data.iloc[:, 1:].to_numpy(dtype=np.float32)

n_frames, n_coord_cols = coords.shape
if n_coord_cols % 2 != 0:
    raise ValueError(f"Expected an even number of coordinate columns, got {n_coord_cols}")

n_flies = n_coord_cols // 2
print(f"Loaded {n_frames} frames for {n_flies} flies")

# Reshape from (frames, 2*flies) -> (flies, frames, 2)
fly_data = coords.reshape(n_frames, n_flies, 2).transpose(1, 0, 2)

# Allow partial missingness per fly, but require enough valid data overall
valid_frame_mask = np.isfinite(fly_data).all(axis=2)   # shape: (n_flies, n_frames)
valid_fraction_per_fly = valid_frame_mask.mean(axis=1)

print("Valid-frame fraction per fly:")
print(f"  min    = {valid_fraction_per_fly.min():.4f}")
print(f"  mean   = {valid_fraction_per_fly.mean():.4f}")
print(f"  median = {np.median(valid_fraction_per_fly):.4f}")
print(f"  max    = {valid_fraction_per_fly.max():.4f}")

# Keep flies with enough usable data
keep_mask = valid_fraction_per_fly >= (1.0 - max_nan_fraction)
fly_data = fly_data[keep_mask]
valid_frame_mask = valid_frame_mask[keep_mask]

n_valid_flies = fly_data.shape[0]
print(f"Keeping {n_valid_flies} flies with >= {(1.0 - max_nan_fraction):.0%} valid frames")

if n_valid_flies < 2:
    raise ValueError("Need at least 2 flies after filtering")

# Optional interpolation
if interpolate_missing_data:
    print("Interpolating missing coordinates within each fly...")
    for i in tqdm(range(n_valid_flies), desc="Interpolating flies", dynamic_ncols=True):
        df_xy = pd.DataFrame(fly_data[i], columns=["x", "y"])
        df_xy = df_xy.interpolate(method="linear", limit_direction="both")
        fly_data[i] = df_xy.to_numpy(dtype=np.float32)

    # After interpolation, recompute validity mask
    valid_frame_mask = np.isfinite(fly_data).all(axis=2)

# Fit one shared scaler on all finite coordinates
flat = fly_data.reshape(-1, 2)
finite_rows = np.isfinite(flat).all(axis=1)

scaler = MinMaxScaler()
scaler.fit(flat[finite_rows])

# Transform only finite rows; preserve NaNs if interpolation is off
flat_scaled = flat.copy()
flat_scaled[finite_rows] = scaler.transform(flat[finite_rows]).astype(np.float32)

fly_data_scaled = flat_scaled.reshape(fly_data.shape).astype(np.float32)

# Split by fly
fly_order = np.random.permutation(n_valid_flies)
n_train_flies = int(0.8 * n_valid_flies)

train_fly_indices = fly_order[:n_train_flies]
val_fly_indices = fly_order[n_train_flies:]

print(f"Train flies: {len(train_fly_indices)} | Val flies: {len(val_fly_indices)}")

class RandomWindowDataset(Dataset):
    def __init__(self, fly_data, valid_frame_mask, fly_indices, timesteps, rollout_steps, samples_per_epoch):
        self.fly_data = fly_data
        self.valid_frame_mask = valid_frame_mask
        self.timesteps = timesteps
        self.rollout_steps = rollout_steps
        self.samples_per_epoch = samples_per_epoch
        self.required_len = timesteps + rollout_steps
        self.n_frames = fly_data.shape[1]

        if self.n_frames < self.required_len:
            raise ValueError(
                f"Not enough frames ({self.n_frames}) for timesteps={timesteps} "
                f"and rollout_steps={rollout_steps}"
            )

        # For each fly, precompute all valid start positions where the full
        # input window + target rollout contain no missing data.
        self.fly_indices = []
        self.valid_starts_per_fly = []

        for fly_idx in fly_indices:
            mask = self.valid_frame_mask[fly_idx].astype(np.int32)

            # Convolution counts how many valid frames are in each candidate window
            valid_counts = np.convolve(
                mask,
                np.ones(self.required_len, dtype=np.int32),
                mode="valid"
            )
            valid_starts = np.flatnonzero(valid_counts == self.required_len)

            if len(valid_starts) > 0:
                self.fly_indices.append(int(fly_idx))
                self.valid_starts_per_fly.append(valid_starts)

        if len(self.fly_indices) == 0:
            raise ValueError("No flies have any fully valid windows for the current timesteps/rollout_steps")

        print(
            f"Dataset built from {len(self.fly_indices)} flies "
            f"with at least one fully valid window"
        )

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        # Sample a fly, then sample one of its valid windows
        fly_slot = np.random.randint(0, len(self.fly_indices))
        fly_idx = self.fly_indices[fly_slot]
        start = int(np.random.choice(self.valid_starts_per_fly[fly_slot]))

        x = self.fly_data[fly_idx, start:start + self.timesteps]   # (timesteps, 2)

        # Target = next K true positions
        future_positions = self.fly_data[
            fly_idx,
            start + self.timesteps : start + self.timesteps + self.rollout_steps
        ]  # (rollout_steps, 2)

        return torch.from_numpy(x), torch.from_numpy(future_positions)

train_ds = RandomWindowDataset(
    fly_data=fly_data_scaled,
    valid_frame_mask=valid_frame_mask,
    fly_indices=train_fly_indices,
    timesteps=timesteps,
    rollout_steps=rollout_steps,
    samples_per_epoch=train_samples_per_epoch,
)

val_ds = RandomWindowDataset(
    fly_data=fly_data_scaled,
    valid_frame_mask=valid_frame_mask,
    fly_indices=val_fly_indices,
    timesteps=timesteps,
    rollout_steps=rollout_steps,
    samples_per_epoch=val_samples_per_epoch,
)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
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
        # x: (batch, seq_len, 2)
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]   # last timestep
        pred = self.fc(last_hidden)   # (batch, 2)
        return pred

model = XYLSTM(input_size=2, hidden_size=hidden_size, num_layers=num_layers).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.HuberLoss(delta=huber_delta)


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
        # Multi-step autoregressive rollout loss
        current_window = xb.clone()
        loss = 0.0

        for k in range(rollout_steps):
            pred_delta = model(current_window)                      # (batch, 2)
            pred_pos = current_window[:, -1, :] + pred_delta       # reconstructed next position
            true_pos = yb[:, k, :]                                  # true next position at rollout step k

            loss = loss + criterion(pred_pos, true_pos)

            # Roll window forward using the predicted position
            current_window = torch.cat(
                [current_window[:, 1:, :], pred_pos.unsqueeze(1)],
                dim=1
            )

        loss = loss / rollout_steps

        loss.backward()
        optimizer.step()

        running_train += loss.item() * xb.size(0)
        train_bar.set_postfix(loss=f"{loss.item():.6f}")

    train_loss = running_train / len(train_ds)
    train_losses.append(train_loss)

    # Validation
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

            current_window = xb.clone()
            loss = 0.0

            for k in range(rollout_steps):
                pred_delta = model(current_window)                      # (batch, 2)
                pred_pos = current_window[:, -1, :] + pred_delta
                true_pos = yb[:, k, :]

                loss = loss + criterion(pred_pos, true_pos)

                current_window = torch.cat(
                    [current_window[:, 1:, :], pred_pos.unsqueeze(1)],
                    dim=1
                )

            loss = loss / rollout_steps
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
plt.ylabel("Huber rollout loss")
plt.legend()
plt.title(f"LSTM training and validation loss (K={rollout_steps} rollout)")
plt.tight_layout()
plt.savefig(loss_plot_path, dpi=300)
plt.close()
print(f"Saved loss plot to: {loss_plot_path}")

# Quick one-step prediction example from a held-out validation fly
model.eval()

fly_slot = np.random.randint(0, len(val_ds.fly_indices))
example_fly = val_ds.fly_indices[fly_slot]
example_start = int(np.random.choice(val_ds.valid_starts_per_fly[fly_slot]))

test_seq_scaled = fly_data_scaled[example_fly, example_start:example_start + timesteps]
last_input_scaled = test_seq_scaled[-1]
true_next_scaled = fly_data_scaled[example_fly, example_start + timesteps]

test_seq = torch.tensor(test_seq_scaled[None, ...], dtype=torch.float32, device=device)

with torch.no_grad():
    pred_delta_scaled = model(test_seq).cpu().numpy()[0]

pred_next_scaled = last_input_scaled + pred_delta_scaled

test_seq_unscaled = scaler.inverse_transform(test_seq_scaled)
true_next_unscaled = scaler.inverse_transform(true_next_scaled.reshape(1, -1))[0]
pred_next_unscaled = scaler.inverse_transform(pred_next_scaled.reshape(1, -1))[0]

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

# Seed rollout from a held-out validation fly
seed_fly = val_ds.fly_indices[0]
seed_start = int(val_ds.valid_starts_per_fly[0][0])

seed = fly_data_scaled[seed_fly, seed_start:seed_start + timesteps].copy()
moving_window = torch.tensor(seed, dtype=torch.float32, device=device)

simulated = []

model.eval()
with torch.no_grad():
    rollout_bar = tqdm(range(generate_steps), desc="Rollout", dynamic_ncols=True)
    for _ in rollout_bar:
        pred_delta = model(moving_window.unsqueeze(0)).cpu().numpy()[0]

        last_pos = moving_window[-1].detach().cpu().numpy()
        next_pos = last_pos + pred_delta

        simulated.append(next_pos)

        # Slide window and append reconstructed next position
        moving_window = torch.roll(moving_window, shifts=-1, dims=0)
        moving_window[-1] = torch.tensor(next_pos, dtype=torch.float32, device=device)

simulated = np.array(simulated, dtype=np.float32)
simulated_unscaled = scaler.inverse_transform(simulated)
seed_unscaled = scaler.inverse_transform(seed)

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