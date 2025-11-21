import math
from typing import List, Optional, Tuple
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


class Chomp1d(nn.Module):
    """Removes padding on the right to preserve causality.

    In a temporal conv net we often use padding to keep the output length
    the same as the input length. For causal convolutions, we pad only on
    the left and then chomp off extra elements on the right.
    """

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, length + chomp_size)
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """A single residual block in the Temporal Convolutional Network.

    Each block has:
      - dilated causal Conv1d
      - weight normalization
      - activation + dropout
      - second Conv1d + activation + dropout
      - residual connection
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.utils.weight_norm(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.utils.weight_norm(
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # If in_channels != out_channels, use 1x1 conv to match dimensions
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )
        self.relu = nn.ReLU()

        self.init_weights()

    def init_weights(self) -> None:
        for m in [self.conv1, self.conv2]:
            nn.init.kaiming_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        if isinstance(self.downsample, nn.Conv1d):
            nn.init.kaiming_normal_(self.downsample.weight.data)
            if self.downsample.bias is not None:
                self.downsample.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """Stack of TemporalBlocks with exponentially increasing dilation.

    Args:
        num_inputs: number of input channels.
        num_channels: list with the number of channels in each hidden layer.
        kernel_size: convolution kernel size.
        dropout: dropout rate.
    """

    def __init__(
        self,
        num_inputs: int,
        num_channels: List[int],
        kernel_size: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            in_ch = num_inputs if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            dilation = 2 ** i
            layers.append(
                TemporalBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: tensor of shape (batch, channels, seq_len).

        Returns:
            tensor of shape (batch, num_channels[-1], seq_len).
        """
        return self.network(x)


class TCNForecaster(nn.Module):
    """Temporal Convolutional Network for sequence forecasting.

    This model:
      * encodes a history window with a TCN encoder
      * takes the last time step's hidden state
      * projects it to the forecast horizon with an MLP head.

    Shapes:
      - Input:  (batch, history_length, input_dim)
      - Output: (batch, horizon, target_dim)

    For univariate forecasting, input_dim = target_dim = 1.
    """

    def __init__(
        self,
        history_length: int,
        horizon: int,
        input_dim: int = 1,
        target_dim: int = 1,
        encoder_hidden_size: int = 128,
        encoder_layers: int = 3,
        kernel_size: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()

        # TCN expects (batch, channels, seq_len)
        num_channels = [encoder_hidden_size] * encoder_layers
        self.tcn = TemporalConvNet(
            num_inputs=input_dim,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dropout=dropout,
        )

        self.history_length = history_length
        self.horizon = horizon
        self.input_dim = input_dim
        self.target_dim = target_dim

        self.head = nn.Sequential(
            nn.Linear(num_channels[-1], encoder_hidden_size),
            nn.ReLU(),
            nn.Linear(encoder_hidden_size, horizon * target_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: tensor of shape (batch, history_length, input_dim)

        Returns:
            y_hat: tensor of shape (batch, horizon, target_dim)
        """
        # Rearrange to (batch, channels=input_dim, seq_len=history_length)
        x = x.permute(0, 2, 1)

        y_tcn = self.tcn(x)  # (batch, hidden, seq_len)
        # Use representation at last time step
        last_hidden = y_tcn[:, :, -1]  # (batch, hidden)

        out = self.head(last_hidden)  # (batch, horizon * target_dim)
        out = out.view(-1, self.horizon, self.target_dim)
        return out


class WindowedTimeSeriesDataset(Dataset):
    """Sliding-window dataset for supervised time-series forecasting.

    Given a 1D series y[0...T-1], it builds samples:
      - input:  y[i : i + history_length]
      - target: y[i + history_length : i + history_length + horizon]

    Args:
        series: 1D array-like of floats.
        history_length: number of past points used as input.
        horizon: number of future points to predict.
        stride: step between consecutive windows.
    """

    def __init__(
        self,
        series: np.ndarray,
        history_length: int,
        horizon: int,
        stride: int = 1,
    ):
        super().__init__()
        series = np.asarray(series, dtype=np.float32)
        self.series = series
        self.history_length = history_length
        self.horizon = horizon
        self.stride = stride

        if series.ndim != 1:
            raise ValueError("`series` must be 1D (univariate).")

        max_start = len(series) - history_length - horizon
        if max_start < 0:
            raise ValueError(
                "Series too short for the given history_length and horizon."
            )
        self.indices = np.arange(0, max_start + 1, stride, dtype=int)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        i = self.indices[idx]
        x = self.series[i : i + self.history_length]
        y = self.series[i + self.history_length : i + self.history_length + self.horizon]

        # Add feature dimension = 1
        x = torch.from_numpy(x).view(-1, 1)  # (history_length, 1)
        y = torch.from_numpy(y).view(-1, 1)  # (horizon, 1)
        return x, y


class MultiWindowedTimeSeriesDataset(Dataset):
    """Sliding-window dataset for *multivariate* time-series forecasting.

    series: 2D array (time, features)

    Returns
    -------
    x : (history_length, num_features)
    y : (horizon,        num_features)
    """

    def __init__(
        self,
        series: np.ndarray,
        history_length: int,
        horizon: int,
        stride: int = 1,
    ):
        super().__init__()
        series = np.asarray(series, dtype=np.float32)
        if series.ndim != 2:
            raise ValueError("`series` must be 2D: (time, features).")

        self.series = series
        self.history_length = history_length
        self.horizon = horizon
        self.stride = stride

        max_start = len(series) - history_length - horizon
        if max_start < 0:
            raise ValueError(
                "Series too short for the given history_length and horizon."
            )
        self.indices = np.arange(0, max_start + 1, stride, dtype=int)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        i = self.indices[idx]
        x = self.series[i : i + self.history_length, :]
        y = self.series[i + self.history_length : i + self.history_length + self.horizon, :]
        return torch.from_numpy(x), torch.from_numpy(y)



def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    criterion: nn.Module,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        y_hat = model(x)
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()

        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        n += batch_size

    return total_loss / n


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> float:
    model.eval()
    total_loss = 0.0
    n = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        y_hat = model(x)
        loss = criterion(y_hat, y)

        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        n += batch_size

    return total_loss / n


def fit_tcn(
    series: np.ndarray,
    history_length: int,
    horizon: int,
    batch_size: int = 32,
    num_epochs: int = 50,
    learning_rate: float = 1e-3,
    val_ratio: float = 0.2,
    encoder_hidden_size: int = 128,
    encoder_layers: int = 3,
    kernel_size: int = 2,
    dropout: float = 0.2,
    seed: int = 1,
) -> Tuple[TCNForecaster, dict]:
    """Convenience function to train a TCNForecaster on a univariate series.

    Splits the series into train/validation at the end, builds windowed
    datasets, trains the model, and returns the fitted model plus a dict
    with training history.

    Returns:
        model, history where history has keys:
            - 'train_loss'
            - 'val_loss'
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train/val split on the raw series
    n = len(series)
    if n <= history_length + horizon:
        raise ValueError("Series is too short for the given history_length and horizon.")

    split_idx = int(n * (1 - val_ratio))
    train_series = series[:split_idx]
    # Extend validation slice backward so it has enough history for windows
    val_series = series[split_idx - history_length - horizon :]

    train_ds = WindowedTimeSeriesDataset(train_series, history_length, horizon)
    val_ds = WindowedTimeSeriesDataset(val_series, history_length, horizon)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = TCNForecaster(
        history_length=history_length,
        horizon=horizon,
        input_dim=1,
        target_dim=1,
        encoder_hidden_size=encoder_hidden_size,
        encoder_layers=encoder_layers,
        kernel_size=kernel_size,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    train_losses = []
    val_losses = []

    best_val = math.inf
    best_state = None

    for epoch in range(1, num_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, criterion)
        val_loss = evaluate_epoch(model, val_loader, device, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.5f} | val_loss={val_loss:.5f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    history = {"train_loss": train_losses, "val_loss": val_losses}
    return model, history


def fit_tcn_multivariate(
    series_2d: np.ndarray,
    history_length: int,
    horizon: int,
    batch_size: int = 32,
    num_epochs: int = 50,
    learning_rate: float = 1e-3,
    val_ratio: float = 0.2,
    encoder_hidden_size: int = 128,
    encoder_layers: int = 3,
    kernel_size: int = 2,
    dropout: float = 0.2,
    seed: int = 1,
):
    """Train a TCNForecaster on a *multivariate* series.

    series_2d: 2D array (time, features)

    The model uses all features as inputs and predicts all of them jointly.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    series_2d = np.asarray(series_2d, dtype=np.float32)
    if series_2d.ndim != 2:
        raise ValueError("`series_2d` must be 2D: (time, features).")

    T, D = series_2d.shape
    if T <= history_length + horizon:
        raise ValueError("Series is too short for the given history_length and horizon.")

    # train/val split on the time axis
    split_idx = int(T * (1 - val_ratio))
    train_series = series_2d[:split_idx, :]
    # extend val slice backward to have enough history for windows
    val_series = series_2d[split_idx - history_length - horizon :, :]

    train_ds = MultiWindowedTimeSeriesDataset(train_series, history_length, horizon)
    val_ds = MultiWindowedTimeSeriesDataset(val_series, history_length, horizon)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = TCNForecaster(
        history_length=history_length,
        horizon=horizon,
        input_dim=D,
        target_dim=D,  # predict all features
        encoder_hidden_size=encoder_hidden_size,
        encoder_layers=encoder_layers,
        kernel_size=kernel_size,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    train_losses = []
    val_losses = []
    best_val = math.inf
    best_state = None

    for epoch in range(1, num_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, criterion)
        val_loss = evaluate_epoch(model, val_loader, device, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"[Multi] Epoch {epoch:03d} | train_loss={train_loss:.5f} "
            f"| val_loss={val_loss:.5f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    history = {"train_loss": train_losses, "val_loss": val_losses}
    return model, history


def forecast(
    model: TCNForecaster,
    history: np.ndarray,
) -> np.ndarray:
    """Generate a forecast from a fitted model and a history window.

    Args:
        model: trained TCNForecaster.
        history: 1D array-like of length == model.history_length.

    Returns:
        forecast array of shape (horizon,)
    """
    model.eval()
    device = next(model.parameters()).device

    history = np.asarray(history, dtype=np.float32)
    if history.ndim != 1:
        raise ValueError("`history` must be 1D.")
    if len(history) != model.history_length:
        raise ValueError(
            f"`history` must have length {model.history_length}, "
            f"got {len(history)}."
        )

    x = torch.from_numpy(history).view(1, -1, 1)  # (1, history_length, 1)
    x = x.to(device)

    with torch.no_grad():
        y_hat = model(x)  # (1, horizon, 1)
    return y_hat.cpu().numpy().reshape(-1)


def forecast_multivariate(
    model: TCNForecaster,
    history_2d: np.ndarray,
) -> np.ndarray:
    """Generate a forecast for a *multivariate* series.

    history_2d: 2D array of shape (history_length, input_dim)
    Returns: array of shape (horizon, target_dim)
    """
    model.eval()
    device = next(model.parameters()).device

    history_2d = np.asarray(history_2d, dtype=np.float32)
    if history_2d.ndim != 2:
        raise ValueError("`history_2d` must be 2D (time, features).")
    if history_2d.shape[0] != model.history_length:
        raise ValueError(
            f"`history_2d` must have length {model.history_length}, "
            f"got {history_2d.shape[0]}."
        )

    # shape: (1, history_length, input_dim)
    x = torch.from_numpy(history_2d).unsqueeze(0)
    x = x.to(device)

    with torch.no_grad():
        y_hat = model(x)  # (1, horizon, target_dim)

    # return (horizon, target_dim)
    return y_hat.cpu().numpy()[0]


if __name__ == "__main__":
    # Example usage on a dummy sine-wave series.
    timesteps = np.linspace(0, 100, 1000, dtype=np.float32)
    series = np.sin(0.2 * timesteps) + 0.1 * np.random.randn(len(timesteps)).astype(
        np.float32
    )

    HISTORY = 48
    HORIZON = 12

    model, history = fit_tcn(
        series,
        history_length=HISTORY,
        horizon=HORIZON,
        batch_size=32,
        num_epochs=10,
        learning_rate=1e-3,
        encoder_hidden_size=128,
        encoder_layers=3,
        kernel_size=3,
        dropout=0.1,
    )

    # Take the last HISTORY points as context and forecast the next HORIZON points.
    ctx = series[-HISTORY:]
    y_forecast = forecast(model, ctx)

    print("Last observed values:", ctx[-5:])
    print("First 5-step forecast:", y_forecast[:5])
