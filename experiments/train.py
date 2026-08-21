"""Train the DMA-PULSE Transformer on a local prepared sequence dataset.

Expected NPZ input:
    X: float array shaped [N, 5, 11]
    y: integer labels shaped [N]

The repository does not distribute the PULSE dataset. This script therefore
accepts a local dataset path and does not claim to reproduce the published
benchmark unless the original data and experimental settings are supplied.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dma_pulse.data import validate_representation  # noqa: E402
from dma_pulse.model import PulseTransformer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DMA-PULSE Transformer.")
    parser.add_argument("--data", type=Path, required=True, help="Local .npz dataset")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--classes", type=int, default=2)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = (
        torch.device("cuda")
        if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
        else torch.device("cpu")
    )

    data = np.load(args.data)
    x = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
    validate_representation(x)

    split = int(0.8 * len(x))
    x_train, x_test = x[:split], x[split:]
    y_train, y_test = y[:split], y[split:]

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=args.batch_size,
        shuffle=True,
    )

    model = PulseTransformer(num_classes=args.classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(xb)

        model.eval()
        with torch.no_grad():
            logits = model(torch.from_numpy(x_test).to(device))
            predictions = logits.argmax(dim=1).cpu().numpy()
        accuracy = accuracy_score(y_test, predictions)
        print(
            f"epoch={epoch:03d} "
            f"loss={running_loss / max(len(x_train), 1):.4f} "
            f"test_accuracy={accuracy:.4f}"
        )

    output = ROOT / "outputs"
    output.mkdir(exist_ok=True)
    torch.save(model.state_dict(), output / "transformer.pt")
    print(f"saved={output / 'transformer.pt'}")


if __name__ == "__main__":
    main()
