import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np

from src.data_loader import load_dataset
from src.model import EllipseNet
from src import config

def train():
    # הגדר seed לשחזוריות
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)

    # טען את דאטא האימון בלבד
    print("🔄 Loading training data...")
    X, y = load_dataset(config.DATA_DIR, split='train')
    X_tensor = torch.tensor(X).permute(0, 3, 1, 2)  # (N, H, W, C) → (N, C, H, W)
    y_tensor = torch.tensor(y)

    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    # הגדר את המודל
    model = EllipseNet()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 Model parameters: {total_params:,}")

    # הפרדה בין סיווג לרגרסיה מתוך הפלט
    def custom_loss(output, target):
        # Classification loss - apply sigmoid here for training
        is_ellipse_pred = torch.sigmoid(output[:, 0])  # Shape: (batch_size,)
        is_ellipse_true = target[:, 0]  # Shape: (batch_size,)

        # Binary Cross Entropy
        loss_cls = nn.BCELoss()(is_ellipse_pred, is_ellipse_true)

        # MSE לרגרסיה רק כשיש אליפסה
        mask = is_ellipse_true > 0.5
        if mask.any():
            loss_reg = nn.MSELoss()(output[mask][:, 1:], target[mask][:, 1:])
        else:
            loss_reg = torch.tensor(0.0, device=output.device)

        return loss_cls + loss_reg

    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    losses = []

    # לולאת אימון
    print(f"🚀 Starting training for {config.EPOCHS} epochs...")
    model.train()
    for epoch in range(config.EPOCHS):
        running_loss = 0.0
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = custom_loss(outputs, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{config.EPOCHS}, Loss: {avg_loss:.4f}")

    # שמור את המודל
    torch.save(model.state_dict(), config.MODEL_PATH)
    print(f"\n✔️ Model saved to {config.MODEL_PATH}")

    # גרף loss
    plt.plot(losses)
    plt.title("Training Loss (Training Set Only)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(config.LOSS_PLOT_PATH)
    print(f"📈 Loss plot saved to {config.LOSS_PLOT_PATH}")
    plt.close()

if __name__ == "__main__":
    train()
