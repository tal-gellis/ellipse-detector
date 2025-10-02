import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

from src.data_loader import load_dataset
from src.model import EllipseNet
from src import config

def evaluate():
    # טען את דאטא הבדיקה בלבד
    print("🔄 Loading test data...")
    X, y = load_dataset(config.DATA_DIR, split='test')
    X_tensor = torch.tensor(X).permute(0, 3, 1, 2)
    y_tensor = torch.tensor(y)

    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE)

    # טען את המודל
    print("📂 Loading trained model...")
    model = EllipseNet()
    model.load_state_dict(torch.load(config.MODEL_PATH))
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    all_preds = []
    all_targets = []

    print("🧪 Evaluating on test set...")
    with torch.no_grad():
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            preds = model(batch_X)
            
            # Apply sigmoid to classification output
            preds[:, 0] = torch.sigmoid(preds[:, 0])
            
            all_preds.append(preds.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Calculate evaluation metrics
    abs_errors = np.abs(all_preds - all_targets)
    avg_error = np.mean(abs_errors, axis=0)

    print("\n📊 Test Set Results:")
    print("=" * 40)
    param_names = ['isEllipse', 'x_norm', 'y_norm', 'major_norm', 'minor_norm', 'angle_norm']
    for i, name in enumerate(param_names):
        print(f"{name:12} avg absolute error: {avg_error[i]:.4f}")

    # Classification accuracy
    cls_preds = (all_preds[:, 0] > 0.5).astype(float)
    cls_targets = all_targets[:, 0]
    accuracy = np.mean(cls_preds == cls_targets)
    print(f"{'Accuracy':12} (classification): {accuracy:.4f}")
    
    # Additional classification metrics
    tp = np.sum((cls_preds == 1) & (cls_targets == 1))
    tn = np.sum((cls_preds == 0) & (cls_targets == 0))
    fp = np.sum((cls_preds == 1) & (cls_targets == 0))
    fn = np.sum((cls_preds == 0) & (cls_targets == 1))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    print(f"{'Precision':12}: {precision:.4f}")
    print(f"{'Recall':12}: {recall:.4f}")

    # גרף שגיאה מוחלטת ממוצעת
    plt.bar(param_names, avg_error)
    plt.title("Test Set: Average Absolute Error per Parameter")
    plt.ylabel("Error")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("outputs/metrics.png")
    print("\n📊 Evaluation graph saved to outputs/metrics.png")
    plt.close()
