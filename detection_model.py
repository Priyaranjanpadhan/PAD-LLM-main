import torch.nn as nn
import torch.optim as optim
import torch
class DetectionModel(nn.Module):
    def __init__(self):
        super(DetectionModel, self).__init__()
        self.fc = nn.Sequential(
            # nn.Linear(1, 256),
            # nn.LazyLinear(256): Automatically detects whether input features are 64D (FIX 4) or 1D (old baseline) so it never crashes!
            nn.LazyLinear(256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.fc(x)

def _get_best_device():
    # Selects Apple Silicon GPU(mps) if available, otherwise CUDA or CPU.
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def train_detection_model(model, X_train, y_train, epochs=100, patience=100, min_delta=1e-4):

    device = _get_best_device()
    model.to(device)
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    pos_weight = len(y_train) / (2 * y_train.sum())
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.clone().detach().float())
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()
    best_loss = float('inf')
    patience_counter = 0
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        # loss.backward(retain_graph=True) # not needed for standard backward pass and wastes memory
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")
        if loss.item() < best_loss - min_delta:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            break

def test_detection_model(model, X_test, y_test,threshold):
    device = _get_best_device()
    model.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)
    model.eval()
    with torch.no_grad():
        logits = model(X_test)
        predictions = torch.sigmoid(logits)
        predictions_rounded = (predictions > threshold).float()
        TP = ((predictions_rounded == 1) & (y_test.view(-1, 1) == 1)).sum().item()
        FN = ((predictions_rounded == 0) & (y_test.view(-1, 1) == 1)).sum().item()
        Recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        FP = ((predictions_rounded == 1) & (y_test.view(-1, 1) == 0)).sum().item()
        Precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        F_measure = (2 * Precision * Recall) / (Precision + Recall) if (Precision + Recall) > 0 else 0
        print(f"Recall: {Recall * 100:.2f}%")
        print(f"Precision: {Precision * 100:.2f}%")
        print(f"F-Measure (F1 Score): {F_measure * 100:.2f}%")
