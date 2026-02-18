import torch, torch.nn as nn, numpy as np
from sklearn.model_selection import train_test_split

class TCN(nn.Module):
    def __init__(self, n_features, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_features, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, 3, dilation=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(128, 128, 3, dilation=4, padding=4),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, n_classes)
        )
    def forward(self, x):
        x = x.transpose(1, 2)  # [B, F, T]
        return self.net(x)

model_name = "gesture_tcn.pt"

X, y = np.load("npy/X.npy"), np.load("npy/y.npy")
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)
train_tensor = torch.tensor(X_train, dtype=torch.float32)
val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
y_val_t = torch.tensor(y_val, dtype=torch.long)

model = TCN(63, len(set(y)))
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(30):
    model.train()
    opt.zero_grad()
    out = model(train_tensor)
    loss = loss_fn(out, y_train_t)
    loss.backward()
    opt.step()

    model.eval()
    with torch.no_grad():
        val_out = model(val_tensor)
        acc = (val_out.argmax(1) == y_val_t).float().mean().item()
    print(f"Epoch {epoch}: Loss={loss.item():.3f}, ValAcc={acc:.3f}")

torch.save(model.state_dict(), f"models/{model_name}")