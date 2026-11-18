import numpy as np
import torch
from torch import nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# 1. pictures + answers, split into practice and hidden test
digits = load_digits()
X = digits.data / 16.0          # 64 numbers per picture, made 0–1
y = digits.target               # the correct answers
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train, X_test = torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32)
y_train, y_test = torch.tensor(y_train), torch.tensor(y_test)

# 2. the machine: 64 → 32 helper judges → 10 final judges
torch.manual_seed(0)
model = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 10))

def accuracy():
    guesses = model(X_test).argmax(dim=1)          # highest score wins
    return (guesses == y_test).float().mean().item()

# 3. before training
print(f"Before training: {accuracy():.0%} right")

# 4. training: guess → check → nudge knobs → repeat
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()
for step in range(200):
    loss = loss_fn(model(X_train), y_train)   # how wrong were we?
    optimizer.zero_grad()
    loss.backward()                           # which way to nudge each knob
    optimizer.step()                          # nudge them

# 5. after training
print(f"After training:  {accuracy():.0%} right")

# 6. save the knobs = our healthy patient
w = {name: p.detach().numpy() for name, p in model.state_dict().items()}
np.savez("src/wsp_env/data/healthy.npz", **w)
print("Saved healthy.npz:", {k: v.shape for k, v in w.items()})
