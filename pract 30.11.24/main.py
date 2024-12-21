import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

T = 1000  
tau = 10  

# Временной ряд: синусоида + шум
time = np.arange(1, T + 1, dtype=np.float32)
x = np.sin(0.01 * time) + np.random.normal(0, 0.2, (T,))

features = np.zeros((T - tau, tau))
for i in range(tau):
    features[:, i] = x[i: T - tau + i]

labels = x[tau:]  # (T - tau,)

features_tensor = torch.tensor(features, dtype=torch.float32)  # (T - tau, tau)
labels_tensor = torch.tensor(labels, dtype=torch.float32).view(-1, 1)  # (T - tau, 1)

print(features_tensor.shape)  # Ожидаемый размер: (T - tau, tau)
print(labels_tensor.shape)    # Ожидаемый размер: (T - tau, 1)

dataset = TensorDataset(features_tensor, labels_tensor)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

class Net(nn.Module):
    def __init__(self, tau):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(tau, 64)  # tau признаков -> 64 нейрона
        self.fc2 = nn.Linear(64, 1)    # 64 нейрона -> 1 выход

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

net = Net(tau)

# Функция обучения
def train(net, train_loader, loss_fn, epochs, lr):
    optimizer = optim.Adam(net.parameters(), lr=lr)
    net.train()
    history = []

    for epoch in range(epochs):
        running_loss = 0.0
        for X, y in train_loader:
            optimizer.zero_grad()
            outputs = net(X)
            loss = loss_fn(outputs, y)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
        
        history.append(running_loss / len(train_loader))
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}")
    
    return history

loss_fn = nn.MSELoss()
history = train(net, train_loader, loss_fn, epochs=1000, lr=0.01)

plt.figure(figsize=(10, 5))
plt.plot(range(1, len(history) + 1), history)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.show()

net.eval()
with torch.no_grad():
    onestep_preds = net(features_tensor).detach().numpy()

plt.figure(figsize=(12, 6))
plt.plot(time[tau:], x[tau:], label='Actual Data')
plt.plot(time[tau:], onestep_preds, label='1-Step Prediction', linestyle='--')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.title('1-Step Predictions vs Actual Data')
plt.show()