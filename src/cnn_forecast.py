
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np

class CNNForecast(nn.Module):
    def __init__(self):
        super(CNNForecast, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(12)
        self.fc = nn.Linear(16 * 12, 12)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

def train_model():
    df = pd.read_csv("../data/synthetic_merchant_volumes.csv")
    merchants = df['merchant_id'].unique()
    X, y = [], []

    for m in merchants:
        series = df[df['merchant_id'] == m]['volume'].values
        X.append(series[:12])
        y.append(series[12:24])

    X = np.array(X).reshape(-1, 1, 12)
    y = np.array(y)

    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = CNNForecast()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    for epoch in range(10):
        for batch_X, batch_y in loader:
            pred = model(batch_X)
            loss = loss_fn(pred, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "../models/cnn_forecast.pth")
    print("Model trained and saved.")

if __name__ == "__main__":
    train_model()
