
from generate_data import generate_synthetic_data
from cnn_forecast import train_model
from optimize_loans import optimize_advance
import pandas as pd
import torch
from cnn_forecast import CNNForecast

def main():
    print("Step 1: Generating synthetic data...")
    generate_synthetic_data().to_csv("data/synthetic_merchant_volumes.csv", index=False)

    print("Step 2: Training CNN model...")
    train_model()

    print("Step 3: Forecasting and optimizing...")
    df = pd.read_csv("data/synthetic_merchant_volumes.csv")
    merchant_series = df[df['merchant_id'] == 0]['volume'].values[:12]

    model = CNNForecast()
    model.load_state_dict(torch.load("models/cnn_forecast.pth"))
    model.eval()
    forecast_input = torch.tensor(merchant_series.reshape(1, 1, 12), dtype=torch.float32)
    forecast = model(forecast_input).detach().numpy().flatten().tolist()

    result = optimize_advance(forecast)
    print("Final optimized advance offer:", result)

if __name__ == "__main__":
    main()
