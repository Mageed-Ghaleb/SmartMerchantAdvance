
import numpy as np
import pandas as pd

def generate_synthetic_data(num_merchants=100, months=24):
    data = []
    for merchant_id in range(num_merchants):
        base_volume = np.random.uniform(5000, 50000)
        seasonality = np.sin(np.linspace(0, 3*np.pi, months)) * base_volume * 0.2
        noise = np.random.normal(0, base_volume * 0.05, months)
        volumes = base_volume + seasonality + noise
        data.append(pd.DataFrame({
            'merchant_id': merchant_id,
            'month': range(1, months + 1),
            'volume': volumes
        }))
    return pd.concat(data, ignore_index=True)

if __name__ == "__main__":
    df = generate_synthetic_data()
    df.to_csv("../data/synthetic_merchant_volumes.csv", index=False)
    print("Synthetic data generated and saved to data/")
