import pandas as pd
import numpy as np

def load_energy_data():
    rng = pd.date_range("2023-01-01", periods=500, freq="H")
    
    df = pd.DataFrame({
        "datetime": rng,
        "energy": 200 + 50*np.sin(np.arange(500)/24) + np.random.normal(0, 10, 500)
    })

    df.set_index("datetime", inplace=True)
    return df