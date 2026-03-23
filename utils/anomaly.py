import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

def autoencoder_anomaly(df):

    data = df[["energy"]].values

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    # Define Autoencoder
    input_dim = data_scaled.shape[1]

    input_layer = Input(shape=(input_dim,))
    encoded = Dense(8, activation="relu")(input_layer)
    encoded = Dense(4, activation="relu")(encoded)

    decoded = Dense(8, activation="relu")(encoded)
    decoded = Dense(input_dim, activation="linear")(decoded)

    autoencoder = Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer="adam", loss="mse")

    # Train
    autoencoder.fit(data_scaled, data_scaled,
                    epochs=20,
                    batch_size=32,
                    verbose=0)

    # Reconstruction
    reconstructed = autoencoder.predict(data_scaled)
    mse = np.mean(np.power(data_scaled - reconstructed, 2), axis=1)

    threshold = np.percentile(mse, 95)

    df["anomaly"] = mse > threshold
    df["reconstruction_error"] = mse

    return df