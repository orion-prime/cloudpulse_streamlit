import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

def detect_anomalies(df):
    features = df[["Cost", "Usage", "Cost_Change", "Rolling_Avg"]]

    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    # Isolation Forest
    iso = IsolationForest(contamination=0.05, random_state=42)
    df["IF_Anomaly"] = iso.fit_predict(X)
    df["IF_Anomaly"] = df["IF_Anomaly"].map({1: 0, -1: 1})

    # Autoencoder
    input_dim = X.shape[1]
    inp = Input(shape=(input_dim,))
    encoded = Dense(8, activation="relu")(inp)
    encoded = Dense(4, activation="relu")(encoded)
    decoded = Dense(8, activation="relu")(encoded)
    decoded = Dense(input_dim, activation="linear")(decoded)

    autoencoder = Model(inp, decoded)
    autoencoder.compile(optimizer="adam", loss="mse")
    autoencoder.fit(X, X, epochs=40, batch_size=16, verbose=0)

    recon = autoencoder.predict(X)
    error = np.mean(np.square(X - recon), axis=1)
    threshold = np.percentile(error, 95)

    df["AE_Anomaly"] = (error > threshold).astype(int)

    df["Final_Anomaly"] = ((df["IF_Anomaly"] == 1) |
                            (df["AE_Anomaly"] == 1)).astype(int)

    return df
