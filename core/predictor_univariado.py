# core/predictor.py
import numpy as np
import pandas as pd

def predict_next_day(model, last_sequence, scaler):
    sequence_scaled = scaler.transform(last_sequence.reshape(-1, 1))
    X_input = sequence_scaled[-60:].reshape(1, 60, 1)
    prediction = model.predict(X_input)
    return scaler.inverse_transform(prediction)[0][0]

def prepare_input(df, scaler, sequence_length=60):
    scaled_data = scaler.transform(df[['Close']].values)
    return np.array([scaled_data[-sequence_length:].flatten()])
