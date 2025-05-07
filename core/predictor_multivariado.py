# core/predictor_02.py
import numpy as np


def predict_next_day(model, last_sequence, scaler):
    """
    Faz a previsão do próximo dia com base na última sequência multivariada (60 x N).
    """
    sequence_scaled = scaler.transform(last_sequence)
    X_input = sequence_scaled[-60:].reshape(1, 60, sequence_scaled.shape[1])
    prediction = model.predict(X_input)
    predicted_scaled = np.zeros((1, scaler.n_features_in_))
    predicted_scaled[0, 0] = prediction[0][0]  # previsão apenas para 'Fechamento'
    return scaler.inverse_transform(predicted_scaled)[0][0]  # retorna o valor real do fechamento


def prepare_multivariate_sequence(df, scaler, feature_cols, sequence_length=60):
    """
    Prepara os últimos 'sequence_length' dados com múltiplas features.
    """
    data = df[feature_cols].values[-sequence_length:]
    return data
