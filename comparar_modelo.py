# comparar_modelo.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cloudpickle
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import load_model
from core.data_preprocessing_multivariado import merge_and_clean_csv, preprocess_data, split_data

MODEL_DIR = "models"
RESULTS_DIR = "results"
FEATURE_COLS = ['Fechamento', 'Retorno_%', 'MM9', 'RSI']

os.makedirs(RESULTS_DIR, exist_ok=True)

def comparar_modelo(papel: str, csv_path: str):
    papel = papel.upper()

    # Carregar modelo e scaler
    model_path = os.path.join(MODEL_DIR, f'model_lstm_multivariado_{papel}.keras')
    scaler_path = os.path.join(MODEL_DIR, f'scaler_lstm_multivariado_{papel}.pkl')

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError(f"❌ Modelo '{papel}' não encontrado. Treine-o antes.")

    model = load_model(model_path)
    with open(scaler_path, 'rb') as f:
        scaler = cloudpickle.load(f)

    # Carregar e preparar dados
    df = merge_and_clean_csv([csv_path])
    _, X, y = preprocess_data(df, feature_cols=FEATURE_COLS, sequence_length=60)
    _, X_test, _, y_test = split_data(X, y, test_size=0.2)

    y_pred_scaled = model.predict(X_test)
    y_test_true = scaler.inverse_transform(np.hstack([y_test, np.zeros((len(y_test), scaler.n_features_in_ - 1))]))[:, 0]
    y_pred_true = scaler.inverse_transform(np.hstack([y_pred_scaled, np.zeros((len(y_pred_scaled), scaler.n_features_in_ - 1))]))[:, 0]

    # Métricas
    rmse = mean_squared_error(y_test_true, y_pred_true, squared=False)
    mae = mean_absolute_error(y_test_true, y_pred_true)

    # Plot
    datas = df['Data'].iloc[-len(y_test_true):].reset_index(drop=True)
    datas = pd.to_datetime(datas).dt.to_pydatetime()

    plot_path = os.path.join(RESULTS_DIR, f'comparativo_teste_multivariado_{papel}.png')
    plt.figure(figsize=(12, 5))
    plt.plot(datas, y_test_true, label='Real')
    plt.plot(datas, y_pred_true, label='Previsto')
    plt.title(f'Comparação Real x Previsto - {papel}')
    plt.xlabel('Data')
    plt.ylabel('Preço (R$)')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    return f"📏 RMSE: {rmse:.4f} | MAE: {mae:.4f}", plot_path
