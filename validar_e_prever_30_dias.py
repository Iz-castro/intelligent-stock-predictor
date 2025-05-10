# validar_e_prever_30_dias.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cloudpickle
from datetime import timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import load_model
from core.data_preprocessing_multivariado import merge_and_clean_csv, preprocess_data, split_data
from core.predictor_multivariado import prepare_multivariate_sequence, predict_next_day

MODEL_DIR = "models"
RESULTS_DIR = "results"
FEATURE_COLS = ['Fechamento', 'Retorno_%', 'MM9', 'RSI']

os.makedirs(RESULTS_DIR, exist_ok=True)

def validar_e_prever_30_dias(papel: str, csv_path: str):
    papel = papel.upper()

    model_path = os.path.join(MODEL_DIR, f'model_lstm_multivariado_{papel}.keras')
    scaler_path = os.path.join(MODEL_DIR, f'scaler_lstm_multivariado_{papel}.pkl')

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError(f"❌ Modelo '{papel}' não encontrado. Treine-o antes.")

    model = load_model(model_path)
    with open(scaler_path, 'rb') as f:
        scaler = cloudpickle.load(f)

    df = merge_and_clean_csv([csv_path])
    _, X, y = preprocess_data(df, feature_cols=FEATURE_COLS, sequence_length=60)
    _, X_test, _, y_test = split_data(X, y, test_size=0.2)

    y_pred_scaled = model.predict(X_test)
    y_test_true = scaler.inverse_transform(np.hstack([y_test, np.zeros((len(y_test), scaler.n_features_in_ - 1))]))[:, 0]
    y_pred_true = scaler.inverse_transform(np.hstack([y_pred_scaled, np.zeros((len(y_pred_scaled), scaler.n_features_in_ - 1))]))[:, 0]

    ultimos_30_true = y_test_true[-30:]
    ultimos_30_pred = y_pred_true[-30:]

    # ✅ CORREÇÃO AQUI:
    datas_validacao = pd.to_datetime(
        df['Data'].iloc[-len(y_test_true):].reset_index(drop=True)[-30:]
    )

    # Previsão futura
    entrada = prepare_multivariate_sequence(df, scaler, FEATURE_COLS, sequence_length=60)
    futuro = []
    limite_min = entrada[-1, 0] * 0.8
    limite_max = entrada[-1, 0] * 1.2

    for _ in range(6):
        for _ in range(5):
            valor = predict_next_day(model, entrada, scaler)
            valor_clipped = max(min(valor, limite_max), limite_min)
            nova_linha = entrada[-1].copy()
            nova_linha[0] = valor_clipped
            entrada = np.vstack([entrada[1:], nova_linha])
            futuro.append(valor_clipped)

    ultima_data = pd.to_datetime(df['Data'].iloc[-1])
    datas_futuras = []
    contador = 1
    while len(datas_futuras) < 30:
        proxima = ultima_data + timedelta(days=contador)
        if proxima.weekday() < 5:
            datas_futuras.append(proxima)
        contador += 1

    # Plot
    plot_path = os.path.join(RESULTS_DIR, f'validacao_e_previsao_30_dias_{papel}.png')
    plt.figure(figsize=(12, 5))
    plt.plot(datas_validacao, ultimos_30_true, label='Real (últimos 30)')
    plt.plot(datas_validacao, ultimos_30_pred, label='Previsto (últimos 30)')
    plt.plot(datas_futuras, futuro, label='Previsto (futuros)', linestyle='--')
    plt.title(f'Validação + Previsão 30 dias - {papel}')
    plt.xlabel('Data')
    plt.ylabel('Preço (R$)')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    return f"✅ Previsão para os próximos 30 dias gerada com sucesso!", plot_path
