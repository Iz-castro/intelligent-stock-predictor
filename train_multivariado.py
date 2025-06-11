# train_multivariado.py

import os
import pandas as pd
import numpy as np
import cloudpickle
import matplotlib.pyplot as plt
from tqdm.keras import TqdmCallback

from core.data_preprocessing_multivariado import preprocess_data
from core.model_lstm_multivariado import build_lstm_model
from core.model_gru_multivariado import build_gru_model  # Novo import
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import mean_squared_error

MODEL_DIR = "models"
RESULTS_DIR = "results"
SEQUENCE_LENGTH = 60
FEATURE_COLS = ['Fechamento', 'Retorno_%', 'MM9', 'RSI']

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def treinar_modelo(papel: str, df: pd.DataFrame, tipo_modelo: str = 'lstm'):
    """
    Treina o modelo multivariado (LSTM ou GRU) para o papel informado.

    Args:
        papel (str): Nome do papel (ativo).
        df (pd.DataFrame): DataFrame com os dados já pré-processados.
        tipo_modelo (str): 'lstm' ou 'gru'.

    Returns:
        tuple: Mensagem de sucesso e caminho do gráfico.
    """
    papel = papel.upper()

    if not isinstance(df, pd.DataFrame):
        raise TypeError("❌ Entrada inválida: esperado um DataFrame.")

    if len(df) < 100:
        raise ValueError("❌ Poucos dados. É necessário pelo menos 100 linhas após o pré-processamento.")

    # Pré-processar dados (gera X, y e o scaler)
    scaler, X, y = preprocess_data(df, feature_cols=FEATURE_COLS, sequence_length=SEQUENCE_LENGTH)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Construir e treinar modelo conforme escolha
    if tipo_modelo == 'gru':
        model = build_gru_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        model_prefix = 'gru'
    else:
        model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        model_prefix = 'lstm'

    model_path = os.path.join(MODEL_DIR, f"model_{model_prefix}_multivariado_{papel}.keras")
    checkpoint_cb = ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss', mode='min', verbose=1)
    earlystop_cb = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
    reduce_lr_cb = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=32,
        callbacks=[checkpoint_cb, earlystop_cb, reduce_lr_cb, TqdmCallback(verbose=1)],
        verbose=0
    )

    # Recarregar o melhor modelo salvo
    from tensorflow.keras.models import load_model
    model = load_model(model_path)

    # Salvar scaler
    scaler_path = os.path.join(MODEL_DIR, f"scaler_{model_prefix}_multivariado_{papel}.pkl")
    with open(scaler_path, "wb") as f:
        cloudpickle.dump(scaler, f)

    # Gerar gráfico de treino
    grafico_path = os.path.join(RESULTS_DIR, f"treinamento_multivariado_{model_prefix}_{papel}.png")
    plt.figure(figsize=(10, 4))
    plt.plot(history.history['loss'], label='Treinamento')
    plt.plot(history.history['val_loss'], label='Validação')
    plt.title(f'Erro durante o Treinamento - {papel} ({tipo_modelo.upper()})')
    plt.xlabel('Épocas')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(grafico_path)
    plt.close()

    # Métricas finais
    y_pred = model.predict(X_test)
    y_true = y_test.values.flatten()
    rmse = mean_squared_error(y_true, y_pred.flatten(), squared=False)
    mse_final = history.history['val_loss'][-1]
    metricas = f"MSE Validação: {mse_final:.5f} | RMSE: {rmse:.5f}"

    return (f"✅ Modelo {tipo_modelo.upper()} para {papel} treinado com sucesso!", metricas), grafico_path
