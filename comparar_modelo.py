import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cloudpickle
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error
from tensorflow.keras.models import load_model
from core.data_preprocessing_multivariado import merge_and_clean_csv, preprocess_data, split_data

MODEL_DIR = "models"
RESULTS_DIR = "results"
FEATURE_COLS = ['Fechamento', 'Retorno_%', 'MM9', 'RSI']

os.makedirs(RESULTS_DIR, exist_ok=True)

def salvar_metricas_completas(papel, modelo, rmse, mae, mape, r2, medae, dir_acc, caminho_csv="results/metricas_modelos.csv"):
    nova_linha = pd.DataFrame({
        "Papel": [papel],
        "Modelo": [modelo],
        "RMSE": [rmse],
        "MAE": [mae],
        "MAPE (%)": [mape],
        "R2": [r2],
        "MedAE": [medae],
        "Directional_Acc (%)": [dir_acc],
        "DataHora": [pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")]
    })

    if os.path.exists(caminho_csv):
        df_existente = pd.read_csv(caminho_csv)
        # Remove entradas antigas desse papel+modelo
        df_existente = df_existente[~((df_existente["Papel"] == papel) & (df_existente["Modelo"] == modelo))]
        df_final = pd.concat([df_existente, nova_linha], ignore_index=True)
    else:
        df_final = nova_linha

    df_final.to_csv(caminho_csv, index=False)
    return caminho_csv

def comparar_modelo(papel: str, csv_path: str, tipo_modelo: str = 'lstm'):
    papel = papel.upper()
    model_prefix = tipo_modelo.lower()  # 'lstm' ou 'gru'

    # Carregar modelo e scaler
    model_path = os.path.join(MODEL_DIR, f'model_{model_prefix}_multivariado_{papel}.keras')
    scaler_path = os.path.join(MODEL_DIR, f'scaler_{model_prefix}_multivariado_{papel}.pkl')

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError(f"❌ Modelo '{papel}' ({tipo_modelo.upper()}) não encontrado. Treine-o antes.")

    model = load_model(model_path)
    with open(scaler_path, 'rb') as f:
        scaler = cloudpickle.load(f)

    # Carregar e preparar dados
    df = merge_and_clean_csv([csv_path])
    _, X, y = preprocess_data(df, feature_cols=FEATURE_COLS, sequence_length=60)
    _, X_test, _, y_test = split_data(X, y, test_size=0.2)

    # Previsão e métricas
    y_pred_scaled = model.predict(X_test)
    y_test_true = scaler.inverse_transform(np.hstack([y_test, np.zeros((len(y_test), scaler.n_features_in_ - 1))]))[:, 0]
    y_pred_true = scaler.inverse_transform(np.hstack([y_pred_scaled, np.zeros((len(y_pred_scaled), scaler.n_features_in_ - 1))]))[:, 0]

    # Cálculo das métricas
    rmse = mean_squared_error(y_test_true, y_pred_true, squared=False)
    mae = mean_absolute_error(y_test_true, y_pred_true)
    mape = np.mean(np.abs((y_test_true - y_pred_true) / y_test_true)) * 100
    r2 = r2_score(y_test_true, y_pred_true)
    medae = median_absolute_error(y_test_true, y_pred_true)
    dir_real = np.sign(np.diff(y_test_true))
    dir_pred = np.sign(np.diff(y_pred_true))
    dir_acc = (dir_real == dir_pred).mean() * 100

    # Salvar as métricas completas em CSV
    salvar_metricas_completas(
        papel=papel,
        modelo=model_prefix,
        rmse=rmse,
        mae=mae,
        mape=mape,
        r2=r2,
        medae=medae,
        dir_acc=dir_acc
    )

    # Plot
    datas = df['Data'].iloc[-len(y_test_true):].reset_index(drop=True)
    datas = pd.to_datetime(datas).dt.to_pydatetime()

    plot_path = os.path.join(RESULTS_DIR, f'comparativo_teste_multivariado_{model_prefix}_{papel}.png')
    plt.figure(figsize=(12, 5))
    plt.plot(datas, y_test_true, label='Real')
    plt.plot(datas, y_pred_true, label='Previsto')
    plt.title(f'Comparação Real x Previsto - {papel} ({tipo_modelo.upper()})')
    plt.xlabel('Data')
    plt.ylabel('Preço (R$)')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    # Exibir métricas completas no app
    texto_metricas = (
        f"📏 RMSE: {rmse:.4f}\n"
        f"📏 MAE: {mae:.4f}\n"
        f"📏 MAPE: {mape:.2f}%\n"
        f"📏 MedAE: {medae:.4f}\n"
        f"📏 R²: {r2:.4f}\n"
        f"📏 Directional Accuracy: {dir_acc:.2f}%\n"
    )

    return texto_metricas, plot_path
