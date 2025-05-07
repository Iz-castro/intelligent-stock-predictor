# predict_validated_30dias_02.py
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cloudpickle
from datetime import timedelta
from tensorflow.keras.models import load_model
from core.data_preprocessing_multivariado import merge_and_clean_csv, preprocess_data, split_data
from core.predictor_multivariado import predict_next_day, prepare_multivariate_sequence
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Configurações de diretório
RAW_DIR = 'data/raw'
MODEL_DIR = 'models'
RESULTS_DIR = 'results'
FEATURE_COLS = ['Fechamento', 'Retorno_%', 'MM9', 'RSI']

os.makedirs(RESULTS_DIR, exist_ok=True)

# Carregar e preparar dados
csv_files = glob.glob(os.path.join(RAW_DIR, '*.csv'))
df = merge_and_clean_csv(csv_files)
model = load_model(os.path.join(MODEL_DIR, 'model_lstm_multivariado.keras'))

with open(os.path.join(MODEL_DIR, 'scaler_lstm_multivariado.pkl'), 'rb') as f:
    scaler = cloudpickle.load(f)

# Pré-processamento e divisão
scaler_data, X, y = preprocess_data(df, feature_cols=FEATURE_COLS, sequence_length=60)
X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)

# Previsão nos últimos 30 dias do conjunto de teste
y_pred_scaled = model.predict(X_test)
y_test_true = scaler.inverse_transform(np.hstack([y_test, np.zeros((len(y_test), scaler.n_features_in_ - 1))]))[:, 0]
y_pred_true = scaler.inverse_transform(np.hstack([y_pred_scaled, np.zeros((len(y_pred_scaled), scaler.n_features_in_ - 1))]))[:, 0]

ultimos_30_true = y_test_true[-30:]
ultimos_30_pred = y_pred_true[-30:]

# Datas de validação
base_index = df['Data'].iloc[-len(y_test_true):].reset_index(drop=True)
datas_validacao = pd.to_datetime(base_index[-30:]).dt.to_pydatetime()

# Métricas
rmse = mean_squared_error(ultimos_30_true, ultimos_30_pred, squared=False)
mae = mean_absolute_error(ultimos_30_true, ultimos_30_pred)
print(f"📏 RMSE últimos 30 dias: {rmse:.4f}")
print(f"📏 MAE  últimos 30 dias: {mae:.4f}")

# Previsão dos próximos 30 dias (blocos de 5)
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

# Gerar datas futuras úteis
ultima_data = pd.to_datetime(df['Data'].iloc[-1])
datas_futuras = []
contador = 1
while len(datas_futuras) < 30:
    proxima = ultima_data + timedelta(days=contador)
    if proxima.weekday() < 5:
        datas_futuras.append(proxima)
    contador += 1

datas_futuras = pd.to_datetime(datas_futuras).to_pydatetime()

# Salvar previsões futuras
pd.DataFrame({
    'Data': pd.to_datetime(datas_futuras).strftime('%Y-%m-%d'),
    'Fechamento_Previsto': futuro
}).to_csv(os.path.join(RESULTS_DIR, 'previsao_30_dias_futuros_multivariado.csv'), index=False)

# Gráficos
plt.figure(figsize=(12, 5))
plt.plot(datas_validacao, ultimos_30_true, label='Real (últimos 30 dias)')
plt.plot(datas_validacao, ultimos_30_pred, label='Previsto (últimos 30 dias)')
plt.plot(datas_futuras, futuro, label='Previsto (futuros)', linestyle='--')
plt.title('Validação + Previsão (Modelo Multivariado) para os Próximos 30 Dias Úteis')
plt.xlabel('Data')
plt.ylabel('Preço (R$)')
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'validacao_e_previsao_30_dias_multivariado.png'))
plt.close()

print("✅ Validação e previsão multivariada dos próximos 30 dias geradas com sucesso!")
