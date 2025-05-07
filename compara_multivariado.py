# compara_02.py
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cloudpickle
from tensorflow.keras.models import load_model
from core.data_preprocessing_multivariado import merge_and_clean_csv, preprocess_data, split_data
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Diretórios
RAW_DIR = 'data/raw'
MODEL_DIR = 'models'
RESULTS_DIR = 'results'
FEATURE_COLS = ['Fechamento', 'Retorno_%', 'MM9', 'RSI']

os.makedirs(RESULTS_DIR, exist_ok=True)

# Carregar modelo e scaler
model = load_model(os.path.join(MODEL_DIR, 'model_lstm_multivariado.keras'))
with open(os.path.join(MODEL_DIR, 'scaler_lstm_multivariado.pkl'), 'rb') as f:
    scaler = cloudpickle.load(f)

# Carregar e preparar os dados
csv_files = glob.glob(os.path.join(RAW_DIR, '*.csv'))
df = merge_and_clean_csv(csv_files)
scaler_data, X, y = preprocess_data(df, feature_cols=FEATURE_COLS, sequence_length=60)
X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)

# Prever sobre X_test
y_pred_scaled = model.predict(X_test)

# Reverter escala apenas da coluna de fechamento
y_test_true = scaler.inverse_transform(np.hstack([y_test, np.zeros((len(y_test), scaler.n_features_in_ - 1))]))[:, 0]
y_pred_true = scaler.inverse_transform(np.hstack([y_pred_scaled, np.zeros((len(y_pred_scaled), scaler.n_features_in_ - 1))]))[:, 0]

# Salvar CSV de comparação
comparativo = pd.DataFrame({
    'Real': y_test_true,
    'Previsto': y_pred_true
})
comparativo.to_csv(os.path.join(RESULTS_DIR, 'comparativo_teste_multivariado.csv'), index=False)

# Calcular métricas
rmse = mean_squared_error(y_test_true, y_pred_true, squared=False)
mae = mean_absolute_error(y_test_true, y_pred_true)
print(f"📏 RMSE total: {rmse:.4f}")
print(f"📏 MAE  total: {mae:.4f}")

# Plotar comparação
datas = df['Data'].iloc[-len(y_test_true):].reset_index(drop=True)
datas = pd.to_datetime(datas).dt.to_pydatetime()

plt.figure(figsize=(12, 5))
plt.plot(datas, y_test_true, label='Real')
plt.plot(datas, y_pred_true, label='Previsto')
plt.title('Comparação entre valores reais e previstos (modelo multivariado)')
plt.xlabel('Data')
plt.ylabel('Preço (R$)')
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'comparativo_teste_multivariado.png'))
plt.close()

print("✅ Comparação gerada com sucesso!")
