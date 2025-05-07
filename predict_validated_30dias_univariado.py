# predict_validated_30dias.py
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cloudpickle
from tensorflow.keras.models import load_model
from core.data_preprocessing_univariado import merge_and_clean_csv, preprocess_data, split_data
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import timedelta

# Diretórios
RAW_DIR = 'data/raw'
MODEL_DIR = 'models'
RESULTS_DIR = 'results'

os.makedirs(RESULTS_DIR, exist_ok=True)

# Carregar dados e modelo
csv_files = glob.glob(os.path.join(RAW_DIR, '*.csv'))
df = merge_and_clean_csv(csv_files)
model = load_model(os.path.join(MODEL_DIR, 'model_lstm.keras'))

with open(os.path.join(MODEL_DIR, 'scaler_lstm.pkl'), 'rb') as f:
    scaler = cloudpickle.load(f)

# Pré-processar e separar dados
scaler_data, X, y = preprocess_data(df, feature_col='Fechamento', sequence_length=60)
X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)

# Últimos 30 dias do conjunto de teste
X_test_input = X_test.to_numpy().reshape(-1, 60, 1)
y_pred_scaled = model.predict(X_test_input)

y_test_true = scaler.inverse_transform(y_test.values.reshape(-1, 1))
y_pred_true = scaler.inverse_transform(y_pred_scaled)

# Selecionar últimos 30 dias
ultimos_30_pred = y_pred_true[-30:].flatten()
ultimos_30_true = y_test_true[-30:].flatten()

# Gerar datas correspondentes aos últimos 30 dias
base_index = df['Data'].iloc[-len(y_test_true):].reset_index(drop=True)
datas_validacao = pd.to_datetime(base_index[-30:]).dt.to_pydatetime()

# Calcular métricas de erro
rmse = mean_squared_error(ultimos_30_true, ultimos_30_pred, squared=False)
mae = mean_absolute_error(ultimos_30_true, ultimos_30_pred)
print(f"📏 RMSE últimos 30 dias: {rmse:.4f}")
print(f"📏 MAE  últimos 30 dias: {mae:.4f}")

# Plotar validação
plt.figure(figsize=(12, 5))
plt.plot(datas_validacao, ultimos_30_true, label='Real')
plt.plot(datas_validacao, ultimos_30_pred, label='Previsto')
plt.title('Validação: Previsão dos Últimos 30 Dias do Conjunto de Teste')
plt.xlabel('Data')
plt.ylabel('Preço (R$)')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'validacao_30_dias.png'))
plt.close()

# Previsão futura com blocos de 5 dias recursivos
sequencia = df['Fechamento'].tolist()[-60:]
futuro = []
entrada = np.array(sequencia)

limite_min = entrada[-1] * 0.8
limite_max = entrada[-1] * 1.2

for _ in range(6):  # 6 blocos de 5 dias = 30 dias
    for _ in range(5):
        entrada_reshape = entrada[-60:].reshape(-1, 1)
        previsao = model.predict(entrada_reshape.reshape(1, 60, 1))
        valor = scaler.inverse_transform(previsao)[0][0]
        valor_clipped = max(min(valor, limite_max), limite_min)
        futuro.append(valor_clipped)
        entrada = np.append(entrada, valor_clipped)

# Gerar datas futuras (30 dias úteis)
ultima_data = pd.to_datetime(df['Data'].iloc[-1])
datas_futuras = []
contador = 1
while len(datas_futuras) < 30:
    proxima = ultima_data + timedelta(days=contador)
    if proxima.weekday() < 5:
        datas_futuras.append(proxima)
    contador += 1

datas_futuras = pd.to_datetime(datas_futuras).to_pydatetime()

# Salvar previsão futura
df_futuro = pd.DataFrame({
    'Data': pd.to_datetime(datas_futuras).strftime('%Y-%m-%d'),
    'Fechamento_Previsto': futuro
})
df_futuro.to_csv(os.path.join(RESULTS_DIR, 'previsao_30_dias_futuros.csv'), index=False)

# Plotar gráfico combinado
plt.figure(figsize=(12, 5))
plt.plot(datas_validacao, ultimos_30_true, label='Real (últimos 30 dias)')
plt.plot(datas_validacao, ultimos_30_pred, label='Previsto (últimos 30 dias)')
plt.plot(datas_futuras, futuro, label='Previsto (futuros)', linestyle='--')
plt.title('Validação + Previsão para os Próximos 30 Dias Úteis')
plt.xlabel('Data')
plt.ylabel('Preço (R$)')
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'validacao_e_previsao_30_dias.png'))
plt.close()

print("✅ Validação e previsão dos próximos 30 dias geradas com sucesso!")
# predict_validated_30dias.py
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cloudpickle
from tensorflow.keras.models import load_model
from core.data_preprocessing import merge_and_clean_csv, preprocess_data, split_data
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import timedelta

# Diretórios
RAW_DIR = 'data/raw'
MODEL_DIR = 'models'
RESULTS_DIR = 'results'

os.makedirs(RESULTS_DIR, exist_ok=True)

# Carregar dados e modelo
csv_files = glob.glob(os.path.join(RAW_DIR, '*.csv'))
df = merge_and_clean_csv(csv_files)
model = load_model(os.path.join(MODEL_DIR, 'model_lstm.keras'))

with open(os.path.join(MODEL_DIR, 'scaler_lstm.pkl'), 'rb') as f:
    scaler = cloudpickle.load(f)

# Pré-processar e separar dados
scaler_data, X, y = preprocess_data(df, feature_col='Fechamento', sequence_length=60)
X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)

# Últimos 30 dias do conjunto de teste
X_test_input = X_test.to_numpy().reshape(-1, 60, 1)
y_pred_scaled = model.predict(X_test_input)

y_test_true = scaler.inverse_transform(y_test.values.reshape(-1, 1))
y_pred_true = scaler.inverse_transform(y_pred_scaled)

# Selecionar últimos 30 dias
ultimos_30_pred = y_pred_true[-30:].flatten()
ultimos_30_true = y_test_true[-30:].flatten()

# Gerar datas correspondentes aos últimos 30 dias
base_index = df['Data'].iloc[-len(y_test_true):].reset_index(drop=True)
datas_validacao = pd.to_datetime(base_index[-30:]).dt.to_pydatetime()

# Calcular métricas de erro
rmse = mean_squared_error(ultimos_30_true, ultimos_30_pred, squared=False)
mae = mean_absolute_error(ultimos_30_true, ultimos_30_pred)
print(f"📏 RMSE últimos 30 dias: {rmse:.4f}")
print(f"📏 MAE  últimos 30 dias: {mae:.4f}")

# Plotar validação
plt.figure(figsize=(12, 5))
plt.plot(datas_validacao, ultimos_30_true, label='Real')
plt.plot(datas_validacao, ultimos_30_pred, label='Previsto')
plt.title('Validação: Previsão dos Últimos 30 Dias do Conjunto de Teste')
plt.xlabel('Data')
plt.ylabel('Preço (R$)')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'validacao_30_dias.png'))
plt.close()

# Previsão futura com blocos de 5 dias recursivos
sequencia = df['Fechamento'].tolist()[-60:]
futuro = []
entrada = np.array(sequencia)

limite_min = entrada[-1] * 0.8
limite_max = entrada[-1] * 1.2

for _ in range(6):  # 6 blocos de 5 dias = 30 dias
    for _ in range(5):
        entrada_reshape = entrada[-60:].reshape(-1, 1)
        previsao = model.predict(entrada_reshape.reshape(1, 60, 1))
        valor = scaler.inverse_transform(previsao)[0][0]
        valor_clipped = max(min(valor, limite_max), limite_min)
        futuro.append(valor_clipped)
        entrada = np.append(entrada, valor_clipped)

# Gerar datas futuras (30 dias úteis)
ultima_data = pd.to_datetime(df['Data'].iloc[-1])
datas_futuras = []
contador = 1
while len(datas_futuras) < 30:
    proxima = ultima_data + timedelta(days=contador)
    if proxima.weekday() < 5:
        datas_futuras.append(proxima)
    contador += 1

datas_futuras = pd.to_datetime(datas_futuras).to_pydatetime()

# Salvar previsão futura
df_futuro = pd.DataFrame({
    'Data': pd.to_datetime(datas_futuras).strftime('%Y-%m-%d'),
    'Fechamento_Previsto': futuro
})
df_futuro.to_csv(os.path.join(RESULTS_DIR, 'previsao_30_dias_futuros.csv'), index=False)

# Plotar gráfico combinado
plt.figure(figsize=(12, 5))
plt.plot(datas_validacao, ultimos_30_true, label='Real (últimos 30 dias)')
plt.plot(datas_validacao, ultimos_30_pred, label='Previsto (últimos 30 dias)')
plt.plot(datas_futuras, futuro, label='Previsto (futuros)', linestyle='--')
plt.title('Validação + Previsão para os Próximos 30 Dias Úteis')
plt.xlabel('Data')
plt.ylabel('Preço (R$)')
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'validacao_e_previsao_30_dias.png'))
plt.close()

print("✅ Validação e previsão dos próximos 30 dias geradas com sucesso!")
