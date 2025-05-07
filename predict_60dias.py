# predict_60dias.py
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cloudpickle
from tensorflow.keras.models import load_model
from core.data_preprocessing import merge_and_clean_csv
from core.predictor import predict_next_day
from datetime import datetime, timedelta

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

# Obter os últimos 60 preços para previsão
sequence = df.tail(60)['Fechamento'].values.tolist()

if len(sequence) < 60:
    raise ValueError("❌ São necessários pelo menos 60 valores para gerar as previsões.")

futuro = []
entrada = np.array(sequence)

for _ in range(60):
    entrada_reshape = entrada[-60:].reshape(-1, 1)
    previsao = predict_next_day(model, entrada_reshape, scaler)
    futuro.append(previsao)
    entrada = np.append(entrada, previsao)

# Gerar datas simuladas a partir da última data
ultima_data = pd.to_datetime(df['Data'].iloc[-1])
datas_futuras = []
contador = 1

while len(datas_futuras) < 60:
    proxima = ultima_data + timedelta(days=contador)
    if proxima.weekday() < 5:  # ignorar sábado/domingo
        datas_futuras.append(proxima.strftime('%Y-%m-%d'))
    contador += 1

# Salvar como DataFrame
df_previsao = pd.DataFrame({
    'Data': datas_futuras,
    'Fechamento_Previsto': futuro
})

df_previsao.to_csv(os.path.join(RESULTS_DIR, 'previsao_60_dias.csv'), index=False)

# Plotar gráfico
plt.figure(figsize=(12, 5))
plt.plot(df['Data'].tail(60), df['Fechamento'].tail(60), label='Histórico Recente')
plt.plot(df_previsao['Data'], df_previsao['Fechamento_Previsto'], label='Previsão (60 dias)', linestyle='--')
plt.title('Previsão de Fechamento para os Próximos 60 Dias Úteis')
plt.xlabel('Data')
plt.ylabel('Preço (R$)')
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'previsao_60_dias.png'))
plt.close()

print("✅ Previsão dos próximos 60 dias gerada com sucesso!")
