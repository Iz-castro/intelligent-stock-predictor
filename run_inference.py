# run_inference.py
import os
import glob
import pandas as pd
import numpy as np
import cloudpickle
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from core.data_preprocessing import merge_and_clean_csv, clean_and_format_dataframe
from core.predictor import predict_next_day

# Diretórios
RAW_DIR = 'data/raw'
MODEL_DIR = 'models'
RESULTS_DIR = 'results'

os.makedirs(RESULTS_DIR, exist_ok=True)

# Passo 1: Carregar últimos dados
csv_files = glob.glob(os.path.join(RAW_DIR, '*.csv'))
df = merge_and_clean_csv(csv_files)

# Passo 2: Carregar modelo e scaler
model_path = os.path.join(MODEL_DIR, 'model_lstm.keras')
scaler_path = os.path.join(MODEL_DIR, 'scaler_lstm.pkl')

model = load_model(model_path)
with open(scaler_path, 'rb') as f:
    scaler = cloudpickle.load(f)

# Passo 3: Previsão com os últimos 60 dias
last_sequence = df.tail(60).copy()
last_sequence = last_sequence.reset_index(drop=True)

if len(last_sequence) < 60:
    raise ValueError("❌ Dados insuficientes para previsão. São necessários pelo menos 60 dias de dados.")

X_input = last_sequence['Fechamento'].values.reshape(-1, 1)
predicted_value = predict_next_day(model, X_input, scaler)

print(f"📈 Previsão para o próximo dia útil: R$ {predicted_value:.2f}")
# Salvar previsão textual
with open(os.path.join(RESULTS_DIR, 'previsao_proximo_dia.txt'), 'w') as f:
    f.write(f"Previsão para o próximo dia útil: R$ {predicted_value:.2f}\n")

# Plotar últimos valores reais + previsão
fechamentos_reais = last_sequence['Fechamento'].tolist()
fechamentos_reais.append(predicted_value)

dias = [f"Dia {i+1}" for i in range(60)] + ["Previsão"]

plt.figure(figsize=(12, 5))
plt.plot(dias, fechamentos_reais, marker='o', linestyle='-', label='Preço')
plt.axvline(x=59.5, color='gray', linestyle='--', label='Previsão')
plt.title("Previsão de Fechamento para o Próximo Dia")
plt.xlabel("Dias")
plt.ylabel("Preço (R$)")
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.tight_layout()

graph_path = os.path.join(RESULTS_DIR, 'grafico_previsao.png')
plt.savefig(graph_path)
plt.close()

print(f"📊 Gráfico salvo em: {graph_path}")