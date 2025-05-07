# run_inference.py
import os
import glob
import pandas as pd
import numpy as np
import cloudpickle
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from core.data_preprocessing_univariado import merge_and_clean_csv, preprocess_data, split_data
from core.predictor_univariado import predict_next_day

# Diretórios
RAW_DIR = 'data/raw'
MODEL_DIR = 'models'
RESULTS_DIR = 'results'

os.makedirs(RESULTS_DIR, exist_ok=True)

# Passo 1: Carregar e processar os dados
csv_files = glob.glob(os.path.join(RAW_DIR, '*.csv'))
df = merge_and_clean_csv(csv_files)

# Passo 2: Carregar modelo e scaler
model_path = os.path.join(MODEL_DIR, 'model_lstm.keras')
scaler_path = os.path.join(MODEL_DIR, 'scaler_lstm.pkl')

model = load_model(model_path)
with open(scaler_path, 'rb') as f:
    scaler = cloudpickle.load(f)

# Passo 3: Pré-processar os dados para simulação de avaliação
data_scaler, X, y = preprocess_data(df, feature_col='Fechamento', sequence_length=60)
X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)

X_test_input = X_test.to_numpy().reshape(-1, 60, 1)
y_pred_scaled = model.predict(X_test_input)

# Reverter a escala
y_test_true = scaler.inverse_transform(y_test.values.reshape(-1, 1))
y_pred_true = scaler.inverse_transform(y_pred_scaled)

# Salvar gráfico de comparação
plt.figure(figsize=(12, 5))
plt.plot(y_test_true, label='Real')
plt.plot(y_pred_true, label='Previsto')
plt.title('Comparação entre valores reais e previstos (dados de teste)')
plt.xlabel('Dias')
plt.ylabel('Preço (R$)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'real_vs_previsto.png'))
plt.close()

# Passo 4: Previsão do próximo dia
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
print(f"📊 Comparativo salvo em: results/real_vs_previsto.png")