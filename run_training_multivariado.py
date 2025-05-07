# run_training_02.py
import os
import glob
import pandas as pd
import numpy as np
from core.data_preprocessing_multivariado import merge_and_clean_csv, preprocess_data, split_data
from core.model_lstm_multivariado import build_lstm_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import cloudpickle
import matplotlib.pyplot as plt

# Diretórios
RAW_DIR = 'data/raw'
MODEL_DIR = 'models'
RESULTS_DIR = 'results'

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Passo 1: Carregar arquivos CSV da pasta raw
csv_files = glob.glob(os.path.join(RAW_DIR, '*.csv'))
print(f"📁 Encontrados {len(csv_files)} arquivos CSV para processamento.")

# Passo 2: Unificar e limpar os dados
df = merge_and_clean_csv(csv_files)
df.to_csv(os.path.join(RESULTS_DIR, 'dados_unificados_multivariado.csv'), index=False)

# Passo 3: Pré-processamento com múltiplas features
feature_cols = ['Fechamento', 'Retorno_%', 'MM9', 'RSI']
scaler, X, y = preprocess_data(df, feature_cols=feature_cols, sequence_length=60)

# Ajustar dimensões
X_np = X
X_train, X_test, y_train, y_test = train_test_split(X_np, y, test_size=0.2, shuffle=False)

# Passo 4: Construir e treinar o modelo
model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
callback = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=32,
    callbacks=[callback],
    verbose=1
)

# Passo 5: Salvar modelo e scaler
model.save(os.path.join(MODEL_DIR, 'model_lstm_multivariado.keras'))
with open(os.path.join(MODEL_DIR, 'scaler_lstm_multivariado.pkl'), 'wb') as f:
    cloudpickle.dump(scaler, f)

# Passo 6: Plotar gráfico de treino
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Treino')
plt.plot(history.history['val_loss'], label='Validação')
plt.title('Erro do Modelo Multivariado durante o Treinamento')
plt.xlabel('Épocas')
plt.ylabel('MSE')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'treinamento_multivariado.png'))
plt.close()

print("✅ Modelo multivariado treinado e salvo com sucesso!")
