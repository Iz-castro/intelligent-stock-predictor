# run_training.py
import os
import glob
import pandas as pd
from core.data_preprocessing_univariado import merge_and_clean_csv, preprocess_data, split_data
from core.model_lstm_univariado import build_lstm_model
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
df.to_csv(os.path.join(RESULTS_DIR, 'dados_unificados.csv'), index=False)

# Passo 3: Pré-processamento
scaler, X, y = preprocess_data(df, feature_col='Fechamento', sequence_length=60)
X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)

# Ajustar entrada para o modelo
X_train = X_train.to_numpy().reshape(-1, 60, 1)
X_test = X_test.to_numpy().reshape(-1, 60, 1)

# Passo 4: Construir e treinar o modelo
model = build_lstm_model(X_train.shape)

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
model.save(os.path.join(MODEL_DIR, 'model_lstm.keras'))
with open(os.path.join(MODEL_DIR, 'scaler_lstm.pkl'), 'wb') as f:
    cloudpickle.dump(scaler, f)

# Passo 6: Plotar gráfico de treino
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Treino')
plt.plot(history.history['val_loss'], label='Validação')
plt.title('Erro do Modelo durante o Treinamento')
plt.xlabel('Épocas')
plt.ylabel('MSE')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'treinamento_lstm.png'))
plt.close()

print("✅ Modelo treinado e salvo com sucesso!")
