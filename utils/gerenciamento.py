import os
import pandas as pd

def criar_papel(papel):
    papel = papel.strip().upper()
    path = f"data/historico_global/{papel}.csv"
    os.makedirs("data/historico_global", exist_ok=True)
    if os.path.exists(path):
        return f"⚠️ Papel {papel} já existe."
    df_vazio = pd.DataFrame(columns=['Data', 'Fechamento', 'Abertura', 'Maxima', 'Minima', 'Volume', 'Variacao', 'Retorno_%', 'MM9', 'RSI'])
    df_vazio.to_csv(path, index=False)
    return f"✅ Papel {papel} criado com sucesso."

def remover_papel(papel):
    papel = papel.strip().upper()
    path = f"data/historico_global/{papel}.csv"
    if os.path.exists(path):
        os.remove(path)
        return f"🗑️ Histórico do papel {papel} removido com sucesso."
    return f"⚠️ Papel {papel} não encontrado."

def remover_modelo(papel):
    papel = papel.strip().upper()
    model_path = f"models/model_lstm_multivariado_{papel}.keras"
    scaler_path = f"models/scaler_lstm_multivariado_{papel}.pkl"
    mensagens = []

    if os.path.exists(model_path):
        os.remove(model_path)
        mensagens.append(f"✅ Modelo de {papel} removido.")
    else:
        mensagens.append("⚠️ Modelo não encontrado.")

    if os.path.exists(scaler_path):
        os.remove(scaler_path)
        mensagens.append(f"✅ Scaler de {papel} removido.")
    else:
        mensagens.append("⚠️ Scaler não encontrado.")

    return "\n".join(mensagens)
