# utils/metrics.py
import os
import pandas as pd
from datetime import datetime

def salvar_metricas(papel: str, rmse: float, mae: float, caminho_csv: str = "results/metricas_modelos.csv", substituir: bool = True):
    nova_linha = pd.DataFrame({
        "Papel": [papel],
        "RMSE": [rmse],
        "MAE": [mae],
        "DataHora": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
    })

    if os.path.exists(caminho_csv):
        df_existente = pd.read_csv(caminho_csv)
        if substituir:
            df_existente = df_existente[df_existente["Papel"] != papel]
        df_final = pd.concat([df_existente, nova_linha], ignore_index=True)
    else:
        df_final = nova_linha

    df_final.to_csv(caminho_csv, index=False)
    return caminho_csv