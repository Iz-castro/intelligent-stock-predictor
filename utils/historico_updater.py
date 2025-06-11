import os
import pandas as pd
from core.data_preprocessing_multivariado import clean_and_format_dataframe

def atualiza_historico_from_upload(papel: str, uploaded_file):
    """
    Atualiza o histórico consolidado do papel especificado com base em um arquivo CSV enviado via Gradio.
    """
    import pandas as pd
    from core.data_preprocessing_multivariado import clean_and_format_dataframe
    import os

    if not papel:
        return "❌ Nenhum papel informado."

    papel = papel.strip().upper()
    historico_path = f"data/historico_global/{papel}.csv"
    os.makedirs("data/historico_global", exist_ok=True)

    try:
        novos_dados = clean_and_format_dataframe(pd.read_csv(uploaded_file))
    except Exception as e:
        return f"❌ Erro ao ler o arquivo CSV: {e}"

    if os.path.exists(historico_path):
        historico = pd.read_csv(historico_path)
        combinado = pd.concat([historico, novos_dados]).drop_duplicates(subset='Data')
    else:
        combinado = novos_dados

    combinado = combinado.sort_values('Data').reset_index(drop=True)

    try:
        combinado.to_csv(historico_path, index=False)
    except Exception as e:
        return f"❌ Erro ao salvar histórico: {e}"

    return f"✅ Histórico do papel {papel} atualizado. Total de registros: {len(combinado)}"
