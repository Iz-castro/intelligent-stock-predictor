import gradio as gr
import os
import pandas as pd
from train_multivariado import treinar_modelo
from comparar_modelo import comparar_modelo
from validar_e_prever_30_dias import validar_e_prever_30_dias
from core.data_preprocessing_multivariado import merge_and_clean_csv

RESULTS_DIR = "results"
MODEL_DIR = "models"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def listar_papeis_treinados():
    if not os.path.isdir(MODEL_DIR):
        return []
    arquivos = os.listdir(MODEL_DIR)
    papeis = set()
    for nome in arquivos:
        if nome.startswith("model_lstm_multivariado_") and nome.endswith(".keras"):
            papel = nome.replace("model_lstm_multivariado_", "").replace(".keras", "")
            papeis.add(papel)
    return sorted(papeis)

def wrapper_preprocessar(arquivos):
    if not arquivos:
        return pd.DataFrame(), None, None
    file_paths = [f.name for f in arquivos]
    df = merge_and_clean_csv(file_paths)
    export_path = os.path.join(RESULTS_DIR, "dados_preprocessados.csv")
    df.to_csv(export_path, index=False)
    return df, df, export_path

def wrapper_treinar(papel, df):
    import sys
    import io

    if df is None or df.empty:
        return "❌ Nenhum dado pré-processado disponível.", None, "", ""

    # Captura o stdout do model.fit
    old_stdout = sys.stdout
    sys.stdout = mystdout = io.StringIO()

    try:
        resultado, grafico = treinar_modelo(papel, df)
        log_treino = mystdout.getvalue()
    finally:
        sys.stdout = old_stdout

    return resultado[0], grafico, resultado[1], log_treino

def wrapper_comparar(papel, arquivo):
    if not arquivo:
        return "❌ Por favor, envie um arquivo CSV.", None
    return comparar_modelo(papel, arquivo.name)

def wrapper_validar(papel, arquivo):
    if not arquivo:
        return "❌ Por favor, envie um arquivo CSV.", None
    return validar_e_prever_30_dias(papel, arquivo.name)

with gr.Blocks(theme=gr.themes.Monochrome()) as app:
    gr.Markdown("""# 📈 Intelligent Stock Predictor - Multivariado""")

    with gr.Tabs():
        # Aba 1 - Pré-processar + Treinar
        with gr.Tab("Pré-processar + Treinar"):
            arquivos_csv = gr.File(label="Selecione CSVs de histórico", file_types=['.csv'], file_count="multiple")
            preprocessar_btn = gr.Button("🔍 Pré-processar Arquivos")
            tabela_saida = gr.Dataframe(label="Dados Unificados (ordenados por data)")
            dados_unificados = gr.State()
            download_csv = gr.File(label="CSV Limpo Exportado")

            preprocessar_btn.click(fn=wrapper_preprocessar, inputs=[arquivos_csv], outputs=[tabela_saida, dados_unificados, download_csv])

            papel_input = gr.Textbox(label="Nome do Papel (ex: PETR4)")
            treinar_btn = gr.Button("🔧 Treinar Modelo")
            resultado_saida = gr.Textbox(label="Mensagem de Treinamento")
            grafico_saida = gr.Image(label="Gráfico de Perda")
            metricas_saida = gr.Textbox(label="Últimas Métricas de MSE")
            log_saida = gr.Textbox(label="Log do Treinamento", lines=20)

            treinar_btn.click(fn=wrapper_treinar,
                              inputs=[papel_input, dados_unificados],
                              outputs=[resultado_saida, grafico_saida, metricas_saida, log_saida])

        # Aba 2 - Comparar
        with gr.Tab("Comparar com Teste"):
            papel_dropdown = gr.Dropdown(label="Escolha o Papel Treinado", choices=listar_papeis_treinados(), interactive=True)
            atualizar_btn = gr.Button("🔄 Atualizar Lista de Papéis")
            csv_input_comp = gr.File(label="Upload do CSV de histórico")
            comparar_btn = gr.Button("📊 Comparar Previsão")
            resultado_comp = gr.Textbox(label="Métricas de Previsão (RMSE, MAE)")
            grafico_comp = gr.Image(label="Gráfico Real x Previsto")

            def atualizar_dropdown():
                return gr.update(choices=listar_papeis_treinados())

            atualizar_btn.click(fn=atualizar_dropdown, inputs=[], outputs=[papel_dropdown])
            comparar_btn.click(fn=wrapper_comparar, inputs=[papel_dropdown, csv_input_comp], outputs=[resultado_comp, grafico_comp])

        # Aba 3 - Previsão Futura
        with gr.Tab("Previsão Futura"):
            papel_dropdown2 = gr.Dropdown(label="Escolha o Papel Treinado", choices=listar_papeis_treinados(), interactive=True)
            atualizar_btn2 = gr.Button("🔄 Atualizar Lista de Papéis")
            csv_input_val = gr.File(label="Upload do CSV de histórico")
            validar_btn = gr.Button("🔮 Validar e Prever 30 dias")
            resultado_val = gr.Textbox(label="Resultado da Previsão")
            grafico_val = gr.Image(label="Gráfico 30 dias: Real + Futuro")

            atualizar_btn2.click(fn=atualizar_dropdown, inputs=[], outputs=[papel_dropdown2])
            validar_btn.click(fn=wrapper_validar, inputs=[papel_dropdown2, csv_input_val], outputs=[resultado_val, grafico_val])

if __name__ == "__main__":
    app.launch()
