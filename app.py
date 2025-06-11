import gradio as gr
import os
import pandas as pd
from train_multivariado import treinar_modelo
from comparar_modelo import comparar_modelo
from validar_e_prever_30_dias import validar_e_prever_30_dias
from core.data_preprocessing_multivariado import merge_and_clean_csv
from utils.historico_updater import atualiza_historico_from_upload
from utils.gerenciamento import criar_papel, remover_papel, remover_modelo

RESULTS_DIR = "results"
MODEL_DIR = "models"
os.makedirs(RESULTS_DIR, exist_ok=True)

def listar_papeis():
    caminho = "data/historico_global"
    if not os.path.exists(caminho):
        return []
    return sorted([f.replace(".csv", "") for f in os.listdir(caminho) if f.endswith(".csv")])

def atualizar_dropdown_papeis():
    return gr.update(choices=listar_papeis())

def wrapper_criar_papel(papel):
    return criar_papel(papel)

def wrapper_atualizar_historico(papel, arquivo):
    if not arquivo:
        return "❌ Nenhum arquivo foi enviado."
    return atualiza_historico_from_upload(papel, arquivo)

def wrapper_preprocessar(papel):
    caminho_historico = f"data/historico_global/{papel}.csv"
    if not os.path.exists(caminho_historico):
        return pd.DataFrame(), None, None
    df = pd.read_csv(caminho_historico)
    export_path = os.path.join(RESULTS_DIR, f"dados_preprocessados_{papel}.csv")
    df.to_csv(export_path, index=False)
    return df, df, export_path

def wrapper_treinar(papel, _, tipo_modelo):
    import sys
    import io

    caminho_historico = f"data/historico_global/{papel}.csv"
    if not os.path.exists(caminho_historico):
        return f"❌ Histórico consolidado de {papel} não encontrado.", None, "", ""

    df = pd.read_csv(caminho_historico)
    old_stdout = sys.stdout
    sys.stdout = mystdout = io.StringIO()

    try:
        resultado, grafico = treinar_modelo(papel, df, tipo_modelo=tipo_modelo)
        log_treino = mystdout.getvalue()
    finally:
        sys.stdout = old_stdout

    return resultado[0], grafico, resultado[1], log_treino

def wrapper_comparar(papel, tipo_modelo):
    caminho_historico = f"data/historico_global/{papel}.csv"
    if not os.path.exists(caminho_historico):
        return f"❌ Histórico consolidado de {papel} não encontrado.", None
    return comparar_modelo(papel, caminho_historico, tipo_modelo=tipo_modelo)

def wrapper_validar(papel, tipo_modelo):
    caminho_historico = f"data/historico_global/{papel}.csv"
    if not os.path.exists(caminho_historico):
        return f"❌ Histórico consolidado de {papel} não encontrado.", None
    return validar_e_prever_30_dias(papel, caminho_historico, tipo_modelo=tipo_modelo)

with gr.Blocks(theme=gr.themes.Monochrome()) as app:
    gr.Markdown("""# 📈 Intelligent Stock Predictor""")

    with gr.Tabs():
        # Aba 1 - Criar novo papel
        with gr.Tab("➕ Criar Papel"):
            novo_papel_input = gr.Textbox(label="Nome do novo papel (ex: PETR4)")
            botao_criar = gr.Button("Criar Papel")
            saida_criacao = gr.Textbox(label="Status")
            botao_criar.click(fn=wrapper_criar_papel, inputs=[novo_papel_input], outputs=[saida_criacao])

        # Aba 2 - Adicionar Dados ao Papel
        with gr.Tab("📤 Adicionar Histórico"):
            papel_input_hist = gr.Dropdown(label="Selecione o Papel", choices=listar_papeis(), interactive=True)
            btn_refresh_papeis = gr.Button("🔄 Atualizar lista de papéis")
            btn_refresh_papeis.click(fn=atualizar_dropdown_papeis, inputs=[], outputs=[papel_input_hist])

            upload_historico = gr.File(label="Upload CSV", file_types=[".csv"])
            botao_atualizar = gr.Button("📂 Atualizar Histórico")
            saida_atualizacao = gr.Textbox(label="Status da Atualização")
            botao_atualizar.click(fn=wrapper_atualizar_historico, inputs=[papel_input_hist, upload_historico], outputs=[saida_atualizacao])

        # Aba 3 - Treinar Modelo
        with gr.Tab("🔧 Pré-processar + Treinar"):
            papel_treino = gr.Dropdown(label="Selecione o Papel", choices=listar_papeis(), interactive=True)
            btn_refresh_treino = gr.Button("🔄 Atualizar lista de papéis")
            btn_refresh_treino.click(fn=atualizar_dropdown_papeis, inputs=[], outputs=[papel_treino])

            modelo_dropdown = gr.Dropdown(label="Tipo de Modelo", choices=["lstm", "gru"], value="lstm")

            preprocessar_btn = gr.Button("🔍 Pré-processar")
            tabela_saida = gr.Dataframe(label="Dados do Papel")
            dados_unificados = gr.State()
            download_csv = gr.File(label="CSV Exportado")

            preprocessar_btn.click(fn=wrapper_preprocessar, inputs=[papel_treino], outputs=[tabela_saida, dados_unificados, download_csv])

            treinar_btn = gr.Button("🚀 Treinar Modelo")
            resultado_saida = gr.Textbox(label="Mensagem de Treinamento")
            grafico_saida = gr.Image(label="Gráfico de Perda")
            metricas_saida = gr.Textbox(label="Últimas Métricas de MSE")
            log_saida = gr.Textbox(label="Log do Treinamento", lines=20)

            treinar_btn.click(
                fn=wrapper_treinar,
                inputs=[papel_treino, dados_unificados, modelo_dropdown],
                outputs=[resultado_saida, grafico_saida, metricas_saida, log_saida]
            )

        # Aba 4 - Avaliação & Previsão
        with gr.Tab("📊 Avaliação e Previsão"):
            papel_avaliar = gr.Dropdown(label="Escolha o Papel", choices=listar_papeis(), interactive=True)
            btn_refresh_avaliar = gr.Button("🔄 Atualizar lista de papéis")
            btn_refresh_avaliar.click(fn=atualizar_dropdown_papeis, inputs=[], outputs=[papel_avaliar])

            modelo_dropdown_eval = gr.Dropdown(label="Tipo de Modelo", choices=["lstm", "gru"], value="lstm")

            comparar_btn = gr.Button("📈 Comparar com Teste")
            resultado_comp = gr.Textbox(label="Métricas de Previsão")
            grafico_comp = gr.Image(label="Gráfico Real x Previsto")

            validar_btn = gr.Button("🔮 Validar e Prever 30 dias")
            resultado_val = gr.Textbox(label="Resultado da Previsão")
            grafico_val = gr.Image(label="Gráfico Previsão 30 dias")

            comparar_btn.click(
                fn=wrapper_comparar,
                inputs=[papel_avaliar, modelo_dropdown_eval],
                outputs=[resultado_comp, grafico_comp]
            )
            validar_btn.click(
                fn=wrapper_validar,
                inputs=[papel_avaliar, modelo_dropdown_eval],
                outputs=[resultado_val, grafico_val]
            )

        # Aba 5 - Gerenciar Papéis
        with gr.Tab("🧾 Gerenciar Papéis"):
            papel_para_apagar = gr.Textbox(label="Nome do papel a apagar")
            btn_apagar_papel = gr.Button("🗑️ Apagar Histórico do Papel")
            out_apagar_papel = gr.Textbox(label="Status Remoção")
            btn_apagar_papel.click(fn=remover_papel, inputs=[papel_para_apagar], outputs=[out_apagar_papel])

            papel_modelo = gr.Textbox(label="Nome do papel para apagar treino")
            btn_apagar_modelo = gr.Button("🗑️ Apagar Modelo Treinado")
            out_apagar_modelo = gr.Textbox(label="Status Remoção de Modelo")
            btn_apagar_modelo.click(fn=remover_modelo, inputs=[papel_modelo], outputs=[out_apagar_modelo])

if __name__ == "__main__":
    app.launch()
