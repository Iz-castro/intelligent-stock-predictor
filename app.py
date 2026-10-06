"""Interface Gradio do Intelligent Stock Predictor.

Permite escolher o papel, as datas de corte, a rede (GRU, LSTM ou nenhuma), as
etapas do estudo e os grupos de métricas exibidos. Cada execução baixa os dados do
Yahoo Finance (ou usa o cache), treina os modelos e salva tudo em results/<ATIVO>/.

Uso:
    python app.py
"""
import contextlib
import inspect
import io
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # silencia o log informativo do TensorFlow

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import gradio as gr  # noqa: E402
import pandas as pd  # noqa: E402

import config  # noqa: E402
import pipeline  # noqa: E402
from core import dados  # noqa: E402

ATIVOS = ["PETR4", "VALE3", "ITUB4", "BBAS3", "BBDC4", "ABEV3", "WEGE3", "BBSE3", "TAEE11", "CMIG4"]
REDES = {"GRU": "gru", "LSTM": "lstm", "Nenhuma (só modelos estatísticos)": None}
ETAPAS = ["Diagnóstico", "Previsão e métricas", "Risco (VaR)", "Projeção GBM"]
GRUPOS_METRICAS = {
    "Erro no preço (RMSE, MAE, MAPE, R²)": ["RMSE_preco", "MAE_preco", "MAPE_preco_%", "R2_preco"],
    "Erro no retorno (RMSE, MASE, U de Theil, R² fora da amostra)":
        ["RMSE_ret_%", "MASE_ret", "U_Theil", "R2_fora_amostra_%"],
    "Direção (acerto e teste binomial)": ["acerto_direcional_%", "binomial_p"],
    "Diebold-Mariano contra o passeio aleatório": ["DM_estat", "DM_p"],
}


def _tabela(df: pd.DataFrame | None, casas: int = 4):
    return None if df is None else df.round(casas).reset_index()


def _validar_data(texto: str, nome: str) -> str:
    try:
        return pd.Timestamp(texto.strip()).strftime("%Y-%m-%d")
    except (ValueError, AttributeError) as erro:
        raise ValueError(f"Data inválida em '{nome}': use o formato AAAA-MM-DD.") from erro


def executar(ativo, inicio, fim_treino, fim_validacao, rede, criterio, etapas, grupos_metricas, atualizar,
             progresso=gr.Progress()):
    log = io.StringIO()
    saida = [None] * 16

    try:
        with contextlib.redirect_stdout(log):
            if not ativo:
                raise ValueError("Escolha ou digite um papel.")
            inicio = _validar_data(inicio, "início")
            fim_treino = _validar_data(fim_treino, "fim do treino")
            fim_validacao = _validar_data(fim_validacao, "fim da validação")
            if not inicio < fim_treino < fim_validacao:
                raise ValueError("As datas precisam seguir a ordem início < fim do treino < fim da validação.")

            config.definir_ativo(ativo, inicio, fim_treino, fim_validacao)
            print(f"Ativo: {config.TICKER} | resultados em results/{config.NOME_ATIVO}/")

            progresso(0.05, desc="Carregando dados")
            base, rotulos = pipeline.etapa_dados(atualizar=atualizar)
            saida[1] = _tabela(dados.resumo_periodos(base.index))
            diag = aval = risc = proj = None

            if "Diagnóstico" in etapas:
                progresso(0.15, desc="Diagnóstico")
                diag = pipeline.etapa_diagnostico(base, rotulos)
                saida[2] = _tabela(diag["raiz_unitaria"])
                saida[3] = _tabela(diag["momentos"].to_frame("valor"), 5)
                saida[4] = _tabela(diag["dependencia"])
                saida[5], saida[6] = diag["figuras"]

            if "Previsão e métricas" in etapas:
                tipo = REDES[rede]
                progresso(0.35, desc=f"Previsão (ARIMA{' e ' + tipo.upper() if tipo else ''})")
                aval = pipeline.etapa_avaliacao(base, rotulos, tipo_rede=tipo, criterio_arima=criterio)
                colunas = [c for g in (grupos_metricas or GRUPOS_METRICAS) for c in GRUPOS_METRICAS[g]]
                saida[7] = _tabela(aval["tabela"][colunas])
                tab = aval["tabela"]
                saida[8] = (f"**Teste:** {tab.attrs['pregoes']} pregões, {tab.attrs['taxa_dias_alta_%']:.1f}% de "
                            f"dias de alta. **ARIMA escolhido no treino ({aval['arima']['criterio']}):** {aval['arima']['ordem']} "
                            f"{'com' if aval['arima']['com_constante'] else 'sem'} constante. "
                            "U de Theil abaixo de 1, R² fora da amostra positivo e Diebold-Mariano negativo "
                            "com p-valor baixo indicariam ganho sobre o passeio aleatório.")
                figuras = aval["figuras"]
                saida[9] = figuras[0] if len(figuras) == 2 else None   # curva de treino, se houver rede
                saida[10] = figuras[-1]

            if "Risco (VaR)" in etapas:
                progresso(0.75, desc="Backtest do VaR")
                risc = pipeline.etapa_risco(base, rotulos)
                saida[11] = _tabela(risc["resumo"])
                saida[12] = risc["figuras"][0]

            if "Projeção GBM" in etapas:
                progresso(0.85, desc="Projeção GBM")
                vol = risc["vol_ewma_proximo_pregao"] if risc else None
                proj = pipeline.etapa_projecao(base, vol)
                saida[13] = _tabela(proj["cenarios"].T.rename_axis("medida"), 2)
                saida[14] = proj["figuras"][0]

            progresso(0.95, desc="Resumo")
            saida[15] = pipeline.gerar_resumo(base, rotulos, diag, aval, risc, proj)
            print("Concluído.")
    except Exception as erro:  # a mensagem aparece no log da interface
        log.write(f"\nERRO: {type(erro).__name__}: {erro}\n")

    saida[0] = log.getvalue()
    arquivos = sorted(str(p) for p in config.DIR_RESULTADOS.glob("*") if p.is_file())
    return saida + [arquivos or None]


# A partir do Gradio 6 o tema é passado no launch(); antes disso, no Blocks()
TEMA = gr.themes.Monochrome()
TEMA_NO_LAUNCH = "theme" in inspect.signature(gr.Blocks.launch).parameters

with gr.Blocks(title="Intelligent Stock Predictor", **({} if TEMA_NO_LAUNCH else {"theme": TEMA})) as app:
    gr.Markdown("# Intelligent Stock Predictor\n"
                "Previsão um passo à frente, backtest de VaR e projeção por GBM para papéis da B3 "
                "(dados ajustados do Yahoo Finance). Projeto educacional, sem recomendação de investimento.")

    with gr.Row():
        with gr.Column(scale=1):
            ativo = gr.Dropdown(ATIVOS, value="PETR4", label="Papel", allow_custom_value=True,
                                info="Escolha da lista ou digite outro código da B3 (o sufixo .SA é incluído)")
            with gr.Row():
                inicio = gr.Textbox(value=config.DATA_INICIO, label="Início dos dados")
                fim_treino = gr.Textbox(value=config.FIM_TREINO, label="Fim do treino")
                fim_validacao = gr.Textbox(value=config.FIM_VALIDACAO, label="Fim da validação")
            with gr.Row():
                rede = gr.Radio(list(REDES), value="GRU", label="Rede neural")
                criterio = gr.Radio([("AICc", "aicc"), ("BIC", "bic")], value=config.CRITERIO_ARIMA,
                                    label="Critério do ARIMA")
            etapas = gr.CheckboxGroup(ETAPAS, value=ETAPAS, label="Etapas")
            grupos_metricas = gr.CheckboxGroup(list(GRUPOS_METRICAS), value=list(GRUPOS_METRICAS),
                                               label="Métricas exibidas na previsão")
            atualizar = gr.Checkbox(value=True, label="Baixar dados atualizados do Yahoo (desmarque para usar o cache)")
            botao = gr.Button("Executar", variant="primary")
        with gr.Column(scale=1):
            log = gr.Textbox(label="Log da execução", lines=14, max_lines=30)
            periodos = gr.Dataframe(label="Períodos", interactive=False)

    with gr.Tabs():
        with gr.Tab("Diagnóstico"):
            raiz = gr.Dataframe(label="Raiz unitária (ADF e KPSS)", interactive=False)
            with gr.Row():
                momentos = gr.Dataframe(label="Momentos do log-retorno", interactive=False)
                dependencia = gr.Dataframe(label="Dependência temporal (p-valores)", interactive=False)
            fig_serie = gr.Image(label="Série, retorno e retorno ao quadrado", type="filepath")
            fig_acf = gr.Image(label="ACF e PACF", type="filepath")
        with gr.Tab("Previsão e métricas"):
            nota = gr.Markdown()
            metricas = gr.Dataframe(label="Métricas no teste", interactive=False)
            fig_comp = gr.Image(label="Previsões no teste", type="filepath")
            fig_treino = gr.Image(label="Curva de aprendizado da rede", type="filepath")
        with gr.Tab("Risco (VaR)"):
            var = gr.Dataframe(label="Backtest do VaR EWMA de 1 dia", interactive=False)
            fig_var = gr.Image(label="P&L e VaR", type="filepath")
        with gr.Tab("Projeção GBM"):
            gbm = gr.Dataframe(label="Cenários de um ano", interactive=False)
            fig_gbm = gr.Image(label="Leque de projeção", type="filepath")
        with gr.Tab("Resumo e arquivos"):
            resumo = gr.Markdown()
            arquivos = gr.File(label="Arquivos gerados", file_count="multiple")

    botao.click(
        executar,
        inputs=[ativo, inicio, fim_treino, fim_validacao, rede, criterio, etapas, grupos_metricas, atualizar],
        outputs=[log, periodos, raiz, momentos, dependencia, fig_serie, fig_acf,
                 metricas, nota, fig_treino, fig_comp, var, fig_var, gbm, fig_gbm, resumo, arquivos],
    )


if __name__ == "__main__":
    app.launch(**({"theme": TEMA} if TEMA_NO_LAUNCH else {}))
