"""Etapas do estudo, chamadas pelo main.py, pelo app.py (Gradio) e pelo notebook.

Cada etapa devolve um dicionário com as tabelas e a lista de gráficos salvos em
config.DIR_RESULTADOS, que é results/<ATIVO>/.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from core import avaliacao, dados, diagnostico, graficos, modelos_estatisticos, risco


def _exibir_ou_fechar(figura, mostrar: bool):
    if mostrar:
        plt.show()
    else:
        plt.close(figura)


def etapa_dados(atualizar: bool = True):
    brutos = dados.baixar_precos(atualizar=atualizar)
    base = dados.montar_series(brutos)
    rotulos = dados.rotular_periodos(base.index)
    dados.resumo_periodos(base.index)  # valida se treino, validação e teste têm dados
    return base, rotulos


def etapa_diagnostico(base: pd.DataFrame, rotulos: pd.Series, mostrar: bool = False):
    raiz = diagnostico.tabela_raiz_unitaria(base)
    momentos = diagnostico.momentos_retorno(base["log_ret"])
    dependencia = diagnostico.dependencia_temporal(base["log_ret"], risco.volatilidade_ewma(base["log_ret"]))

    raiz.to_csv(config.DIR_RESULTADOS / "diagnostico_raiz_unitaria.csv")
    momentos.to_csv(config.DIR_RESULTADOS / "diagnostico_momentos.csv", header=["valor"])
    dependencia.to_csv(config.DIR_RESULTADOS / "diagnostico_dependencia.csv")

    figuras = []
    for figura, caminho in (graficos.painel_serie(base, rotulos), graficos.painel_acf_pacf(base)):
        figuras.append(caminho)
        _exibir_ou_fechar(figura, mostrar)
    return {"raiz_unitaria": raiz, "momentos": momentos, "dependencia": dependencia, "figuras": figuras}


def etapa_avaliacao(base: pd.DataFrame, rotulos: pd.Series, tipo_rede: str | None = "gru",
                    criterio_arima: str | None = None, mostrar: bool = False, verbose: int = 0):
    """Previsão um passo à frente do log-retorno no período de teste."""
    log_ret = base["log_ret"]
    treino = rotulos == "treino"
    datas_teste = base.index[rotulos.values == "teste"]
    real = log_ret.reindex(datas_teste)

    previsoes = {
        "passeio_aleatorio": modelos_estatisticos.previsao_passeio_aleatorio(datas_teste),
        "passeio_com_deriva": modelos_estatisticos.previsao_com_deriva(log_ret[treino.values], datas_teste),
    }

    escolha = modelos_estatisticos.selecionar_arima(base["log_preco"][treino.values], criterio_arima)
    previsoes["arima"], ajuste_arima = modelos_estatisticos.previsao_arima(
        log_ret, treino, escolha["ordem"], escolha["com_constante"], datas_teste)
    print(f"[avaliacao] ARIMA escolhido no treino: {escolha['ordem']} "
          f"{'com' if escolha['com_constante'] else 'sem'} constante "
          f"({escolha['criterio']} {escolha['valor_criterio']:.1f})")

    historico = None
    if tipo_rede:
        from core import rede
        vol = risco.volatilidade_ewma(log_ret)
        features = rede.montar_features(base, vol)
        prev_rede, historico, _ = rede.treinar_rede(features, log_ret, rotulos, tipo=tipo_rede, verbose=verbose)
        previsoes[tipo_rede] = prev_rede.reindex(datas_teste)
        print(f"[avaliacao] {tipo_rede.upper()} treinada em {len(historico['loss'])} épocas "
              f"(melhor na validação: {int(np.argmin(historico['val_loss'])) + 1})")

    tabela = avaliacao.avaliar(previsoes, real, base["Fechamento"], log_ret[treino.values])
    tabela.to_csv(config.DIR_RESULTADOS / "metricas_teste.csv")
    pd.DataFrame(previsoes).assign(real=real).to_csv(config.DIR_RESULTADOS / "previsoes_teste.csv")

    resultados_graficos = []
    if historico:
        resultados_graficos.append(graficos.curva_treino(historico, tipo_rede))
    resultados_graficos.append(graficos.comparacao_teste(previsoes, real, base["Fechamento"], tipo_rede or ""))
    figuras = []
    for figura, caminho in resultados_graficos:
        figuras.append(caminho)
        _exibir_ou_fechar(figura, mostrar)

    return {"tabela": tabela, "previsoes": previsoes, "arima": escolha,
            "arima_params": ajuste_arima.params, "historico_rede": historico, "figuras": figuras}


def vol_ewma_proximo_pregao(log_ret: pd.Series) -> float:
    """Volatilidade EWMA prevista para o pregão seguinte ao último observado."""
    vol = risco.volatilidade_ewma(log_ret)
    lam = config.LAMBDA_EWMA
    return float(np.sqrt(lam * vol.iloc[-1] ** 2 + (1 - lam) * log_ret.iloc[-1] ** 2))


def etapa_risco(base: pd.DataFrame, rotulos: pd.Series, mostrar: bool = False):
    """VaR EWMA de 1 dia, avaliado no período de teste."""
    vol = risco.volatilidade_ewma(base["log_ret"])
    mascara = pd.Series(rotulos.values == "teste", index=base.index)
    resumo, serie = risco.backtest_var(base["log_ret"], vol, mascara)
    resumo.to_csv(config.DIR_RESULTADOS / "backtest_var.csv")

    figura, caminho = graficos.backtest_var(serie, resumo)
    _exibir_ou_fechar(figura, mostrar)
    return {"resumo": resumo, "vol_ewma_proximo_pregao": vol_ewma_proximo_pregao(base["log_ret"]),
            "figuras": [caminho]}


def etapa_projecao(base: pd.DataFrame, vol_ewma_proxima: float | None = None, mostrar: bool = False):
    """GBM de um ano a partir do último fechamento, em três cenários de parâmetros."""
    log_ret = base["log_ret"]
    if vol_ewma_proxima is None:
        vol_ewma_proxima = vol_ewma_proximo_pregao(log_ret)
    s0 = float(base["Fechamento"].iloc[-1])
    deriva = float(log_ret.mean())
    vol_hist = float(log_ret.std())
    cenarios = {
        "deriva histórica, vol histórica": (deriva, vol_hist),
        "deriva histórica, vol EWMA atual": (deriva, vol_ewma_proxima),
        "deriva zero, vol histórica": (0.0, vol_hist),
    }

    linhas = []
    principal = None
    for nome, (mu_d, sig_d) in cenarios.items():
        analitico, trajetorias, resumo = risco.projecao_gbm(s0, mu_d, sig_d, base.index[-1])
        linhas.append({"cenario": nome, "deriva_log_anual_%": 100 * mu_d * config.DIAS_UTEIS_ANO,
                       "vol_anual_%": 100 * sig_d * np.sqrt(config.DIAS_UTEIS_ANO), **resumo})
        if principal is None:
            principal = (nome, analitico, trajetorias)

    tabela = pd.DataFrame(linhas).set_index("cenario")
    tabela.to_csv(config.DIR_RESULTADOS / "projecao_gbm_cenarios.csv")
    principal[1].to_csv(config.DIR_RESULTADOS / "projecao_gbm_trajetoria_analitica.csv")

    figura, caminho = graficos.projecao_gbm(base["Fechamento"], principal[1], principal[2], principal[0])
    _exibir_ou_fechar(figura, mostrar)
    return {"cenarios": tabela, "figuras": [caminho]}


def gerar_resumo(base, rotulos, diag=None, aval=None, risc=None, proj=None) -> str:
    """Resumo numérico em markdown (results/<ATIVO>/resumo.md) com as etapas executadas."""
    linhas = [
        f"# Resumo numérico: {config.NOME_ATIVO}",
        "",
        f"Gerado em {pd.Timestamp.now():%d/%m/%Y %H:%M}. Dados de {base.index[0]:%d/%m/%Y} "
        f"a {base.index[-1]:%d/%m/%Y}.",
        "",
        "## Períodos",
        "",
        dados.resumo_periodos(base.index).to_markdown(),
        "",
    ]
    if diag:
        linhas += [
            "## Raiz unitária", "", diag["raiz_unitaria"].round(4).to_markdown(), "",
            "## Momentos do log-retorno (amostra completa)", "",
            diag["momentos"].to_frame("valor").round(5).to_markdown(), "",
            "## Dependência temporal (p-valores)", "", diag["dependencia"].round(4).to_markdown(), "",
        ]
    if aval:
        tab = aval["tabela"]
        colunas = ["RMSE_preco", "MAPE_preco_%", "R2_preco", "RMSE_ret_%", "MASE_ret", "U_Theil",
                   "R2_fora_amostra_%", "acerto_direcional_%", "binomial_p", "DM_estat", "DM_p"]
        linhas += [
            f"## Previsão um passo à frente no teste ({tab.attrs['pregoes']} pregões; "
            f"{tab.attrs['taxa_dias_alta_%']:.1f}% de dias de alta)", "",
            f"ARIMA selecionado no treino ({aval['arima']['criterio']}): {aval['arima']['ordem']} "
            f"{'com' if aval['arima']['com_constante'] else 'sem'} constante.", "",
            tab[colunas].round(4).to_markdown(), "",
        ]
    if risc:
        linhas += ["## Backtest do VaR EWMA de 1 dia (teste)", "", risc["resumo"].round(4).to_markdown(), ""]
    if proj:
        linhas += ["## Projeção GBM de um ano", "", proj["cenarios"].round(2).to_markdown(), ""]

    texto = "\n".join(linhas)
    (config.DIR_RESULTADOS / "resumo.md").write_text(texto, encoding="utf-8")
    return texto
