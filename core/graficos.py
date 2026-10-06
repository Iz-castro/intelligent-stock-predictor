"""Gráficos do projeto com identidade visual única."""
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf

import config

COR_REAL = "#1A1A1A"
COR_ARIMA = "#009E73"
COR_REDE = "#D55E00"
COR_PA = "#7A7A7A"
COR_IC = "#0072B2"
COR_ALERTA = "#CC0000"
CORES_PERIODO = {"treino": "#FFFFFF", "validacao": "#F2F2F2", "teste": "#E6EEF7"}
def _rodape() -> str:
    return f"Izael Castro | Fonte: Yahoo Finance ({config.TICKER}, preços ajustados)"

plt.rcParams.update({
    "figure.dpi": 110, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.labelsize": 10, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False,
})


def _finalizar(fig, nome: str):
    fig.text(0.99, 0.005, _rodape(), ha="right", va="bottom", fontsize=7, color="0.45", style="italic")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    caminho = config.DIR_RESULTADOS / f"{nome}.png"
    fig.savefig(caminho, dpi=150, bbox_inches="tight")
    return caminho


def _sombrear_periodos(ax, rotulos: pd.Series):
    for nome in ("validacao", "teste"):
        datas = rotulos.index[rotulos.values == nome]
        if len(datas):
            ax.axvspan(datas[0], datas[-1], color=CORES_PERIODO[nome], zorder=0)


def painel_serie(base: pd.DataFrame, rotulos: pd.Series):
    fig, eixos = plt.subplots(3, 1, figsize=(12, 8.5), sharex=True)
    eixos[0].plot(base.index, base["Fechamento"], color=COR_REAL, lw=1)
    eixos[0].set_yscale("log")
    eixos[0].set_ylabel("R$ (escala log)")
    eixos[0].set_title(f"{config.NOME_ATIVO}: preço ajustado, log-retorno e retorno ao quadrado")
    eixos[1].plot(base.index, 100 * base["log_ret"], color=COR_IC, lw=0.5)
    eixos[1].axhline(0, color="0.5", lw=0.7)
    eixos[1].set_ylabel("Log-retorno (%)")
    eixos[2].plot(base.index, 1e4 * base["log_ret"] ** 2, color=COR_REDE, lw=0.5)
    eixos[2].set_ylabel("Retorno² (%²)")
    for ax in eixos:
        _sombrear_periodos(ax, rotulos)
    y_topo = eixos[0].get_ylim()[1]
    for nome, rotulo in (("treino", "TREINO"), ("validacao", "VALID."), ("teste", "TESTE")):
        datas = rotulos.index[rotulos.values == nome]
        eixos[0].text(datas[0], y_topo, f" {rotulo}", va="top", fontsize=8, color="0.4", fontweight="bold")
    return fig, _finalizar(fig, "01_serie_retorno_volatilidade")


def painel_acf_pacf(base: pd.DataFrame, defasagens: int = 40):
    series = {
        "Log-preço": base["log_preco"],
        "Log-retorno": base["log_ret"],
        "Log-retorno ao quadrado": base["log_ret"] ** 2,
    }
    fig, eixos = plt.subplots(3, 2, figsize=(12, 9))
    for linha, (nome, serie) in enumerate(series.items()):
        plot_acf(serie, lags=defasagens, zero=False, ax=eixos[linha, 0], title="", color=COR_IC,
                 vlines_kwargs={"colors": COR_IC})
        plot_pacf(serie, lags=defasagens, zero=False, ax=eixos[linha, 1], title="", method="ywm",
                  color=COR_REDE, vlines_kwargs={"colors": COR_REDE})
        eixos[linha, 0].set_title(f"ACF: {nome}")
        eixos[linha, 1].set_title(f"PACF: {nome}")
        maximos = (np.abs(acf(serie, nlags=defasagens)[1:]).max(),
                   np.abs(pacf(serie, nlags=defasagens, method="ywm")[1:]).max())
        for ax, maximo in zip(eixos[linha], maximos):
            limite = min(1.05, max(0.12, 1.2 * maximo))
            ax.set_ylim(-limite, limite)
    return fig, _finalizar(fig, "02_acf_pacf")


def curva_treino(historico: dict, tipo: str):
    fig, ax = plt.subplots(figsize=(9, 4))
    epocas = np.arange(1, len(historico["loss"]) + 1)
    ax.plot(epocas, historico["loss"], color=COR_REAL, marker="o", ms=3, label="Treino")
    ax.plot(epocas, historico["val_loss"], color=COR_REDE, marker="o", ms=3, label="Validação")
    melhor = int(np.argmin(historico["val_loss"])) + 1
    ax.axvline(melhor, color="0.5", ls=":", lw=1)
    ax.text(melhor, ax.get_ylim()[1], f" melhor época: {melhor}", va="top", fontsize=8, color="0.4")
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.set_title(f"Curva de aprendizado: {tipo.upper()} (alvo padronizado)")
    ax.set(xlabel="Época", ylabel="MSE")
    ax.legend()
    return fig, _finalizar(fig, f"03_curva_treino_{tipo}")


def comparacao_teste(previsoes: dict, real_ret: pd.Series, preco: pd.Series, tipo_rede: str,
                     ultimos: int = 60):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1.6, 1]})

    datas = real_ret.index[-ultimos:]
    preco_ant = preco.shift(1).reindex(datas)
    ax1.plot(datas, preco.reindex(datas), color=COR_REAL, lw=1.8, marker="o", ms=3.5, label="Real", zorder=4)
    # o passeio aleatório é desenhado por último, tracejado, para ficar visível
    # sobre as outras previsões, que praticamente coincidem com ele
    estilos = {"arima": (COR_ARIMA, "ARIMA", "-", 1.6),
               tipo_rede: (COR_REDE, tipo_rede.upper(), "-", 1.6),
               "passeio_aleatorio": (COR_PA, "Passeio aleatório (preço de ontem)", (0, (4, 3)), 1.2)}
    for nome, (cor, rotulo, ls, lw) in estilos.items():
        if nome in previsoes:
            ax1.plot(datas, preco_ant * np.exp(previsoes[nome].reindex(datas)), color=cor, lw=lw, ls=ls,
                     label=rotulo)
    ax1.set_title(f"Últimos {ultimos} pregões do teste: as previsões seguem o preço com um dia de atraso")
    ax1.set_ylabel("R$")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m/%y"))

    lim = 100 * np.nanpercentile(np.abs(real_ret), 99)
    for nome, cor in (("arima", COR_ARIMA), (tipo_rede, COR_REDE)):
        if nome in previsoes:
            ax2.scatter(100 * previsoes[nome].reindex(real_ret.index), 100 * real_ret, s=8, alpha=0.45,
                        color=cor, label=estilos[nome][1])
    ax2.plot([-lim, lim], [-lim, lim], color="0.6", ls=":", lw=1, label="Previsão perfeita")
    ax2.axhline(0, color="0.7", lw=0.6)
    ax2.axvline(0, color="0.7", lw=0.6)
    ax2.set_xlim(-lim / 4, lim / 4)
    ax2.set_ylim(-lim, lim)
    ax2.set_title("Retorno previsto x realizado (teste)")
    ax2.set(xlabel="Previsto (%)", ylabel="Realizado (%)")
    ax2.legend(loc="upper left", fontsize=8)
    return fig, _finalizar(fig, f"04_comparacao_teste_{tipo_rede or 'estatisticos'}")


def backtest_var(serie: pd.DataFrame, resumo: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(serie.index, -serie["perda"] / 1e3, color="0.55", lw=0.6, label="P&L diário")
    for rotulo, cor, ls in (("VaR95", COR_IC, "-"), ("VaR99", COR_ALERTA, "--")):
        ax.plot(serie.index, -serie[rotulo] / 1e3, color=cor, lw=1.1, ls=ls,
                label=f"-{rotulo} EWMA ({resumo.loc[rotulo, 'violacoes']} violações, "
                      f"{resumo.loc[rotulo, 'taxa_obs_%']:.1f}% vs {resumo.loc[rotulo, 'taxa_esperada_%']:.0f}%)")
        violou = serie[f"violou_{rotulo}"]
        ax.scatter(serie.index[violou], -serie["perda"][violou] / 1e3, color=cor, s=14, zorder=5)
    ax.axhline(0, color="0.7", lw=0.6)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%y"))
    ax.set_title(f"Backtest do VaR de 1 dia (EWMA, lambda = {config.LAMBDA_EWMA}): carteira de "
                 f"R$ {config.VALOR_CARTEIRA / 1e6:.0f} milhão em {config.NOME_ATIVO}")
    ax.set_ylabel("R$ mil")
    ax.legend(loc="lower left", fontsize=8, ncol=3)
    return fig, _finalizar(fig, "05_backtest_var")


def projecao_gbm(preco: pd.Series, analitico: pd.DataFrame, trajetorias: np.ndarray, rotulo_cenario: str,
                 historico_dias: int = 504, n_amostra: int = 40):
    fig, ax = plt.subplots(figsize=(13, 5.5))
    hist = preco.iloc[-historico_dias:]
    ax.plot(hist.index, hist, color=COR_REAL, lw=1.1, label="Histórico")

    idx = np.random.default_rng(config.SEMENTE).choice(trajetorias.shape[1], n_amostra, replace=False)
    ax.plot(analitico.index, trajetorias[:, idx], color=COR_IC, lw=0.4, alpha=0.18)
    ax.fill_between(analitico.index, analitico["IC95_inf"], analitico["IC95_sup"], color=COR_IC, alpha=0.07)
    ax.fill_between(analitico.index, analitico["IC68_inf"], analitico["IC68_sup"], color=COR_IC, alpha=0.12)
    for nivel, ls in (("IC95", "--"), ("IC68", ":")):
        ax.plot(analitico.index, analitico[f"{nivel}_inf"], color=COR_IC, lw=1, ls=ls)
        ax.plot(analitico.index, analitico[f"{nivel}_sup"], color=COR_IC, lw=1, ls=ls, label=nivel)
    ax.plot(analitico.index, analitico["mediana"], color=COR_REDE, lw=1.6, label="Mediana")
    ax.plot(analitico.index, analitico["media"], color=COR_REDE, lw=1, ls="--", label="Média")

    fim = analitico.iloc[-1]
    for chave in ("IC95_inf", "IC68_inf", "mediana", "media", "IC68_sup", "IC95_sup"):
        ax.annotate(f"R$ {fim[chave]:.2f}", xy=(analitico.index[-1], fim[chave]), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=8, color="0.25")
    ax.plot(hist.index[-1], hist.iloc[-1], marker="o", ms=10, mfc="none", mec=COR_REAL)
    ax.set_title(f"Projeção GBM de {len(analitico) - 1} pregões: {rotulo_cenario}")
    ax.set_ylabel("R$")
    ax.legend(loc="upper left", fontsize=8, ncol=5)
    ax.set_xlim(hist.index[0], analitico.index[-1] + pd.Timedelta(days=45))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%y"))
    return fig, _finalizar(fig, "06_projecao_gbm")
