"""Coleta via Yahoo Finance, limpeza e montagem das séries usadas no projeto."""
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import config

COLUNAS_PT = {
    "Open": "Abertura",
    "High": "Maxima",
    "Low": "Minima",
    "Close": "Fechamento",
    "Volume": "Volume",
}


def _caminho_cache(ticker: str):
    return config.DIR_DADOS / f"{ticker.replace('.', '_')}.csv"


def _limpar(dados: pd.DataFrame) -> pd.DataFrame:
    """Padroniza colunas e remove registros parciais do Yahoo.

    O Yahoo costuma devolver, para o pregão corrente ou recém-encerrado, uma linha
    com abertura, máxima e mínima zeradas e volume nulo. Essa linha distorce o
    último retorno e é descartada.
    """
    if isinstance(dados.columns, pd.MultiIndex):
        dados.columns = dados.columns.get_level_values(0)
    dados = dados.rename(columns=COLUNAS_PT)[list(COLUNAS_PT.values())]
    dados.index = pd.to_datetime(dados.index).tz_localize(None)
    dados.index.name = "Data"

    parcial = (dados[["Abertura", "Maxima", "Minima"]] <= 0).any(axis=1) | (dados["Volume"] <= 0)
    if parcial.any():
        datas = ", ".join(d.strftime("%d/%m/%Y") for d in dados.index[parcial])
        print(f"[dados] {parcial.sum()} registro(s) parcial(is) removido(s): {datas}")
    # arredondar garante que a série em memória seja idêntica à lida do cache,
    # o que torna reproduzíveis as execuções com e sem download
    precos = ["Abertura", "Maxima", "Minima", "Fechamento"]
    dados[precos] = dados[precos].round(6)
    return dados.loc[~parcial].sort_index()


def _download_yahoo(ticker: str, inicio: str, fim: str | None) -> pd.DataFrame:
    """Download via yfinance.

    Nos testes, depois que o TensorFlow era importado no mesmo processo (o que
    acontece na interface quando uma rede já foi treinada), o yfinance passava a
    falhar com ImpersonateError do curl_cffi. Nesse caso o download roda num
    processo Python separado, que não carrega o TensorFlow.
    """
    if "tensorflow" not in sys.modules:
        import yfinance as yf
        return yf.download(ticker, start=inicio, end=fim, auto_adjust=True, progress=False)

    codigo = ("import sys, yfinance as yf; "
              "d = yf.download(sys.argv[1], start=sys.argv[2], end=sys.argv[3] or None, "
              "auto_adjust=True, progress=False); d.to_pickle(sys.argv[4])")
    with tempfile.TemporaryDirectory() as pasta:
        destino = Path(pasta) / "bruto.pkl"
        subprocess.run([sys.executable, "-c", codigo, ticker, inicio, fim or "", str(destino)],
                       check=True, capture_output=True, text=True, timeout=300)
        return pd.read_pickle(destino)


def baixar_precos(ticker: str | None = None, inicio: str | None = None,
                  fim: str | None = None, atualizar: bool = True) -> pd.DataFrame:
    """Baixa preços diários ajustados (dividendos e JCP) e mantém um cache local.

    Com atualizar=False, ou se o download falhar, usa o último cache salvo em data/.
    Não há reindexação para dias úteis: feriados da B3 ficam fora da série, em vez
    de entrarem como retorno zero.
    """
    ticker = ticker or config.TICKER
    inicio = inicio or config.DATA_INICIO
    cache = _caminho_cache(ticker)

    if atualizar:
        try:
            brutos = _download_yahoo(ticker, inicio, fim)
            if brutos.empty:
                raise ValueError("download vazio")
            dados = _limpar(brutos)
            dados.to_csv(cache)
            print(f"[dados] {ticker}: {len(dados)} pregões de "
                  f"{dados.index[0]:%d/%m/%Y} a {dados.index[-1]:%d/%m/%Y} (cache atualizado)")
            return dados
        except Exception as erro:  # rede indisponível, mudança na API etc.
            if not cache.exists():
                raise RuntimeError(f"Falha no download de {ticker} e não há cache local.") from erro
            warnings.warn(f"Falha no download ({erro}); usando cache local.")

    dados = pd.read_csv(cache, index_col="Data", parse_dates=True)
    dados = dados.loc[dados.index >= pd.Timestamp(inicio)]
    print(f"[dados] {ticker}: {len(dados)} pregões lidos do cache "
          f"({dados.index[0]:%d/%m/%Y} a {dados.index[-1]:%d/%m/%Y})")
    return dados


def montar_series(dados: pd.DataFrame) -> pd.DataFrame:
    """Acrescenta log-preço e log-retorno, r_t = ln(P_t) - ln(P_{t-1})."""
    base = dados.copy()
    base["log_preco"] = np.log(base["Fechamento"])
    base["log_ret"] = base["log_preco"].diff()
    return base.dropna(subset=["log_ret"])


def rotular_periodos(indice: pd.DatetimeIndex) -> pd.Series:
    """Classifica cada data em treino, validacao ou teste conforme config."""
    rotulo = pd.Series("teste", index=indice)
    rotulo[indice <= pd.Timestamp(config.FIM_VALIDACAO)] = "validacao"
    rotulo[indice <= pd.Timestamp(config.FIM_TREINO)] = "treino"
    return rotulo


def resumo_periodos(indice: pd.DatetimeIndex) -> pd.DataFrame:
    rotulo = rotular_periodos(indice)
    linhas = []
    for nome in ("treino", "validacao", "teste"):
        datas = indice[rotulo.values == nome]
        if len(datas) == 0:
            raise ValueError(f"Período de {nome} vazio: revise as datas de corte em config.py "
                             f"(dados de {indice.min():%d/%m/%Y} a {indice.max():%d/%m/%Y}).")
        linhas.append({"periodo": nome, "inicio": datas.min().date(),
                       "fim": datas.max().date(), "pregoes": len(datas)})
    return pd.DataFrame(linhas).set_index("periodo")
