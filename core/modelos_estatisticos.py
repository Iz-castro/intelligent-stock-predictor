"""Referências estatísticas para previsão um passo à frente do log-retorno.

Todas as funções devolvem a previsão do retorno do dia t feita com informação até
t-1, indexada pela data t. O preço previsto é P_{t-1} * exp(retorno previsto).
"""
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

import config


def previsao_passeio_aleatorio(indice_alvo: pd.DatetimeIndex) -> pd.Series:
    """Passeio aleatório sem deriva: o melhor palpite para amanhã é o preço de hoje."""
    return pd.Series(0.0, index=indice_alvo, name="passeio_aleatorio")


def previsao_com_deriva(log_ret_treino: pd.Series, indice_alvo: pd.DatetimeIndex) -> pd.Series:
    """Passeio aleatório com deriva: retorno previsto igual à média do treino."""
    return pd.Series(log_ret_treino.mean(), index=indice_alvo, name="passeio_com_deriva")


def selecionar_arima(log_preco_treino: pd.Series, criterio: str | None = None,
                     max_p: int = 5, max_q: int = 5) -> dict:
    """Escolhe (p, 1, q) para o log-preço com auto_arima (d fixo em 1, busca stepwise).

    Um ARIMA(p,1,q) no log-preço equivale a um ARMA(p,q) no log-retorno; a
    constante do ARMA corresponde à deriva do passeio aleatório. Em séries de
    retorno a superfície do critério de informação é bem plana, e o AICc pode
    escolher ordens diferentes entre downloads quase idênticos; o BIC, mais
    rigoroso, tende a ficar no passeio aleatório, ARIMA(0,1,0).
    """
    import pmdarima as pm

    criterio = (criterio or config.CRITERIO_ARIMA).lower()
    modelo = pm.auto_arima(
        log_preco_treino.values,
        d=1, start_p=0, start_q=0, max_p=max_p, max_q=max_q,
        seasonal=False, stepwise=True, information_criterion=criterio,
        suppress_warnings=True, error_action="ignore",
    )
    p, d, q = modelo.order
    return {"ordem": (p, d, q), "com_constante": bool(modelo.with_intercept),
            "criterio": {"aicc": "AICc", "aic": "AIC", "bic": "BIC"}.get(criterio, criterio), "valor_criterio": float(getattr(modelo, criterio)())}


def previsao_arima(log_ret: pd.Series, mascara_treino: pd.Series, ordem: tuple,
                   com_constante: bool, indice_alvo: pd.DatetimeIndex):
    """Estima o ARMA(p,q) do retorno só no treino e filtra a série completa com os
    parâmetros congelados. Cada previsão usa os dados reais até o dia anterior,
    sem reestimar coeficientes com informação do período avaliado."""
    p, _, q = ordem
    tendencia = "c" if com_constante else "n"
    serie = log_ret.reset_index(drop=True)  # SARIMAX dispensa frequência de calendário

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ajuste = SARIMAX(serie[mascara_treino.values], order=(p, 0, q), trend=tendencia).fit(disp=False)
        filtro = SARIMAX(serie, order=(p, 0, q), trend=tendencia).filter(ajuste.params)

    previsto = pd.Series(np.asarray(filtro.get_prediction().predicted_mean), index=log_ret.index)
    return previsto.reindex(indice_alvo).rename("arima"), ajuste
