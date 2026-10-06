"""Diagnóstico estatístico da série: raiz unitária, momentos e dependência temporal."""
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.stattools import adfuller, kpss

import config


def testes_raiz_unitaria(serie: pd.Series, nome: str, tendencia: bool) -> dict:
    """ADF (H0: raiz unitária) e KPSS (H0: estacionária) com a mesma especificação.

    tendencia=True inclui constante e tendência linear ('ct'), adequado ao log-preço;
    tendencia=False inclui só constante ('c'), adequado ao log-retorno.
    """
    regressao = "ct" if tendencia else "c"
    with warnings.catch_warnings():
        # KPSS avisa quando trunca o p-valor fora da tabela; versões recentes do
        # statsmodels também avisam sobre a mudança futura no tipo de retorno
        warnings.simplefilter("ignore")
        adf = adfuller(serie.dropna(), regression=regressao, autolag="AIC")
        kpss_estat, kpss_p, *_ = kpss(serie.dropna(), regression=regressao, nlags="auto")
    adf_estat, adf_p = adf[0], adf[1]

    if adf_p < 0.05 and kpss_p > 0.05:
        leitura = "estacionaria"
    elif adf_p >= 0.05 and kpss_p <= 0.05:
        leitura = "raiz unitaria"
    else:
        leitura = "inconclusivo"

    return {"serie": nome, "especificacao": regressao,
            "ADF_estat": adf_estat, "ADF_p": adf_p,
            "KPSS_estat": kpss_estat, "KPSS_p": kpss_p, "leitura": leitura}


def tabela_raiz_unitaria(base: pd.DataFrame) -> pd.DataFrame:
    """O log-preço é testado com e sem tendência: a conclusão pode mudar entre as
    duas especificações, e mostrar ambas evita escolher a que convém."""
    linhas = [
        testes_raiz_unitaria(base["log_preco"], "log-preço", tendencia=False),
        testes_raiz_unitaria(base["log_preco"], "log-preço", tendencia=True),
        testes_raiz_unitaria(base["log_ret"], "log-retorno", tendencia=False),
    ]
    return pd.DataFrame(linhas).set_index("serie")


def momentos_retorno(log_ret: pd.Series) -> pd.Series:
    """Momentos diários e anualizados do log-retorno.

    A média do log-retorno estima a deriva do log-preço, mu - sigma^2/2. O mu
    aritmético do GBM é recuperado somando sigma^2/2.
    """
    n = len(log_ret)
    media_d = log_ret.mean()
    dp_d = log_ret.std()
    jb = stats.jarque_bera(log_ret)
    anos = n / config.DIAS_UTEIS_ANO
    return pd.Series({
        "pregoes": n,
        "media_diaria": media_d,
        "dp_diario": dp_d,
        "assimetria": stats.skew(log_ret),
        "curtose_excesso": stats.kurtosis(log_ret),
        "jarque_bera_p": jb.pvalue,
        "deriva_log_anual": media_d * config.DIAS_UTEIS_ANO,
        "erro_padrao_deriva_anual": dp_d * np.sqrt(config.DIAS_UTEIS_ANO) / np.sqrt(anos),
        "vol_anual": dp_d * np.sqrt(config.DIAS_UTEIS_ANO),
        "mu_aritmetico_anual": media_d * config.DIAS_UTEIS_ANO + 0.5 * (dp_d ** 2) * config.DIAS_UTEIS_ANO,
    })


def dependencia_temporal(log_ret: pd.Series, vol_ewma: pd.Series, defasagens=(5, 10, 20)) -> pd.DataFrame:
    """Ljung-Box no retorno (previsibilidade da média), no retorno padronizado pela
    volatilidade EWMA e no retorno ao quadrado (agrupamento de volatilidade), mais o
    ARCH-LM de Engle.

    O Ljung-Box supõe variância constante; com agrupamento de volatilidade ele
    rejeita a ausência de autocorrelação com frequência maior que a nominal. O
    retorno padronizado, r_t / sigma_t, remove a maior parte desse efeito.
    """
    padronizado = (log_ret / vol_ewma).iloc[config.JANELA_INICIAL_EWMA:]
    lb_ret = acorr_ljungbox(log_ret, lags=list(defasagens), return_df=True)
    lb_pad = acorr_ljungbox(padronizado, lags=list(defasagens), return_df=True)
    lb_quad = acorr_ljungbox(log_ret ** 2, lags=list(defasagens), return_df=True)
    tabela = pd.DataFrame({
        "LB_retorno_p": lb_ret["lb_pvalue"].values,
        "LB_retorno_padronizado_p": lb_pad["lb_pvalue"].values,
        "LB_retorno2_p": lb_quad["lb_pvalue"].values,
    }, index=pd.Index(defasagens, name="defasagens"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        arch = het_arch(log_ret, nlags=10)
    tabela["ARCH_LM_10_p"] = arch[1]
    return tabela
