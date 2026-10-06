"""Volatilidade EWMA (RiskMetrics), VaR paramétrico com backtest e projeção por GBM."""
import numpy as np
import pandas as pd
from scipy import stats

import config


def z_bicaudal(nivel: str) -> float:
    """IC68 -> z = 1; IC95 -> z = norm.ppf(0,975)."""
    return {"IC68": 1.0, "IC95": stats.norm.ppf(0.975)}[nivel]


def volatilidade_ewma(log_ret: pd.Series, lam: float = config.LAMBDA_EWMA,
                      janela_inicial: int = config.JANELA_INICIAL_EWMA) -> pd.Series:
    """sigma_t^2 = lambda * sigma_{t-1}^2 + (1 - lambda) * r_{t-1}^2.

    O valor na data t é a previsão para o dia t, feita só com retornos até t-1.
    A variância inicial é a amostral dos primeiros 'janela_inicial' retornos.
    """
    r = log_ret.to_numpy()
    var = np.empty_like(r)
    var[0] = np.var(r[:janela_inicial])
    for t in range(1, len(r)):
        var[t] = lam * var[t - 1] + (1 - lam) * r[t - 1] ** 2
    return pd.Series(np.sqrt(var), index=log_ret.index, name="vol_ewma")


# ---------------------------------------------------------------- VaR e backtest
def kupiec_pof(violacoes: np.ndarray, nivel: float) -> tuple[float, float]:
    """Teste de proporção de falhas (Kupiec, 1995). H0: taxa de violação = 1 - nivel."""
    n, x = len(violacoes), int(violacoes.sum())
    p, p_hat = 1 - nivel, x / n
    if x in (0, n):
        lr = -2 * (n * np.log(1 - p) if x == 0 else n * np.log(p))
    else:
        lr = -2 * ((n - x) * np.log(1 - p) + x * np.log(p)
                   - (n - x) * np.log(1 - p_hat) - x * np.log(p_hat))
    return lr, 1 - stats.chi2.cdf(lr, df=1)


def christoffersen_independencia(violacoes: np.ndarray) -> tuple[float, float]:
    """Teste de independência (Christoffersen, 1998). H0: uma violação hoje não
    altera a chance de violação amanhã, isto é, as falhas não chegam agrupadas."""
    v = violacoes.astype(int)
    anterior, atual = v[:-1], v[1:]
    n00 = np.sum((anterior == 0) & (atual == 0))
    n01 = np.sum((anterior == 0) & (atual == 1))
    n10 = np.sum((anterior == 1) & (atual == 0))
    n11 = np.sum((anterior == 1) & (atual == 1))
    pi0 = n01 / max(n00 + n01, 1)
    pi1 = n11 / max(n10 + n11, 1)
    pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)

    def _log(x):
        return np.log(x) if x > 0 else 0.0

    log_h0 = (n00 + n10) * _log(1 - pi) + (n01 + n11) * _log(pi)
    log_h1 = n00 * _log(1 - pi0) + n01 * _log(pi0) + n10 * _log(1 - pi1) + n11 * _log(pi1)
    lr = -2 * (log_h0 - log_h1)
    return lr, 1 - stats.chi2.cdf(lr, df=1)


def backtest_var(log_ret: pd.Series, vol: pd.Series, mascara: pd.Series,
                 niveis=config.NIVEIS_VAR, valor: float = config.VALOR_CARTEIRA):
    """VaR de 1 dia (normal, média zero, convenção RiskMetrics) comparado à perda
    realizada no mesmo dia. Como vol[t] usa retornos até t-1, não há defasagem
    extra nem informação do próprio dia na previsão."""
    r, s = log_ret[mascara.values], vol[mascara.values]
    perda = -valor * (np.exp(r) - 1)
    serie = pd.DataFrame({"perda": perda})
    linhas = []
    for nivel in niveis:
        z = stats.norm.ppf(nivel)
        var_t = valor * z * s
        violou = (perda > var_t).to_numpy()
        lr_uc, p_uc = kupiec_pof(violou, nivel)
        lr_ind, p_ind = christoffersen_independencia(violou)
        lr_cc = lr_uc + lr_ind
        p_cc = 1 - stats.chi2.cdf(lr_cc, df=2)
        rotulo = f"VaR{int(nivel * 100)}"
        serie[rotulo] = var_t
        serie[f"violou_{rotulo}"] = violou
        linhas.append({"nivel": rotulo, "dias": len(violou), "violacoes": int(violou.sum()),
                       "taxa_obs_%": 100 * violou.mean(), "taxa_esperada_%": 100 * (1 - nivel),
                       "Kupiec_p": p_uc, "Christoffersen_ind_p": p_ind, "cobertura_condicional_p": p_cc})
    return pd.DataFrame(linhas).set_index("nivel"), serie


# --------------------------------------------------------------------- GBM
def projecao_gbm(s0: float, deriva_diaria: float, vol_diaria: float, data_inicial,
                 horizonte: int = config.HORIZONTE_PROJECAO, n_traj: int = config.N_TRAJETORIAS,
                 semente: int = config.SEMENTE):
    """Projeção por Movimento Browniano Geométrico.

    deriva_diaria é a média do log-retorno, que já corresponde a mu - sigma^2/2.
    Por isso o log-preço evolui como ln S_t = ln S_0 + deriva * t + sigma * W_t, sem
    nova subtração de sigma^2/2. Os limites dos intervalos saem dos quantis do
    log-preço (normal) levados de volta à escala de preço, o que respeita a
    assimetria da lognormal.
    """
    t = np.arange(0, horizonte + 1)
    datas = pd.bdate_range(start=data_inicial, periods=horizonte + 1)

    analitico = pd.DataFrame(index=datas)
    analitico["mediana"] = s0 * np.exp(deriva_diaria * t)
    analitico["media"] = s0 * np.exp((deriva_diaria + 0.5 * vol_diaria ** 2) * t)
    for nivel in config.NIVEIS_IC:
        z = z_bicaudal(nivel)
        analitico[f"{nivel}_inf"] = s0 * np.exp(deriva_diaria * t - z * vol_diaria * np.sqrt(t))
        analitico[f"{nivel}_sup"] = s0 * np.exp(deriva_diaria * t + z * vol_diaria * np.sqrt(t))

    gerador = np.random.default_rng(semente)
    choques = gerador.normal(deriva_diaria, vol_diaria, size=(horizonte, n_traj))
    log_traj = np.vstack([np.zeros(n_traj), np.cumsum(choques, axis=0)])
    trajetorias = s0 * np.exp(log_traj)

    st = trajetorias[-1]
    resumo = {
        "S0": s0,
        "mediana_analitica": analitico["mediana"].iloc[-1],
        "media_analitica": analitico["media"].iloc[-1],
        "mediana_simulada": float(np.median(st)),
        "media_simulada": float(st.mean()),
        "P(S_T < S0)_analitica": float(stats.norm.cdf(-deriva_diaria * horizonte / (vol_diaria * np.sqrt(horizonte)))),
        "P(S_T < S0)_simulada": float((st < s0).mean()),
    }
    for nivel in config.NIVEIS_IC:
        resumo[f"{nivel}_inf"] = analitico[f"{nivel}_inf"].iloc[-1]
        resumo[f"{nivel}_sup"] = analitico[f"{nivel}_sup"].iloc[-1]
    return analitico, trajetorias, resumo
