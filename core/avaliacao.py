"""Métricas de previsão um passo à frente, sempre comparadas ao passeio aleatório."""
import numpy as np
import pandas as pd
from scipy import stats


def diebold_mariano(erro_modelo: np.ndarray, erro_ref: np.ndarray, h: int = 1) -> tuple[float, float]:
    """Teste de Diebold-Mariano com correção de Harvey, Leybourne e Newbold.

    Perda quadrática; d_t = e_modelo^2 - e_ref^2. Estatística negativa indica que o
    modelo erra menos que a referência. H0: mesma acurácia esperada.
    """
    d = np.asarray(erro_modelo) ** 2 - np.asarray(erro_ref) ** 2
    n = len(d)
    d_med = d.mean()
    gamma = [np.sum((d[k:] - d_med) * (d[:n - k] - d_med)) / n for k in range(h)]
    var_d = (gamma[0] + 2 * sum(gamma[1:])) / n
    if var_d <= 0:
        return np.nan, np.nan
    dm = d_med / np.sqrt(var_d)
    correcao = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_hln = dm * correcao
    p = 2 * stats.t.sf(abs(dm_hln), df=n - 1)
    return dm_hln, p


def avaliar(previsoes: dict, real_ret: pd.Series, preco: pd.Series,
            log_ret_treino: pd.Series, referencia: str = "passeio_aleatorio") -> pd.DataFrame:
    """Tabela de métricas no período de real_ret.

    previsoes: {nome: Series de log-retorno previsto indexada pela data-alvo}.
    preco: série de fechamento completa (para reconstruir o preço previsto).
    """
    datas = real_ret.index
    preco_ant = preco.shift(1).reindex(datas)
    preco_real = preco.reindex(datas)
    escala_mase = log_ret_treino.abs().mean()  # MAE do passeio aleatório no treino

    erro_ref = (real_ret - previsoes[referencia].reindex(datas)).to_numpy()
    rmse_ref = np.sqrt(np.mean(erro_ref ** 2))
    sse_ref = np.sum(erro_ref ** 2)

    taxa_alta = (real_ret > 0).mean()
    linhas = []
    for nome, prev in previsoes.items():
        prev = prev.reindex(datas)
        erro = (real_ret - prev).to_numpy()
        preco_prev = preco_ant * np.exp(prev)
        erro_preco = (preco_real - preco_prev).to_numpy()

        # acerto direcional só onde o modelo indica direção
        com_direcao = prev.to_numpy() != 0
        if com_direcao.sum() > 0:
            acertos = int((np.sign(prev.to_numpy()[com_direcao]) == np.sign(real_ret.to_numpy()[com_direcao])).sum())
            n_dir = int(com_direcao.sum())
            acerto_dir = 100 * acertos / n_dir
            binom_p = stats.binomtest(acertos, n_dir, 0.5).pvalue
        else:
            acerto_dir, binom_p = np.nan, np.nan

        if nome == referencia:
            dm, dm_p = np.nan, np.nan
        else:
            dm, dm_p = diebold_mariano(erro, erro_ref)

        ss_tot_preco = np.sum((preco_real - preco_real.mean()) ** 2)
        linhas.append({
            "modelo": nome,
            "RMSE_preco": np.sqrt(np.mean(erro_preco ** 2)),
            "MAE_preco": np.mean(np.abs(erro_preco)),
            "MAPE_preco_%": 100 * np.mean(np.abs(erro_preco / preco_real.to_numpy())),
            "R2_preco": 1 - np.sum(erro_preco ** 2) / ss_tot_preco,
            "RMSE_ret_%": 100 * np.sqrt(np.mean(erro ** 2)),
            "MASE_ret": np.mean(np.abs(erro)) / escala_mase,
            "U_Theil": np.sqrt(np.mean(erro ** 2)) / rmse_ref,
            "R2_fora_amostra_%": 100 * (1 - np.sum(erro ** 2) / sse_ref),
            "acerto_direcional_%": acerto_dir,
            "binomial_p": binom_p,
            "DM_estat": dm,
            "DM_p": dm_p,
        })
    tabela = pd.DataFrame(linhas).set_index("modelo")
    tabela.attrs["taxa_dias_alta_%"] = 100 * taxa_alta
    tabela.attrs["pregoes"] = len(datas)
    return tabela
