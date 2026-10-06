"""Rede recorrente (GRU ou LSTM) para prever o log-retorno do pregão seguinte.

Mudanças em relação à versão anterior do projeto:
- o alvo é o log-retorno, e não o preço; prever o nível de uma série com raiz
  unitária leva a rede a copiar o último valor (persistência);
- entradas estacionárias e poucas (retorno, retorno ao quadrado, volatilidade EWMA
  e volume relativo), no lugar de dezenas de indicadores derivados do preço;
- padronização ajustada só no treino e early stopping medido na validação, sem
  tocar no período de teste;
- perda MSE simples; a perda direcional antiga comparava amostras vizinhas de um
  lote embaralhado, o que tornava a penalidade aleatória.
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import config

FEATURES = ["log_ret", "log_ret_quadrado", "vol_ewma", "volume_relativo"]


def montar_features(base: pd.DataFrame, vol_ewma: pd.Series) -> pd.DataFrame:
    """Variáveis de entrada do dia s, todas conhecidas no fechamento de s."""
    f = pd.DataFrame(index=base.index)
    f["log_ret"] = base["log_ret"]
    f["log_ret_quadrado"] = base["log_ret"] ** 2
    # vol_ewma[s] é a previsão para s feita em s-1; o valor atualizado com r_s
    # entra como informação disponível ao fim de s
    lam = config.LAMBDA_EWMA
    f["vol_ewma"] = np.sqrt(lam * vol_ewma ** 2 + (1 - lam) * base["log_ret"] ** 2)
    f["volume_relativo"] = np.log(base["Volume"] / base["Volume"].rolling(21).mean())
    return f.dropna()


def montar_janelas(features: pd.DataFrame, alvo: pd.Series, janela: int = config.JANELA):
    """X[i] = features dos 'janela' pregões anteriores à data i; y[i] = retorno em i."""
    valores = features.to_numpy(dtype="float32")
    alvo = alvo.reindex(features.index)
    X, y, datas = [], [], []
    for i in range(janela, len(features)):
        X.append(valores[i - janela:i])
        y.append(alvo.iloc[i])
        datas.append(features.index[i])
    return np.stack(X), np.asarray(y, dtype="float32"), pd.DatetimeIndex(datas)


def construir_rede(tipo: str, n_features: int, janela: int = config.JANELA):
    from tensorflow import keras

    camada = keras.layers.GRU if tipo == "gru" else keras.layers.LSTM
    modelo = keras.Sequential([
        keras.layers.Input(shape=(janela, n_features)),
        camada(config.UNIDADES),
        keras.layers.Dropout(config.DROPOUT),
        keras.layers.Dense(1),
    ])
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=config.TAXA_APRENDIZADO), loss="mse")
    return modelo


def treinar_rede(features: pd.DataFrame, log_ret: pd.Series, rotulos: pd.Series,
                 tipo: str = "gru", salvar: bool = True, verbose: int = 0):
    """Treina no período de treino, para pela validação e prevê validação e teste.

    Retorna (previsao, historico, modelo), com a previsão do log-retorno indexada
    pela data-alvo e cobrindo validação e teste.
    """
    from tensorflow import keras

    keras.utils.set_random_seed(config.SEMENTE)

    # padronização das entradas ajustada só com o treino
    escala_x = StandardScaler().fit(features[rotulos.reindex(features.index) == "treino"])
    f_pad = pd.DataFrame(escala_x.transform(features), index=features.index, columns=features.columns)

    X, y, datas = montar_janelas(f_pad, log_ret)
    periodo = rotulos.reindex(datas).to_numpy()
    tr, va = periodo == "treino", periodo == "validacao"

    # alvo em unidades de desvio padrão do treino, para estabilizar o gradiente
    escala_y = float(y[tr].std())
    modelo = construir_rede(tipo, X.shape[2])
    parada = keras.callbacks.EarlyStopping(monitor="val_loss", patience=config.PACIENCIA,
                                           restore_best_weights=True)
    historico = modelo.fit(
        X[tr], y[tr] / escala_y,
        validation_data=(X[va], y[va] / escala_y),
        epochs=config.EPOCAS_MAX, batch_size=config.TAMANHO_LOTE,
        callbacks=[parada], verbose=verbose,
    )

    fora_treino = ~tr
    previsto = modelo.predict(X[fora_treino], verbose=0).ravel() * escala_y
    previsao = pd.Series(previsto, index=datas[fora_treino], name=tipo)

    if salvar:
        nome = f"{tipo}_{config.NOME_ATIVO}"
        modelo.save(config.DIR_MODELOS / f"rede_{nome}.keras")
        joblib.dump(escala_x, config.DIR_MODELOS / f"escala_entradas_{nome}.joblib")
        meta = {"tipo": tipo, "features": FEATURES, "janela": config.JANELA,
                "escala_alvo": escala_y, "epocas": len(historico.history["loss"]),
                "fim_treino": config.FIM_TREINO, "fim_validacao": config.FIM_VALIDACAO}
        (config.DIR_MODELOS / f"meta_{nome}.json").write_text(json.dumps(meta, indent=2))

    return previsao, historico.history, modelo
