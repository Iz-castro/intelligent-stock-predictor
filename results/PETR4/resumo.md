# Resumo numérico: PETR4

Gerado em 05/10/2026 23:12. Dados de 04/01/2000 a 02/10/2026.

## Períodos

| periodo   | inicio     | fim        |   pregoes |
|:----------|:-----------|:-----------|----------:|
| treino    | 2000-01-04 | 2022-12-29 |      5631 |
| validacao | 2023-01-02 | 2023-12-28 |       248 |
| teste     | 2024-01-02 | 2026-10-02 |       690 |

## Raiz unitária

| serie       | especificacao   |   ADF_estat |   ADF_p |   KPSS_estat |   KPSS_p | leitura       |
|:------------|:----------------|------------:|--------:|-------------:|---------:|:--------------|
| log-preço   | c               |     -0.4722 |  0.8973 |       8.3968 |     0.01 | raiz unitaria |
| log-preço   | ct              |     -1.7119 |  0.7456 |       1.3653 |     0.01 | raiz unitaria |
| log-retorno | c               |    -29.3189 |  0      |       0.1168 |     0.1  | estacionaria  |

## Momentos do log-retorno (amostra completa)

|                          |      valor |
|:-------------------------|-----------:|
| pregoes                  | 6569       |
| media_diaria             |    0.00059 |
| dp_diario                |    0.02619 |
| assimetria               |   -0.66815 |
| curtose_excesso          |   10.8685  |
| jarque_bera_p            |    0       |
| deriva_log_anual         |    0.14894 |
| erro_padrao_deriva_anual |    0.08143 |
| vol_anual                |    0.41573 |
| mu_aritmetico_anual      |    0.23536 |

## Dependência temporal (p-valores)

|   defasagens |   LB_retorno_p |   LB_retorno_padronizado_p |   LB_retorno2_p |   ARCH_LM_10_p |
|-------------:|---------------:|---------------------------:|----------------:|---------------:|
|            5 |         0.1803 |                     0.1206 |               0 |              0 |
|           10 |         0.001  |                     0.1091 |               0 |              0 |
|           20 |         0.0119 |                     0.3145 |               0 |              0 |

## Previsão um passo à frente no teste (690 pregões; 53.5% de dias de alta)

ARIMA selecionado no treino (AICc): (0, 1, 0) sem constante.

| modelo             |   RMSE_preco |   MAPE_preco_% |   R2_preco |   RMSE_ret_% |   MASE_ret |   U_Theil |   R2_fora_amostra_% |   acerto_direcional_% |   binomial_p |   DM_estat |     DM_p |
|:-------------------|-------------:|---------------:|-----------:|-------------:|-----------:|----------:|--------------------:|----------------------:|-------------:|-----------:|---------:|
| passeio_aleatorio  |       0.5417 |         1.173  |     0.9925 |       1.606  |     0.6143 |    1      |              0      |              nan      |     nan      |   nan      | nan      |
| passeio_com_deriva |       0.5409 |         1.1707 |     0.9925 |       1.6038 |     0.6128 |    0.9987 |              0.2691 |               53.4783 |       0.0735 |    -1.2725 |   0.2036 |
| arima              |       0.5417 |         1.173  |     0.9925 |       1.606  |     0.6143 |    1      |              0      |              nan      |     nan      |   nan      | nan      |
| lstm               |       0.5413 |         1.1763 |     0.9925 |       1.6074 |     0.6158 |    1.0009 |             -0.171  |               45.7971 |       0.0299 |     0.4043 |   0.6861 |

## Backtest do VaR EWMA de 1 dia (teste)

| nivel   |   dias |   violacoes |   taxa_obs_% |   taxa_esperada_% |   Kupiec_p |   Christoffersen_ind_p |   cobertura_condicional_p |
|:--------|-------:|------------:|-------------:|------------------:|-----------:|-----------------------:|--------------------------:|
| VaR95   |    690 |          29 |       4.2029 |                 5 |     0.3238 |                 0.1498 |                    0.2178 |
| VaR99   |    690 |          13 |       1.8841 |                 1 |     0.0376 |                 0.0202 |                    0.0078 |

## Projeção GBM de um ano

| cenario                          |   deriva_log_anual_% |   vol_anual_% |    S0 |   mediana_analitica |   media_analitica |   mediana_simulada |   media_simulada |   P(S_T < S0)_analitica |   P(S_T < S0)_simulada |   IC68_inf |   IC68_sup |   IC95_inf |   IC95_sup |
|:---------------------------------|---------------------:|--------------:|------:|--------------------:|------------------:|-------------------:|-----------------:|------------------------:|-----------------------:|-----------:|-----------:|-----------:|-----------:|
| deriva histórica, vol histórica  |                14.89 |         41.57 | 51.17 |               59.39 |             64.75 |              59.49 |            65.02 |                    0.36 |                   0.36 |      39.19 |      90    |      26.29 |     134.14 |
| deriva histórica, vol EWMA atual |                14.89 |         28.58 | 51.17 |               59.39 |             61.86 |              59.46 |            62.04 |                    0.3  |                   0.3  |      44.62 |      79.04 |      33.92 |     103.99 |
| deriva zero, vol histórica       |                 0    |         41.57 | 51.17 |               51.17 |             55.79 |              51.26 |            56.02 |                    0.5  |                   0.5  |      33.76 |      77.55 |      22.65 |     115.58 |
