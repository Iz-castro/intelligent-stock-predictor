# Manual do Intelligent Stock Predictor

**Como os conteúdos da disciplina de Aplicações em Finanças, Controladoria e Compliance entraram no projeto**

Izael Castro | MBA em Inteligência Artificial e Analytics Aplicadas a Negócios (FGV) | disciplina ministrada pelo Prof. Alvaro Villarinho

---

## 1. Ponto de partida

O projeto nasceu antes da disciplina, como um preditor de preços de ações com redes LSTM e GRU, e chegou à primeira aula com a pergunta que acabou acompanhando o curso: se o preço de uma ação se comporta como uma série aleatória, o que um modelo de aprendizado profundo consegue extrair dele? Na versão original, a GRU prevendo o preço da PETR4 tinha R² de 0,996, mas RMSE praticamente igual ao do modelo que apenas repete o preço do dia anterior (0,5373 contra 0,5377) e acerto direcional de 51%. A rede havia aprendido a persistência, e foi ao longo das aulas que ficou claro por quê.

Reconstruí o projeto aplicando, etapa por etapa, o que vimos em sala. O resultado é um estudo de previsão e risco que roda por uma interface Gradio, na qual escolho o papel, as datas de corte, a rede, as etapas e as métricas exibidas, além da linha de comando e de um notebook para o Colab.

## 2. Mapa: o que veio de cada aula

| Conteúdo da disciplina | Onde aparece no projeto |
|:--|:--|
| Coleta via Yahoo Finance (notebook de estacionariedade, AR e EWMA) | `core/dados.py`: download com `yfinance`, preços ajustados, cache local |
| Hipóteses de Gauss-Markov, autocovariância, ACF e PACF | `core/diagnostico.py` e gráfico `02_acf_pacf.png` |
| Estacionariedade, ADF e KPSS, média constante e não constante | Testes no log-preço (com e sem tendência) e no log-retorno |
| Modelos ingênuos e métricas de erro (aulas de Holt-Winters) | Passeio aleatório como referência obrigatória; MASE, U de Theil, MAPE |
| Box-Jenkins e `auto_arima` (AirPassengers, varejo) | `core/modelos_estatisticos.py`: ARIMA escolhido só no treino |
| Divisão em treino e teste e avaliação fora da amostra | Treino 2015 a 2022, validação 2023, teste 2024 a 2026 |
| RiskMetrics, EWMA com lambda de 0,94, VaR e teste de Kupiec | `core/risco.py` e gráfico `05_backtest_var.png` |
| Movimento Browniano Geométrico, lema de Itô e Monte Carlo (planilha do Excel, Hull 14 e 15) | Projeção de um ano com intervalos de 68% e 95% |
| Intervalos de confiança com z = 1 (68%) e z = 1,96 (95%) | Convenção usada em todo o projeto |

## 3. As etapas, com o conceito por trás de cada uma

### 3.1 Dados

Os preços diários ajustados por dividendos e JCP vêm do Yahoo Finance, como no notebook da aula, o que substituiu o download manual de CSV que eu fazia no Investing. Dois cuidados aparecem nesta etapa. O Yahoo devolve algumas linhas incompletas (abertura, máxima e mínima zeradas, volume nulo), quase todas em feriados da B3, e essas linhas são descartadas. Além disso, optei por não reindexar a série para dias úteis, já que preencher feriados com o preço anterior cria retornos zero que não existiram e reduz artificialmente a variância.

### 3.2 Diagnóstico da série

Esta etapa aplica diretamente as aulas sobre Gauss-Markov e autocovariância. O log-preço tem raiz unitária (ADF com p = 0,88 na especificação com constante), enquanto o log-retorno é estacionário pelos dois testes, exatamente o padrão do slide de média constante e média não constante: a ACF do preço decai muito devagar e a do retorno cai para dentro da faixa logo nas primeiras defasagens.

O ponto que considero mais interessante para discutir está no Ljung-Box. No retorno bruto, ele rejeita a ausência de autocorrelação (p = 0,0001), o que sugeriria algum padrão previsível. O teste, porém, supõe variância constante, a hipótese de homocedasticidade de Gauss-Markov, e a PETR4 tem agrupamento de volatilidade evidente (ARCH-LM com p próximo de zero, curtose em excesso de 14,9). Quando divido o retorno pela volatilidade EWMA do dia, o p-valor sobe para 0,14, de modo que a "autocorrelação" vinha dos períodos turbulentos (2015 e 2016, março de 2020), e não de um padrão explorável na média.

### 3.3 Previsão um passo à frente

Quatro previsões do retorno do pregão seguinte, todas feitas só com informação até o dia anterior, são comparadas no mesmo período de teste:

- **Passeio aleatório**, retorno zero, isto é, o preço de amanhã igual ao de hoje, que é a referência que qualquer modelo precisa superar;
- **Passeio com deriva**, que usa a média do retorno no treino;
- **ARIMA**, com ordem escolhida pelo `auto_arima` (critério AICc, como em aula) e coeficientes estimados apenas no treino;
- **GRU ou LSTM**, agora prevendo o log-retorno, com quatro entradas estacionárias (retorno, retorno ao quadrado, volatilidade EWMA e volume relativo) e parada antecipada medida no período de validação.

A troca do alvo é a principal mudança em relação à versão antiga. Prever o nível de uma série com raiz unitária leva a rede a copiar o último valor; prever o retorno obriga o modelo a mostrar se enxerga algo além disso.

### 3.4 Métricas

Segui a lógica das aulas de séries temporais, de sempre medir o modelo contra um ingênuo. Além de RMSE e MAPE no preço, o projeto calcula o U de Theil (erro do modelo dividido pelo erro do passeio aleatório), o MASE, o R² fora da amostra, o acerto direcional com teste binomial contra 50% e o teste de Diebold-Mariano, que verifica se a diferença de erro entre dois modelos é estatisticamente significativa.

### 3.5 Risco: EWMA e VaR

A volatilidade segue o RiskMetrics, com sigma²(t) = 0,94 sigma²(t-1) + 0,06 r²(t-1), e o VaR de 1 dia a 95% e 99% é aplicado a uma carteira de R$ 1 milhão. O backtest compara a perda de cada dia com o VaR calculado na véspera e usa o teste de Kupiec, visto em aula, complementado pelo de Christoffersen, que verifica se as violações chegam agrupadas.

### 3.6 Projeção por GBM

A projeção de um ano aplica a fórmula da planilha de abertura da última aula, S(t+dt) = S(t) exp[(mu - sigma²/2) dt + sigma raiz(dt) Z], com 10 mil trajetórias e conferência pela solução analítica da lognormal (Hull, capítulos 14 e 15). Dois detalhes de implementação merecem menção. A média do log-retorno estimada nos dados já corresponde a mu - sigma²/2, então ela entra diretamente como deriva do log-preço, sem nova correção de Itô. Os limites dos intervalos de 68% e 95% vêm dos quantis do log-preço levados de volta à escala de preço, o que respeita a assimetria da lognormal; um intervalo do tipo média ± z desvios padrão no preço subestimaria a cauda superior e exageraria a inferior.

## 4. Resultados com a PETR4 (dados até 02/10/2026)

| Modelo | U de Theil | R² do preço | Acerto direcional | Diebold-Mariano (p) |
|:--|--:|--:|--:|--:|
| Passeio aleatório | 1,000 | 0,9925 | sem direção | referência |
| Passeio com deriva | 0,998 | 0,9925 | 53,5% | 0,35 |
| ARIMA(1,1,0) | 1,006 | 0,9924 | 46,6% | 0,02 (pior) |
| GRU | 1,006 | 0,9924 | 48,8% | 0,12 |

Nenhum modelo supera o passeio aleatório nos 690 pregões de teste, e o R² do preço fica em 0,99 para todos, inclusive para o próprio passeio aleatório, o que mostra por que essa métrica enganava na versão antiga. O ARIMA escolhido no treino captura justamente a autocorrelação espúria dos períodos turbulentos e erra mais que a referência com significância estatística. A GRU convergiu para uma previsão quase constante, com desvio padrão de 0,12% ao dia contra 1,60% do retorno realizado.

No risco, o VaR de 95% passou nos dois testes (4,2% de violações). O de 99% registrou 1,9% de violações, com Kupiec p = 0,04 e Christoffersen p = 0,02, o que indica caudas mais pesadas que a normal e violações agrupadas. Na projeção, o resultado depende muito dos parâmetros: com a volatilidade histórica de 45% ao ano, o IC 95% para daqui a um ano vai de R$ 27,50 a R$ 162,28; com a volatilidade EWMA atual, de 29%, ele se estreita para R$ 38,15 a R$ 116,97.

## 5. Como explorar a interface

O roteiro abaixo percorre os principais resultados em poucos minutos.

1. Rodar `python app.py` e abrir http://127.0.0.1:7860.
2. Escolher PETR4, manter as datas padrão, GRU e AICc, marcar todas as etapas e executar (leva menos de um minuto).
3. Mostrar na aba **Diagnóstico** a ACF do preço contra a do retorno, e o Ljung-Box bruto contra o padronizado.
4. Na aba **Previsão e métricas**, mostrar o gráfico em que todas as previsões seguem o preço com um dia de atraso, e a tabela com U de Theil próximo de 1.
5. Trocar o critério para BIC e executar de novo: o ARIMA vira (0,1,0), o próprio passeio aleatório.
6. Na aba **Risco (VaR)**, comparar 95% e 99%; na aba **Projeção GBM**, comparar os cenários de volatilidade.
7. Se houver tempo, repetir com outro papel (VALE3, por exemplo) para mostrar que a conclusão se mantém.

## 6. Próximos passos

Três caminhos ficaram abertos a partir dos resultados. O primeiro é o tratamento da volatilidade na projeção: na PETR4, trocar a volatilidade histórica longa pela do EWMA corrente reduz a largura do intervalo de 95% em cerca de 40%, e um GARCH, que combina reação rápida com reversão à média, tende a ser o meio-termo natural entre os dois. O segundo é o VaR de 99%, reprovado no backtest, que pede uma distribuição de caudas mais pesadas (t de Student) ou um VaR histórico filtrado pela volatilidade. O terceiro é redirecionar a rede neural para prever a volatilidade em vez do retorno, já que o retorno ao quadrado tem dependência temporal forte e o retorno em si quase nenhuma.

## 7. Limitações que reconheço

O teste cobre um único período (2024 a 2026), e uma avaliação com origem móvel ao longo de vários anos seria mais robusta. A rede usa uma única semente, e redes recorrentes variam entre inicializações. A ordem escolhida pelo AICc pode mudar entre downloads quase idênticos do Yahoo, porque a superfície do critério é muito plana em séries de retorno, e por isso o cache local garante a reprodução dos números. O GBM supõe volatilidade constante ao longo do ano, simplificação que a própria diferença entre os cenários histórico e EWMA deixa evidente.
