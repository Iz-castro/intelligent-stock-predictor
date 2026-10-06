"""Parâmetros centrais do projeto. Alterar aqui propaga para todas as etapas."""
from pathlib import Path

# Ativo e período (mesma fonte e início usados no notebook da aula)
TICKER = "PETR4.SA"
NOME_ATIVO = "PETR4"
DATA_INICIO = "2015-01-01"

# Divisão cronológica: o teste começa no primeiro pregão após FIM_VALIDACAO
FIM_TREINO = "2022-12-31"
FIM_VALIDACAO = "2023-12-31"

# ARIMA: critério de seleção da ordem no auto_arima (busca stepwise, d = 1)
# "aicc" segue a convenção das aulas; "bic" penaliza mais e tende ao passeio aleatório
CRITERIO_ARIMA = "aicc"

# Rede neural
JANELA = 60            # pregões de histórico em cada entrada
UNIDADES = 32          # neurônios da camada recorrente
DROPOUT = 0.2
EPOCAS_MAX = 200
PACIENCIA = 15         # early stopping, medido no período de validação
TAMANHO_LOTE = 64
TAXA_APRENDIZADO = 1e-3
SEMENTE = 42

# Risco e projeção
LAMBDA_EWMA = 0.94     # fator de decaimento RiskMetrics para dados diários
JANELA_INICIAL_EWMA = 30
NIVEIS_VAR = (0.95, 0.99)
VALOR_CARTEIRA = 1_000_000.0
DIAS_UTEIS_ANO = 252
HORIZONTE_PROJECAO = 252
N_TRAJETORIAS = 10_000

# Intervalos de confiança bicaudais (mesma convenção das aulas)
# IC68 -> z = 1 (quantil norm.cdf(1) = 0,8413) | IC95 -> z = norm.ppf(0,975) = 1,96
NIVEIS_IC = ("IC68", "IC95")

# Pastas (os resultados ficam em results/<ATIVO>/)
RAIZ = Path(__file__).resolve().parent
DIR_DADOS = RAIZ / "data"
DIR_MODELOS = RAIZ / "models"
DIR_RESULTADOS = RAIZ / "results" / NOME_ATIVO

for _pasta in (DIR_DADOS, DIR_MODELOS, DIR_RESULTADOS):
    _pasta.mkdir(parents=True, exist_ok=True)


def definir_ativo(ticker: str, inicio: str | None = None, fim_treino: str | None = None,
                  fim_validacao: str | None = None) -> None:
    """Troca o ativo e, opcionalmente, as datas de corte em tempo de execução.

    Usado pela interface Gradio e pelo main.py; os módulos leem estes valores no
    momento da chamada, então a troca vale para todas as etapas seguintes.
    """
    global TICKER, NOME_ATIVO, DATA_INICIO, FIM_TREINO, FIM_VALIDACAO, DIR_RESULTADOS
    ticker = ticker.strip().upper()
    if "." not in ticker and ticker[-1:].isdigit():
        ticker += ".SA"  # papéis da B3 no Yahoo levam o sufixo .SA
    TICKER = ticker
    NOME_ATIVO = ticker.split(".")[0]
    DATA_INICIO = inicio or DATA_INICIO
    FIM_TREINO = fim_treino or FIM_TREINO
    FIM_VALIDACAO = fim_validacao or FIM_VALIDACAO
    DIR_RESULTADOS = RAIZ / "results" / NOME_ATIVO
    DIR_RESULTADOS.mkdir(parents=True, exist_ok=True)
