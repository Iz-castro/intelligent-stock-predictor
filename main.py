"""Executa o estudo completo pela linha de comando (padrão: PETR4).

Uso:
    python main.py                  # baixa dados, roda tudo com GRU
    python main.py --ticker VALE3   # outro papel da B3 (o sufixo .SA é incluído)
    python main.py --rede lstm      # troca a rede por LSTM
    python main.py --sem-rede       # só modelos estatísticos (dispensa TensorFlow)
    python main.py --offline        # usa o cache em data/ em vez de baixar
"""
import argparse
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # silencia o log informativo do TensorFlow

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import config  # noqa: E402
import pipeline  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Estudo de previsão e risco de papéis da B3")
    parser.add_argument("--ticker", default=config.TICKER)
    parser.add_argument("--rede", choices=["gru", "lstm"], default="gru")
    parser.add_argument("--sem-rede", action="store_true")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    config.definir_ativo(args.ticker)

    base, rotulos = pipeline.etapa_dados(atualizar=not args.offline)
    diag = pipeline.etapa_diagnostico(base, rotulos)
    aval = pipeline.etapa_avaliacao(base, rotulos, tipo_rede=None if args.sem_rede else args.rede)
    risc = pipeline.etapa_risco(base, rotulos)
    proj = pipeline.etapa_projecao(base, risc["vol_ewma_proximo_pregao"])
    print()
    print(pipeline.gerar_resumo(base, rotulos, diag, aval, risc, proj))


if __name__ == "__main__":
    main()
