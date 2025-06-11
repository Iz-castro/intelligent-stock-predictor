import pandas as pd
import matplotlib.pyplot as plt
import os

RESULTS_DIR = "results"
CSV_PATH = os.path.join(RESULTS_DIR, "metricas_modelos.csv")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Ler as métricas salvas
df = pd.read_csv(CSV_PATH)

# Detecta se existe coluna Modelo; se não, cria a partir de Papel
if "Modelo" not in df.columns:
    novos_papeis = []
    modelos = []
    for p in df["Papel"]:
        if "_gru" in p:
            novos_papeis.append(p.replace("_gru", ""))
            modelos.append("gru")
        elif "_lstm" in p:
            novos_papeis.append(p.replace("_lstm", ""))
            modelos.append("lstm")
        else:
            novos_papeis.append(p)
            modelos.append("lstm")  # Assumindo que o padrão antigo é LSTM
    df["Papel"] = novos_papeis
    df["Modelo"] = modelos

# Gráficos comparativos
def plot_metric(df, metric, ylabel, filename):
    plt.figure(figsize=(8, 5))
    modelos = df["Modelo"].unique()
    for papel in df["Papel"].unique():
        subset = df[df["Papel"] == papel]
        plt.bar(subset["Modelo"], subset[metric], label=papel)
    plt.title(f"Comparativo de {ylabel}")
    plt.xlabel("Modelo")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename))
    plt.close()

# RMSE
if "RMSE" in df.columns:
    plot_metric(df, "RMSE", "RMSE", "grafico_rmse.png")
# MAE
if "MAE" in df.columns:
    plot_metric(df, "MAE", "MAE", "grafico_mae.png")
# MAPE
if "MAPE (%)" in df.columns:
    plot_metric(df, "MAPE (%)", "MAPE (%)", "grafico_mape.png")
# R2
if "R2" in df.columns:
    plot_metric(df, "R2", "R²", "grafico_r2.png")
# Directional Accuracy
if "Directional_Acc (%)" in df.columns:
    plot_metric(df, "Directional_Acc (%)", "Directional Accuracy (%)", "grafico_directional_acc.png")

# Gerar Markdown resumido (exemplo para README)
def gerar_markdown_resumo(df):
    md = "# Comparativo de Modelos\n\n"
    for papel in df["Papel"].unique():
        md += f"## {papel}\n\n"
        tabela = df[df["Papel"] == papel][["Modelo"] + [col for col in df.columns if col not in ["Papel", "Modelo", "DataHora"]]]
        md += tabela.to_markdown(index=False)
        md += "\n\n"
        if "RMSE" in df.columns:
            md += f"![RMSE](results/grafico_rmse.png)\n"
        if "MAE" in df.columns:
            md += f"![MAE](results/grafico_mae.png)\n"
        if "MAPE (%)" in df.columns:
            md += f"![MAPE](results/grafico_mape.png)\n"
        if "R2" in df.columns:
            md += f"![R²](results/grafico_r2.png)\n"
        if "Directional_Acc (%)" in df.columns:
            md += f"![Directional Accuracy](results/grafico_directional_acc.png)\n"
        md += "\n"
    return md

md = gerar_markdown_resumo(df)
with open(os.path.join(RESULTS_DIR, "metricas_resumo.md"), "w", encoding="utf-8") as f:
    f.write(md)

print("Gráficos e markdown gerados em 'results/'. Pronto para usar no README!")
