
# ✅ TODO - Reformulado: Intelligent Stock Predictor (Foco em LSTM Multivariado)

## 🔧 Objetivo Geral
Consolidar o projeto como um preditor robusto de ações baseado em LSTM multivariado, com foco em dados históricos da PETR4 (e futuros suportes a outras ações), apresentando um README claro, aplicação interativa via Gradio e potencial para atualizações contínuas.

---

## ✅ SEÇÃO 1: Introdução e Propósito
- [A] Reescrever a introdução explicando:
  - O problema resolvido (previsão de ações com LSTM)
  - O valor da abordagem multivariada
  - Por que LSTM é apropriado para séries temporais financeiras

---

## ✅ SEÇÃO 2: Resultados Quantitativos

Nesta seção apresentamos o desempenho do modelo **LSTM Multivariado**, treinado individualmente para cada papel com dados históricos obtidos do site *br.investing.com*. A avaliação foi feita com o conjunto de teste (20% dos dados) e validada com métricas padronizadas.

### 📊 Métricas de Desempenho (RMSE / MAE)

| Papel   | RMSE   | MAE    | Última Avaliação       |
|---------|--------|--------|------------------------|
| PETR4   | 1.52   | 1.11   | 2025-05-15 22:10:03     |
| BBAS3   | 1.22   | 0.95   | 2025-05-16 10:37:42     |
| VALE3   | 1.73   | 1.28   | 2025-05-16 11:02:25     |

> *Os valores apresentados foram obtidos a partir da previsão sobre o conjunto de teste após o treino individual de cada ativo.*

---

### 📈 Gráficos de Comparação Real vs Previsto

#### 🔸 PETR4
![PETR4](results/comparativo_teste_multivariado_PETR4.png)

#### 🔸 BBAS3
![BBAS3](results/comparativo_teste_multivariado_BBAS3.png)

#### 🔸 VALE3
![VALE3](results/comparativo_teste_multivariado_VALE3.png)

---

### 📉 Curvas de Treinamento por Papel

Cada modelo foi treinado com validação automática e *early stopping* para evitar overfitting. Abaixo, as curvas de erro MSE durante o treinamento:

- `treinamento_multivariado_PETR4.png`
- `treinamento_multivariado_BBAS3.png`
- `treinamento_multivariado_VALE3.png`

---

### 📌 Observações

- O modelo mostrou **alta capacidade de generalização** nos três papéis testados, mesmo com variações de comportamento (commodities, bancário e estatal).
- As métricas foram salvas automaticamente via `metricas_modelos.csv` e podem ser auditadas a qualquer momento.


---

## ✅ SEÇÃO 3: Fonte dos Dados
- [A] Informar que os dados foram obtidos manualmente do site: https://br.investing.com/
- [A] Adicionar TODO:
  - 🚧 Atualizar o sistema para ingestão automática via API (Alpha Vantage, Twelve Data, ou B3 oficial)
  - Permitir atualizações diárias automatizadas

---

## ✅ SEÇÃO 4: Resultados Visuais
- [x] Incluir:
  - `comparativo_teste_multivariado.png`
  - `treinamento_multivariado.png`

---

## ✅ SEÇÃO 5: Instruções de Execução
- [A] Explicar como rodar localmente:
  - Clonar repositório
  - Criar venv
  - Instalar requirements
  - Executar treinamento
  - Rodar comparações e previsões
  - Rodar Gradio

---

## ✅ SEÇÃO 6: Aplicação Web
- [A] Indicar que a aplicação está em Gradio
- [A] TODO:
  - 🚀 Avaliar hospedagem (Gradio Spaces, HuggingFace, Streamlit Cloud, Vercel, Render)
  - Permitir uso online (upload de CSV + previsão)

---

## ✅ SEÇÃO 7: Estrutura do Projeto
- [x] Atualizar estrutura no README removendo scripts univariados
- [x] Garantir que os diretórios estejam assim:

```
.
├── core/
│   ├── data_preprocessing_multivariado.py
│   ├── model_lstm_multivariado.py
│   ├── predictor_multivariado.py
├── data/raw/
├── models/
├── results/
├── utils/
│   ├── metrics.py
├── train_multivariado.py
├── compara_modelo.py
├── validar_e_prever_30_dias.py
├── app.py
├── README.md
├── requirements.txt
```

---

## ✅ SEÇÃO 8: Contato e Divulgação
- [A] Inserir LinkedIn e GitHub do autor
		https://github.com/Iz-castro
		www.linkedin.com/in/izcastro
		
- [A] Criar botão/link para versão em inglês futuramente (`README_EN.md`)
- [F] Criar texto de LinkedIn com emojis e link do repositório

---


## ✅ ETAPA FINAL
- [A] Criar `README.md` atualizado, com base em todas as informações desse TODO reformulado
- [A] Incluir gráficos, métricas e passo a passo de execução e inferência
