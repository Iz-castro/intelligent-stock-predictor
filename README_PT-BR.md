# 📈 Intelligent Stock Predictor (LSTM Multivariado)

Este projeto tem como objetivo prever preços de fechamento de ações brasileiras com base em séries temporais multivariadas, utilizando redes neurais LSTM. A aplicação é treinada individualmente para cada papel com dados históricos obtidos do site *br.investing.com*, e conta com visualizações gráficas e interface interativa via Gradio.

---

## ⚠️ AVISO LEGAL

Este projeto tem finalidade estritamente educacional e de pesquisa. Não constitui recomendação de investimento nem orientação financeira.

O uso dos resultados ou previsões geradas por este modelo para tomar decisões de compra ou venda de ativos é de inteira responsabilidade do usuário. O autor não se responsabiliza por eventuais perdas financeiras decorrentes do uso deste software. Sempre consulte um profissional habilitado antes de investir.

---

## 🎯 Objetivo

Desenvolver um sistema inteligente e modular que:
- Realize previsões para o **próximo dia útil** com base em 60 dias anteriores;
- Compare previsões com dados reais de teste;
- Projete **tendências futuras de 30 e 60 dias úteis**;
- Permita uso interativo com upload de dados CSV via Gradio;
- Seja facilmente atualizável e auditável.

---

## 🧠 Modelo Utilizado

### 🔸 LSTM Multivariado
- Entrada:
  - `Fechamento`
  - `Retorno_%`
  - `MM9`
  - `RSI`
- Previsão: Fechamento do próximo dia
- Arquivo: `model_lstm_multivariado.keras`

A arquitetura utilizada consiste em três camadas LSTM sequenciais com Dropouts progressivos para evitar overfitting, seguidas por uma camada densa final que retorna a previsão de fechamento. 
A rede é otimizada com o otimizador Adam e taxa de aprendizado reduzida para garantir estabilidade no treinamento.

---

## 📊 Resultados Quantitativos

| Papel   | RMSE   | MAE    | Última Avaliação       |
|---------|--------|--------|------------------------|
| PETR4   | 1.52   | 1.11   | 2025-05-15 22:10:03     |
| BBAS3   | 1.22   | 0.95   | 2025-05-16 10:37:42     |
| VALE3   | 1.73   | 1.28   | 2025-05-16 11:02:25     |

> As métricas foram geradas com base no conjunto de teste (20%) e salvas em `results/metricas_modelos.csv`.

---

## 📉 Gráficos de Comparação

### 🔹 Comparativo Real x Previsto:
- ![PETR4](results/comparativo_teste_multivariado_PETR4.png)
- ![BBAS3](results/comparativo_teste_multivariado_BBAS3.png)
- ![VALE3](results/comparativo_teste_multivariado_VALE3.png)

### 🔹 Curvas de Treinamento:
- `treinamento_multivariado_PETR4.png`
- `treinamento_multivariado_BBAS3.png`
- `treinamento_multivariado_VALE3.png`

### 🔮 Projeções Futuras:
- `validacao_e_previsao_30_dias_multivariado.png`
- `previsao_60_dias.png`

---

## 📦 Estrutura do Projeto

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
│   └── metrics.py
├── train_multivariado.py
├── compara_modelo.py
├── validar_e_prever_30_dias.py
├── app.py
├── requirements.txt
├── README.md
```

---

## 🔧 Como Executar Localmente

1. Clone o repositório:
```bash
git clone https://github.com/Iz-castro/intelligent-stock-predictor.git
cd intelligent-stock-predictor
```

2. Crie e ative um ambiente virtual:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

4. Execute o treinamento:
```bash
python train_multivariado.py
```

5. Gere comparações e previsões:
```bash
python compara_modelo.py
python validar_e_prever_30_dias.py
```

6. Rode a aplicação interativa:
```bash
python app.py
```

---

## 🌐 Aplicação Web

A aplicação é implementada com **Gradio**, permitindo:
- Upload de novos arquivos `.csv`;
- Treinamento com múltiplos papéis;
- Visualização de métricas e gráficos;
- Previsão baseada em dados atualizados.

> 💡 Futuramente, será hospedada via HuggingFace Spaces ou Streamlit Cloud.

---

## 🔍 Fonte dos Dados

- Os dados foram obtidos manualmente do site: https://br.investing.com/
- 🚧 TODO: Atualizar para uso de API automática (Alpha Vantage, Twelve Data, B3 oficial)

---

## 📢 Contato e Créditos

Desenvolvido por **Izael Castro**  
📬 Email: *izaeldecastro@gmail.com*  
🔗 GitHub: [Iz-castro](https://github.com/Iz-castro)  
🔗 LinkedIn: [www.linkedin.com/in/izcastro](https://www.linkedin.com/in/izcastro)

---

## 📜 Licença

Este projeto está licenciado sob a Licença MIT.  
Consulte o arquivo `LICENSE` para mais detalhes.
