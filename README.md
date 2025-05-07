# 📊 Intelligent Stock Predictor - LSTM Univariado & Multivariado

Este projeto tem como objetivo a construção de modelos de machine learning capazes de prever o comportamento de ativos financeiros com base em séries temporais. Utilizando arquiteturas LSTM (Long Short-Term Memory), o sistema foi projetado para analisar históricos de preços e gerar previsões para prazos variados.

---

## 🎯 Objetivos

- Criar um sistema inteligente que:
  - Preveja o fechamento do preço de ações para o **próximo dia útil**;
  - Valide as previsões com os dados reais de teste;
  - Projete **cenários futuros para 30 e 60 dias** com base no comportamento atual do mercado;
  - Compare abordagens **univariadas** (apenas fechamento) e **multivariadas** (com indicadores técnicos).

---

## 🧠 Modelos treinados

### 🔹 LSTM Univariado
- Entrada: Fechamento dos últimos 60 dias
- Previsão: Fechamento do próximo dia
- Arquivo: `model_lstm_univariado.keras`

### 🔸 LSTM Multivariado (Aprimorado)
- Entrada: 
  - `Fechamento`
  - `Retorno_%` (variação percentual)
  - `MM9` (média móvel de 9 dias)
  - `RSI` (indicador de momentum)
- Previsão: Fechamento do próximo dia
- Arquivo: `model_lstm_multivariado.keras`

---

## 🧪 Validação e Comparativos

As validações utilizam o conjunto de teste, comparando previsões com valores reais e gerando métricas:

- **RMSE** (erro quadrático médio)
- **MAE** (erro absoluto médio)

### 🔍 Gráficos gerados:

- `comparativo_teste_multivariado.png` → Comparativo completo real x previsto
- `validacao_30_dias_multivariado.png` → Zoom nos últimos 30 dias reais
- `validacao_e_previsao_30_dias_multivariado.png` → 30 dias reais + 30 dias futuros
- `previsao_60_dias.png` → Tendência futura para os próximos 60 dias
- `treinamento_multivariado.png` → Curvas de perda no treinamento do modelo

---

## 📁 Estrutura dos arquivos

```
.
├── core/
│   ├── data_preprocessing_univariado.py
│   ├── data_preprocessing_multivariado.py
│   ├── model_lstm_univariado.py
│   ├── model_lstm_multivariado.py
│   ├── predictor_univariado.py
│   └── predictor_multivariado.py
│
├── run_training_univariado.py
├── run_training_multivariado.py
├── predict_validated_30dias_univariado.py
├── predict_validated_30dias_multivariado.py
├── compara_univariado.py
├── compara_multivariado.py
├── requirements.txt
├── README.md
└── app_gradio.py (em desenvolvimento)
```

---

## ▶️ Como executar

1. Instale dependências:
```bash
pip install -r requirements.txt
```

2. Execute o treinamento multivariado:
```bash
python run_training_multivariado.py
```

3. Compare previsões com dados reais:
```bash
python compara_multivariado.py
```

4. Visualize a tendência futura:
```bash
python predict_validated_30dias_multivariado.py
```

---

## 💡 Dicas de uso

- Substitua os arquivos `.csv` em `data/raw/` com históricos reais de qualquer papel de ação
- Os modelos serão automaticamente treinados com base nesses dados
- Resultados e gráficos serão salvos na pasta `results/`

---

## 🔮 O que vem a seguir?

- Integração via interface **Gradio** para uso interativo
- Permitir upload de CSVs e seleção do modelo
- Exportação de relatórios de previsão em tempo real

---

## 🧠 Autor & Licença

Desenvolvido por [Izael Castro] — Repositório educativo/pessoal  
Licença: MIT

## 📬 Contato
Izael Castro
Email: izaeldecastro@gmail.com
GitHub: Iz-castro
