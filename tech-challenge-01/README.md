# tech-challenge-01

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Projeto desenvolvido pelos alunos

1. Carolina Yamada
2. Igor Constantino
3. Raphael Tavela
4. Rodrigo do Vale
5. Vinicius Miura

para o desafio Tech Challenge da Fase 01 do curso Pós-Tech IA Scientist - FIAP

---
## 📊 NPS Estimator
Este projeto implementa um pipeline completo de Machine Learning para estimativa de NPS (Net Promoter Score), incluindo:

Treinamento de modelo com LightGBM <br>
Inferência via código Python<br>
Interface interativa com Streamlit<br>
CLI com Typer<br>
Organização modular e pronta para produção<br>

---
## 🧠 Visão Geral
O objetivo do projeto é estimar o score de NPS de um cliente a partir de variáveis operacionais relacionadas à experiência do cliente, como:

Número de reclamações<br>
Contatos com o atendimento ao cliente<br>
Tempo de resolução de problemas<br>
Atraso na entrega<br>

O modelo treinado pode ser utilizado tanto via linha de comando, quanto por meio de uma interface web interativa.


## Organização do Projeto

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         tech_challenge_01 and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── tech_challenge_01   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes tech_challenge_01 a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── app_nps_inference.py<- Code to run the app with interface that predicts with trained model
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

---
## ⚙️ Tecnologias Utilizadas

- Python 3.10+
- pandas / NumPy
- lightGBM
- joblib
- streamlit
- typer
- loguru

---
## 📦 Instalação
### 1️⃣ Criar e ativar o ambiente virtual
```Shell 
python -m venv .venv
```
**Windows**
```Shell
.venv\Scripts\activate
```
**Linux / macOS**
```Shell
source .venv/bin/activate
```
### 2️⃣ Instalar dependências
```Shell
pip install -r requirements.txt
```
Ou, se preferir instalar o projeto como pacote:
```Shell
pip install -e .
```

---
## 🏋️ Treinamento do Modelo
Para treinar o modelo e salvá-lo localmente:
```Shell
python -m nps_estimator.modeling.train
```
Durante o processo:
- O dataset é carregado
- O modelo LightGBM é treinado
- Métricas como MAE, RMSE e R² são exibidas
- O modelo final é salvo em disco

---
## 🔮 Inferência via Código Python
Exemplo de uso da função de inferência:
```Python
from nps_estimator.modeling.predict import predict_nps

data_input = {"complaints_count": 1,
              "customer_service_contacts": 2,
              "resolution_time_days": 3,
              "delivery_delay_days": 4}

prediction = predict_nps(data_input)
print(prediction)
```

## 🖥️ Interface Web com Streamlit
O projeto inclui uma interface gráfica para inferência interativa.

### ▶️ Executar o app
Na raiz do projeto:
```Shell
streamlit run nps_estimator/modeling/app_nps_inference.py
```
### 🎯 Funcionalidades da interface

- Inputs numéricos para variáveis do cliente
- Botão de predição
- Exibição clara do NPS estimado

### 🧪 Boas Práticas Aplicadas
✅ Separação entre lógica de negócio e interface <br>
✅ Cache de artefatos com lru_cache <br>
✅ Imports absolutos e estrutura de pacote <br>
✅ Tipagem e docstrings padronizadas <br>
✅ Código preparado para API / deploy <br>

### 🚀 Próximos Passos (Sugestões)

- [ ] Criar testes unitários
- [ ] Deploy no Streamlit Cloud
- [ ] Expor API com FastAPI
- [ ] Versionamento de modelos
- [ ] Monitoramento de predições
--------

