import streamlit as st

from nps_estimator.modeling.predict import predict_nps

"""
Aplicativo Streamlit para inferência interativa de NPS.

Este módulo implementa uma interface gráfica utilizando Streamlit que permite
ao usuário inserir informações operacionais de um cliente e obter uma estimativa
do score de NPS a partir de um modelo previamente treinado.

A aplicação coleta os seguintes parâmetros de entrada:
- Número de reclamações do cliente
- Quantidade de contatos com o atendimento ao cliente
- Tempo de resolução de problemas (em dias)
- Tempo de atraso na entrega (em dias)

Após o envio dos dados, o aplicativo chama a função `predict_nps` para realizar
a inferência e exibe o valor estimado de NPS de forma clara e destacada na
interface.

Dependências:
    - streamlit
    - nps_estimator.modeling.predict.predict_nps

Uso:
    Execute o aplicativo a partir da raiz do projeto com:
        streamlit run nps_estimator/modeling/app_nps_inference.py

Observações:
    - Este módulo é responsável apenas pela interface do usuário.
    - A lógica de inferência e carregamento do modelo é delegada à função
      `predict_nps`, garantindo separação de responsabilidades.
"""

# ---------------- CONFIG PAGE ----------------
st.set_page_config(page_title="Estimador de NPS", page_icon="📈", layout="centered")

# ---------------- TITLE ----------------
st.title("📈 Estimador de NPS")
st.write("Insira os dados do cliente para estimar o NPS.")

# ---------------- INPUTS ----------------
complaints_count = st.number_input("Número de reclamações", min_value=0, step=1)

customer_service_contacts = st.number_input(
    "Contatos com atendimento ao cliente", min_value=0, step=1
)

resolution_time_days = st.number_input("Tempo de resolução (dias)", min_value=0, step=1)

delivery_delay_days = st.number_input("Tempo de atraso na entrega (dias)", min_value=0, step=1)

# ---------------- BUTTON ----------------
if st.button("Calcular NPS"):
    data_input = {
        "complaints_count": complaints_count,
        "customer_service_contacts": customer_service_contacts,
        "resolution_time_days": resolution_time_days,
        "delivery_delay_days": delivery_delay_days,
    }

    st.subheader("📋 Resultado da predição")

    prediction = predict_nps(data_input)

    st.metric(label="NPS estimado", value=f"{prediction:.2f}")
