"""
Interfaz de chat en Streamlit para el agente que estima precios de casas.

Mantiene el historial en st.session_state y se lo pasa completo al agente en
cada turno. Las llamadas a la tool del modelo de ML se muestran en un
desplegable debajo de cada respuesta para ver qué datos recibió el pipeline.

Lanzar:
    streamlit run main.py
"""

from pathlib import Path

import streamlit as st

from agent import BOT_NAME, construir_agente, responder

BASE_DIR = Path(__file__).resolve().parent
RUTA_LOGO = BASE_DIR / "images" / "datapath-logo.png"
MENSAJE_BIENVENIDA = (
    f"¡Hola! Soy {BOT_NAME}. Cuéntame cómo es la casa (área, calidad, barrio, "
    "garaje, año de remodelación…) y estimo su precio con el modelo de Machine Learning."
)


@st.cache_resource
def obtener_agente():
    return construir_agente()


def mostrar_llamadas(llamadas: list[dict]):
    if not llamadas:
        return
    with st.expander(f"🔧 Llamadas al modelo de ML ({len(llamadas)})"):
        for llamada in llamadas:
            st.markdown(f"**{llamada['nombre']}**")
            st.json(llamada["argumentos"])


st.set_page_config(page_title="Agente Precio de Casas", page_icon="🏠")
st.title("🏠 Agente IA - Precio de Casas - DATAPATH")

with st.sidebar:
    st.image(str(RUTA_LOGO), width="stretch")
    st.markdown(
        "Agente con **GPT-5.1** (razonamiento *low*) que usa como tool el "
        "pipeline de regresión entrenado con el dataset House Prices de Ames, Iowa."
    )
    if st.button("🗑️ Nueva conversación", width="stretch"):
        st.session_state.mensajes = []
        st.rerun()

if "mensajes" not in st.session_state:
    st.session_state.mensajes = []

try:
    agente = obtener_agente()
except RuntimeError as error:
    st.error(str(error))
    st.stop()

with st.chat_message("assistant"):
    st.markdown(MENSAJE_BIENVENIDA)

for mensaje in st.session_state.mensajes:
    with st.chat_message(mensaje["role"]):
        st.markdown(mensaje["content"])
        mostrar_llamadas(mensaje.get("llamadas", []))

if pregunta := st.chat_input("Describe la casa..."):
    st.session_state.mensajes.append({"role": "user", "content": pregunta})
    with st.chat_message("user"):
        st.markdown(pregunta)

    historial = [
        {"role": m["role"], "content": m["content"]} for m in st.session_state.mensajes
    ]
    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            try:
                respuesta, llamadas = responder(agente, historial)
            except Exception as error:  # noqa: BLE001 - se muestra al usuario
                respuesta, llamadas = f"Ocurrió un error al consultar al agente: {error}", []
        st.markdown(respuesta)
        mostrar_llamadas(llamadas)

    st.session_state.mensajes.append(
        {"role": "assistant", "content": respuesta, "llamadas": llamadas}
    )
