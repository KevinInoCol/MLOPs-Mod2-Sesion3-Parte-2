"""
Agente IA que conversa con el usuario y estima precios de casas con el modelo de ML.

Arma un agente de LangChain (create_agent) con:
  - LLM: configurado en model_config/model.yaml (GPT-5.1, razonamiento low).
  - System prompt: prompt/system_prompt.yaml, con placeholders inyectados en runtime.
  - Tool: predecir_precio_casa (tools/), que invoca el pipeline de Modelo-ML/.

El agente no guarda memoria propia: quien lo usa (main.py) le pasa el historial
completo en cada turno.

Requiere OPENAI_API_KEY en el .env.

Probar desde la terminal:
    python agent.py
"""

import os
from datetime import date
from pathlib import Path

import yaml
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from tools.predecir_precio_casa import predecir_precio_casa

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

BASE_DIR = Path(__file__).resolve().parent
RUTA_MODEL_CONFIG = BASE_DIR / "model_config" / "model.yaml"
RUTA_PROMPT = BASE_DIR / "prompt" / "system_prompt.yaml"
BOT_NAME = "Casi"


def verificar_configuracion():
    """Falla rápido si falta la API key."""
    if not OPENAI_API_KEY:
        raise RuntimeError("Falta OPENAI_API_KEY en el archivo .env")


def cargar_yaml(ruta: Path) -> dict:
    with open(ruta, encoding="utf-8") as f:
        return yaml.safe_load(f)


def construir_llm() -> ChatOpenAI:
    cfg = cargar_yaml(RUTA_MODEL_CONFIG)
    return ChatOpenAI(
        model=cfg["model"],
        reasoning_effort=cfg["reasoning_effort"],
        max_tokens=cfg.get("max_tokens"),
        timeout=cfg.get("timeout"),
        max_retries=cfg.get("max_retries", 2),
    )


def construir_system_prompt() -> str:
    prompt = cargar_yaml(RUTA_PROMPT)["system_prompt"]
    return (
        prompt.replace("{bot_name}", BOT_NAME)
        .replace("{fecha_actual}", date.today().isoformat())
    )


def construir_agente():
    """Devuelve el agente listo para usar con .invoke({"messages": [...]})."""
    verificar_configuracion()
    return create_agent(
        model=construir_llm(),
        tools=[predecir_precio_casa],
        system_prompt=construir_system_prompt(),
        name="agente_precio_casas",
    )


def responder(agente, historial: list[dict]) -> tuple[str, list[dict]]:
    """Ejecuta un turno.

    historial: lista de {"role": "user"|"assistant", "content": str}.
    Devuelve el texto de la respuesta y la lista de llamadas a la tool del turno.
    """
    resultado = agente.invoke({"messages": historial})
    mensajes_nuevos = resultado["messages"][len(historial):]

    llamadas = [
        {"nombre": tc["name"], "argumentos": tc["args"]}
        for m in mensajes_nuevos
        for tc in getattr(m, "tool_calls", None) or []
    ]
    return resultado["messages"][-1].text, llamadas


if __name__ == "__main__":
    verificar_configuracion()
    agente = construir_agente()
    historial: list[dict] = []
    print(f"{BOT_NAME}: Hola, cuéntame cómo es la casa y estimo su precio. (salir para terminar)")
    while (texto := input("Tú: ").strip()).lower() not in {"salir", "exit", "quit"}:
        historial.append({"role": "user", "content": texto})
        respuesta, _ = responder(agente, historial)
        historial.append({"role": "assistant", "content": respuesta})
        print(f"{BOT_NAME}: {respuesta}\n")
