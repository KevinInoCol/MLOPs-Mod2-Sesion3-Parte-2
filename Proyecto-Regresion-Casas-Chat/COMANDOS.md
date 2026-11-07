# Comandos

## 1. Entrar a la carpeta
```bash
cd Proyecto-Regresion-Casas-Chat
```

## 2. Crear entorno e instalar dependencias
```bash
uv venv --python 3.12
uv pip install -r requirements.txt
```

## 3. Configurar la API key
Copiar `.env.example` a `.env` y completar `OPENAI_API_KEY`.

## 4. Lanzar el chat
```bash
uv run streamlit run main.py
```
Abrir: http://localhost:8501

Probar solo el agente en la terminal: `uv run python agent.py`
