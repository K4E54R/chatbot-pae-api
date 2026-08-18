import os
import json
import glob
import streamlit as st
import google.generativeai as genai

st.set_page_config(page_title="Asistente PAE", page_icon="🤖", layout="centered")

# Configuración de la API Key de Gemini
api_key = os.environ.get("MI_API_KEY")
if not api_key and "MI_API_KEY" in st.secrets:
    api_key = st.secrets["MI_API_KEY"]

if api_key:
    genai.configure(api_key=api_key)

st.markdown("""
    <style>
        .stApp { background-color: #0F172A; color: #F8FAFC; }
        header { visibility: hidden; }
        footer { visibility: hidden; }
    </style>
""", unsafe_allow_html=True)

st.title("🤖 Asistente Virtual PAE")
st.caption("Consultas operativas, normativas y cronograma del Programa de Asesores Electorales")

@st.cache_resource
def load_knowledge_base():
    """Carga los fragmentos de conocimiento directamente desde los JSON"""
    chunks = []
    json_files = glob.glob("*.json") + glob.glob("**/*.json", recursive=True)
    for fpath in json_files:
        if "package" in fpath or ".streamlit" in fpath:
            continue
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            chunks.append(item.get("contenido") or item.get("texto") or item.get("text") or str(item))
                        elif isinstance(item, str):
                            chunks.append(item)
                elif isinstance(data, dict):
                    for k, v in data.items():
                        chunks.append(f"{k}: {v}")
        except Exception:
            pass
    return [c for c in chunks if len(c.strip()) > 20]

all_chunks = load_knowledge_base()

def retrieve_context(query, chunks, top_k=6):
    """Búsqueda ligera por relevancia léxica y términos clave"""
    if not chunks:
        return ""
    words = [w.lower() for w in query.split() if len(w) > 3]
    scores = []
    for c in chunks:
        c_lower = c.lower()
        score = sum(2 for w in words if w in c_lower)
        scores.append((score, c))
    scores.sort(key=lambda x: x[0], reverse=True)
    selected = [c for score, c in scores[:top_k] if score > 0]
    
    if not selected:
        selected = chunks[:top_k]
    return "\n\n---\n\n".join(selected)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "¡Hola! 👋 Soy el Asesor virtual del Programa de Asesores Electorales (PAE). ¿En qué duda normativa, operativa o de cronograma te puedo colaborar hoy?"}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Escribe tu consulta aquí..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            context = retrieve_context(prompt, all_chunks, top_k=6)
            
            prompt_completo = f"""Eres el Asistente Virtual Oficial del Programa de Asesores Electorales (PAE) del Tribunal Supremo de Elecciones (TSE) de Costa Rica.

REGLAS DE RESPUESTA:
1. Responde de forma clara, directa, estructurada y profesional utilizando viñetas y negrita.
2. OBLIGACIÓN DE FECHAS Y FUENTES: Cuando la respuesta involucre plazos, cronogramas, trámites o deberes, incluye siempre las FECHAS EXACTAS y las FUENTES NORMATIVAS (artículos de leyes, códigos o reglamentos) que correspondan según el contexto.
3. CIERRE OBLIGATORIO: Al final de cada respuesta añade la referencia documental con el formato:
   - "Entre otros, para más información verifica el manual de la persona asesora en la página [Número de Página o Sección]."

CONTEXTO:
{context}

PREGUNTA:
{prompt}
"""
            model = genai.GenerativeModel("models/gemini-3.6-flash")
            response_stream = model.generate_content(prompt_completo, stream=True)
            
            def stream_output():
                for chunk in response_stream:
                    if chunk.text:
                        yield chunk.text

            reply = st.write_stream(stream_output)
            st.session_state.messages.append({"role": "assistant", "content": reply})
            
        except Exception as err:
            st.error(f"Error al procesar la respuesta: {str(err)}")
