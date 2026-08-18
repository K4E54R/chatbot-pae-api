import os
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import google.generativeai as genai

st.set_page_config(page_title="Asistente PAE", page_icon="🤖", layout="centered")

# Configurar API Key de Gemini
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
st.caption("Consultas operativas y normativas del Programa de Asesores Electorales")

@st.cache_resource
def load_rag():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    vectorstore = Chroma(persist_directory="./db_conocimiento", embedding_function=embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 5})

retriever = load_rag()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "¡Hola! 👋 Soy el Asesor virtual del Programa de Asesores Electorales (PAE). ¿En qué duda normativa u operativa te puedo colaborar hoy?"}
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
            docs = retriever.invoke(prompt)
            context = "\n\n---\n\n".join([d.page_content for d in docs])
            
            prompt_completo = f"""Eres el Asistente Virtual Oficial del Programa de Asesores Electorales (PAE) del Tribunal Supremo de Elecciones (TSE) de Costa Rica.

INSTRUCCIONES DE RESPUESTA:
1. Responde de forma clara, directa, concisa y estructurada basándote en el contexto normativo y operativo.
2. REGLA OBLIGATORIA DE CIERRE: Al final de cada respuesta debes incluir siempre una frase de referencia indicando la página o sección respectiva, usando el formato:
   - "Entre otros, para más información verifica el manual de la persona asesora en la página [Número de Página o Sección]."
   - Si en el contexto aparecen páginas específicas (como Pág. 34, Pág. 44, Pág. 54, etc.), indica el número exacto. Si no aparece el número exacto, indica la sección o tema respectivo del manual.

CONTEXTO:
{context}

PREGUNTA:
{prompt}
"""
            model = genai.GenerativeModel("models/gemini-3.6-flash")
            response_stream = model.generate_content(prompt_completo, stream=True)
            
            def generate_chunks():
                for chunk in response_stream:
                    if chunk.text:
                        yield chunk.text

            reply = st.write_stream(generate_chunks)
            st.session_state.messages.append({"role": "assistant", "content": reply})
            
        except Exception as err:
            st.error(f"Error al procesar la respuesta: {str(err)}")
