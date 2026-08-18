import os
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import google.generativeai as genai

st.set_page_config(page_title="Asistente PAE", page_icon="🤖", layout="centered")

# Configuración de clave Gemini
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
    return vectorstore.as_retriever(search_kwargs={"k": 4})

retriever = load_rag()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "¡Hola! 👋 Soy el Asesor virtual del Programa de Asesores Electorales (PAE). ¿En qué duda normativa u operativa te colaboro hoy?"}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Escribe tu consulta aquí..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        docs = retriever.invoke(prompt)
        context = "\n\n".join([d.page_content for d in docs])
        
        system_instruction = f"""Eres el Asesor virtual del Programa de Asesores Electorales (PAE) del TSE Costa Rica.
Responde de forma clara y precisa basándote ÚNICAMENTE en el siguiente contexto normativo y operativo:

{context}
"""
        model = genai.GenerativeModel("gemini-1.5-flash", system_instruction=system_instruction)
        response = model.generate_content(prompt)
        reply = response.text
        st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})
