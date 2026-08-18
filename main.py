import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Clave de Gemini
MI_API_KEY = "AQ.Ab8RN6IahZY_VufN9Q1OfaMLW_c1HHzIDOy3vxIYP7TL2qwWbw"
os.environ["GOOGLE_API_KEY"] = MI_API_KEY

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Embeddings locales
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# Base vectorial
vectorstore = Chroma(
    persist_directory="./db_conocimiento",
    embedding_function=embeddings
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# Actualización al modelo oficial requerido: gemini-3.6-flash
llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=MI_API_KEY,
    temperature=0.2
)

template = """Eres el asistente virtual oficial del Programa de Asesores Electorales (PAE).
Responde a la consulta basándote ÚNICAMENTE en la información provista en el contexto normativo.
Si la información no está en el contexto, indica amablemente que no dispones de esos datos normativos.

Contexto:
{context}

Pregunta: {question}
Respuesta:"""

prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

class Consulta(BaseModel):
    pregunta: str

@app.post("/api/chat")
async def responder_chat(req: Consulta):
    try:
        respuesta = rag_chain.invoke(req.pregunta)
        return {"respuesta": respuesta}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))