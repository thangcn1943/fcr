import os
from groq import Groq
import dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
import json
from search_doc import retrieve_and_re_rank_advanced
from service.init_model import MODEL
from service.init_model import embeddings

dotenv.load_dotenv('.env') 
groq_api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=groq_api_key)


def rag_aio(question: str):
    vector_db = FAISS.load_local('/home/thangcn/Downloads/datn/faiss_db/pdf_aio', embeddings, allow_dangerous_deserialization=True)
    retrieved_docs, scores = retrieve_and_re_rank_advanced(vector_db, question)
        # Gửi yêu cầu đến mô hình Groq
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful assistant that utilizes retrieved information."},
            {"role": "user", "content": f"Context:\n{retrieved_docs[0]}\n\nQuestion: {question}"},
        ],
        model= MODEL,
    )
    # print(chat_completion.choices[0].message.content)
    return chat_completion.choices[0].message.content

def rag_billionares(question: str):
    vector_db = FAISS.load_local('/home/thangcn/Downloads/datn/faiss_db/pdf_billionares', embeddings, allow_dangerous_deserialization=True)
    retrieved_docs, scores = retrieve_and_re_rank_advanced(vector_db, question)
        # Gửi yêu cầu đến mô hình Groq
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful assistant that utilizes retrieved information."},
            {"role": "user", "content": f"Context:\n{retrieved_docs[0]}\n\nQuestion: {question}"},
        ],
        model= MODEL,
    )
    # print(chat_completion.choices[0].message.content)
    return chat_completion.choices[0].message.content

def rag_economic(question: str):
    vector_db = FAISS.load_local('/home/thangcn/Downloads/datn/faiss_db/pdf_economic', embeddings, allow_dangerous_deserialization=True)
    retrieved_docs, scores = retrieve_and_re_rank_advanced(vector_db, question)
        # Gửi yêu cầu đến mô hình Groq
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful assistant that utilizes retrieved information."},
            {"role": "user", "content": f"Context:\n{retrieved_docs[0]}\n\nQuestion: {question}"},
        ],
        model= MODEL,
    )
    # print(chat_completion.choices[0].message.content)
    return chat_completion.choices[0].message.content