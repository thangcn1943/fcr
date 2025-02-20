import os
from groq import Groq
import dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from service.search_doc import retrieve_and_re_rank_advanced

MODEL = os.getenv('MODEL')
EMBED_MODEL = os.getenv("EMBED_MODEL") # 'BAAI/bge-small-en-v1.5'
embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs={'device': 'cpu'}
)
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

def rag_medical(question: str):
    vector_db = FAISS.load_local('/home/thangcn/Downloads/datn/faiss_db/pdf_medical', embeddings, allow_dangerous_deserialization=True)
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