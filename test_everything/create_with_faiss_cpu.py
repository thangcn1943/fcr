import os
from groq import Groq
import dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore


texts = [
    "Fast language models improve efficiency in AI applications.",
    "They enable real-time interactions and better user experience.",
    "Optimized models reduce computational costs and energy consumption.",
    "Cat and dog are animals",
]

# Create a document for each text
documents = [Document(page_content=text, metadata={'source': 'test'}) for text in texts]

# Khởi tạo vector database với FAISS
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={'device': 'cpu'}
)

index = faiss.IndexFlatL2(len(embeddings.embed_query('thang')))

vector_store = FAISS(
    embedding_function=embeddings,
    index= index ,
    index_to_docstore_id={},
    docstore=InMemoryDocstore()
)

vector_store.add_documents(documents) 

vector_store.save_local('test_everything/test_faiss_cpu')