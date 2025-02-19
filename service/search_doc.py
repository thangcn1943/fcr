import os
from groq import Groq
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from sentence_transformers import CrossEncoder

load_dotenv()

HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
MODEL_NAME = 'BAAI/bge-small-en-v1.5'
FAISS_ROOT = 'vector_db_pdf/index'
RESCORE_MODEL = 'itdainb/PhoRanker'

def search_doc(query: str):
    # retrieve and rerank documents
    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={'device': 'cpu'}
    )
    faiss = FAISS.load_local("test_faiss", embeddings, allow_dangerous_deserialization=True)
    # search for similar documents
    n_docs = faiss.index.ntatol
    print(n_docs)
    docs_with_scores = faiss.similarity_search_with_score(query, k=min(5, n_docs))
    # rerank
    cross_encoder = CrossEncoder(RESCORE_MODEL, device='cpu')
    pairs = [[query, doc] for doc in docs_with_scores]
    scores = cross_encoder.predict(pairs)
    docs_reranked = sorted(zip(docs_with_scores, scores), key=lambda x: x[1], reverse=True)
    
    return docs_reranked

