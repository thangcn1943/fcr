from dotenv import load_dotenv
from sentence_transformers import CrossEncoder
import os

load_dotenv()

RESCORE_MODEL = os.getenv('RESCORE_MODEL') # 'itdainb/PhoRanker'

# rerank search results
# cross_encoder = CrossEncoder(RESCORE_MODEL)
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
def retrieve_and_re_rank_advanced(vector_db, query, k=10):
    # Lấy kết quả từ Vector Database
    docs_with_scores = vector_db.similarity_search_with_score(query, k=k)
    
    # Chuẩn bị dữ liệu cho Cross-Encoder
    doc_texts = [doc.page_content for doc, _ in docs_with_scores]
    pairs = [[query, doc] for doc in doc_texts]
    
    # Dùng Cross-Encoder để đánh giá lại
    rerank_scores = cross_encoder.predict(pairs)
    
    # Sắp xếp lại theo score mới
    ranked_docs = sorted(zip(doc_texts, rerank_scores), key=lambda x: x[1], reverse=True)
    
    results = [doc for doc, _ in ranked_docs]
    scores = [score for _, score in ranked_docs]
    
    return results, scores
