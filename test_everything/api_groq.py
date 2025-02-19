import os

from groq import Groq
import dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

dotenv.load_dotenv('.env') 
groq_api_key = os.getenv("GROQ_API_KEY")

texts = [
    "Fast language models improve efficiency in AI applications.",
    "They enable real-time interactions and better user experience.",
    "Optimized models reduce computational costs and energy consumption.",
    "Cat and dog are animals",
]

# Khởi tạo vector database với FAISS
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={'device': 'cpu'}
)
vector_db = FAISS.from_texts(texts, embeddings)
print(vector_db)
def retrieve_relevant_documents(query):
    docs = vector_db.similarity_search(query, k=1)
    return "\n".join([doc.page_content for doc in docs])

# Người dùng nhập câu hỏi
query = "Cat and dog are what?"
retrieved_docs = retrieve_relevant_documents(query)
print(retrieved_docs)
# Tạo client Groq
client = Groq(api_key=groq_api_key)

# Gửi yêu cầu đến mô hình Groq
chat_completion = client.chat.completions.create(
    messages=[
        {"role": "system", "content": "You are a helpful assistant that utilizes retrieved information."},
        {"role": "user", "content": f"Context:\n{retrieved_docs}\n\nQuestion: {query}"},
    ],
    model="llama-3.3-70b-versatile",
)

print("Question: ",chat_completion.choices[0].message.content)