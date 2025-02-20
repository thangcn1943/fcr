import os
import dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings

dotenv.load_dotenv('.env') 
MODEL = os.getenv("MODEL")

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={'device': 'cpu'}
)