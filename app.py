from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from huggingface_hub import login 
import torch
import time
import streamlit as st
import os
from groq import Groq
import dotenv
import faiss
from service.search_doc import retrieve_and_re_rank_advanced



if __name__ == "__main__":
    main()