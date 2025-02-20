from langchain_community.vectorstores import FAISS
import torch
import time
import streamlit as st
import os
from groq import Groq
from dotenv import load_dotenv
import json
from service.search_doc import retrieve_and_re_rank_advanced
from service.function_for_tools_call import rag_aio, rag_billionares, rag_economic
load_dotenv('.env')
groq_api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=groq_api_key)
MODEL = os.getenv('MODEL')

with open('tools.json', 'r') as f:
    tools = json.load(f)


if __name__ == "__main__":
    # main()