import torch
import time
import streamlit as st
import os
from groq import Groq
from dotenv import load_dotenv
import json
from service.function_for_tools_call import rag_aio, rag_billionares, rag_economic
import streamlit as st
import logging
logging.disable(logging.WARNING)

load_dotenv('.env')
groq_api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=groq_api_key)
MODEL = os.getenv('MODEL')
EMBED_MODEL = os.getenv("EMBED_MODEL")
with open('tools.json', 'r') as f:
    tools = json.load(f)

def run_conversation(user_prompt):
    messages = [
        {
            "role": "system",
            "content": 'Xin chào! Tôi là trợ lý AI được vận hành bởi công nghệ RAG. Tôi chỉ có thể cung cấp thông tin dựa trên những gì tôi truy xuất được từ cơ sở kiến thức của mình. Nếu không tìm thấy thông tin liên quan trong cơ sở dữ liệu, tôi sẽ thông báo bằng cách nói "Tôi không được cung cấp thông tin về chủ đề này." Điều này đảm bảo rằng câu trả lời của tôi được đặt trên nền tảng dữ liệu thực tế thay vì tạo ra thông tin không có nguồn thích hợp.'
        },
        {
            "role": "user",
            "content": user_prompt,
        }
    ]

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        max_tokens=4096
    )

    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls

    if tool_calls:
        # try:
        available_functions = {
            "rag_aio": rag_aio,
            "rag_billionares": rag_billionares,
            "rag_economic": rag_economic
        }

        messages.append(response_message)
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            print(function_name)
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            function_response = function_to_call(**function_args)
            
            return function_response
   
    else:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=4096
        )
        messages.append(response_message)
        final_response = response.choices[0].message.content

        return final_response

def main():

    st.title("QA System")

    # Khởi tạo session state nếu chưa có
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Hiển thị lịch sử hội thoại
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    query = st.chat_input("Enter your query: ")

    start = time.time()
    if query:

        # Thêm tin nhắn của user vào session_state
        st.session_state.messages.append({"role": "user", "content": query})

        # Hiển thị tin nhắn của user
        with st.chat_message("user"):
            st.markdown(query)

        # Gọi model để lấy câu trả lời
        answer = run_conversation(query)

        # Hiển thị tin nhắn của assistant
        with st.chat_message("assistant"):
            full_res = ""
            holder = st.empty()
            for word in answer.split():
                full_res += word + " "
                time.sleep(0.1)
                holder.markdown(full_res + "▌")
            holder.markdown(full_res)

        # Lưu tin nhắn của assistant vào session_state
        st.session_state.messages.append({"role": "assistant", "content": full_res})

        end = time.time()
        print("Time to process query:", end - start)

    else:
        print("Please enter your query")
    end = time.time()
    print("Time to process query: ", end-start)


if __name__ == "__main__":
    main()