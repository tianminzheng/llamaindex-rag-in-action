import streamlit as st
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.llms.openai import OpenAI
import openai
from llama_index.core import SimpleDirectoryReader


secrets= "secret.toml"
st.set_page_config(page_title="文档聊天助手", page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)
openai.api_key = st.secrets.openai_key
st.title("文档聊天助手")
         
if "messages" not in st.session_state.keys(): # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "欢迎。我来帮助你选购合适的手机。"}
    ]

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="请稍等，正在处理。这可能需要1-2分钟。"):
        reader = SimpleDirectoryReader(input_dir="..\data", recursive=True)
        docs = reader.load_data()
        Settings.llm = OpenAI(
            model="gpt-3.5-turbo",
            temperature=0.2,
            system_prompt="""作为一位虚拟商店助手，我将尝试向用户销售手机。
                 同时，在用户需要时，帮助用户提出问题并回答关于下一步的问题。
                 仅根据提供的数据进行回答。
                 如果用户没有指定具体细节，则根据当前的回答给出答案。
                 在需要时一次性提供所有细节，不要为每个产品反复询问用户特定功能。""",
        )
        index = VectorStoreIndex.from_documents(docs)
        return index

index = load_data()

if "chat_engine" not in st.session_state.keys(): # Initialize the chat engine
        st.session_state.chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

if prompt := st.chat_input("请输入你的问题"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# 如果是需要处理的用户消息，则调用LLM获取响应结果
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("处理中..."):
            response = st.session_state.chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) # Add response to message history
            
