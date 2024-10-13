import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage, trim_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from operator import itemgetter

from langchain_core.runnables import RunnablePassthrough
import streamlit as st

# load pdf 
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain.callbacks import StreamlitCallbackHandler

from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import tempfile
from langchain_pinecone import PineconeVectorStore


import string
import random

load_dotenv()


model = ChatOpenAI(model="gpt-3.5-turbo")
pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])
index_name = 'langchainvectors'
index = pc.Index(index_name)

store = {}

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


trimmer = trim_messages(
    max_tokens=65,
    strategy="last",
    token_counter=model,
    include_system=True,
    allow_partial=False,
    start_on="human",
)

config = {"configurable": {"session_id": 'test'}}

is_file_uploaded = False
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

output_parser = StrOutputParser()
chain = (
    RunnablePassthrough.assign(messages=itemgetter("messages") | trimmer)
    | prompt
    | model
    | output_parser
)
with_message_history = RunnableWithMessageHistory(chain, get_session_history, input_messages_key="messages")

st.title("AI-Powered Decision-Making Platform")


def chat_actions():
    st.session_state["chat_history"].append(
        {"role": "user", "content": st.session_state["chat_input"]},
    )
    

    st.session_state["chat_history"].append(
        {
            "role": "assistant",
            "content": rag_chain.invoke({"input": st.session_state["chat_input"]}) if uploaded_files else with_message_history.invoke({"messages": st.session_state["chat_input"]}, config=config),
        },  # This can be replaced with your chat response logic
    )


if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
        

uploaded_files = st.sidebar.file_uploader(label="Upload", type=["pdf"])
if uploaded_files is not None:
    st.write("File uploaded:", uploaded_files.name)
    # Process the uploaded file
    st.write("Wait for process...")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_files.read())
            temp_file_path = tmp_file.name
    # try:
    loader = PyPDFLoader(temp_file_path)
    documents = loader.load()
    
    splits = text_splitter.split_documents(documents)
    index.delete(delete_all=True)
    vectorstore = PineconeVectorStore.from_documents(documents=splits, index_name=index_name, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()
        
    system_prompt = (
        "You are an assistant for question-answering tasks."
        "Use the following pieces of retrieved context to answer"
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}")
        ]
    )
    question_answer_chain = create_stuff_documents_chain(model, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    # response = rag_chain.invoke(
    # {
    #     "input": "What is new tax regime?",
    # }),
    # st.write(response[0]['answer'])
    st.write("Chat with uploaded file", uploaded_files.name)
    # except Exception as e:
    #     st.error("An error occurred while processing file ", e)
    
st.chat_input("Enter your message", on_submit=chat_actions, key="chat_input")
for i in st.session_state["chat_history"]:
    with st.chat_message(name=i["role"]):
        # st.write(i["content"]["answer"] if len(i["content"]["answer"]) else i["content"])
        st.write(i["content"])