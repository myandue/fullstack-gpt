import streamlit as st
import os

# utils
from utils.chat_callback_handler import ChatCallbackHandler
from utils.chatbot_session import ChatBotSession

# File
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.storage import LocalFileStore
from langchain.embeddings.cache import CacheBackedEmbeddings
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Chat
from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.base import BaseCallbackHandler

### Settings
# Set the page configuration
st.set_page_config(
    page_title="DocumentGPT",
    page_icon=":page_facing_up:",
)


# chat
chat = None
chatbot_session = ChatBotSession("document")
chat_handler = ChatCallbackHandler(chatbot_session)

# memory
if chatbot_session.memory is None:
    chatbot_session.set_memory(
        ConversationBufferMemory(
            return_messages=True,
            memory_key="history",
        )
    )


### Functions
# file
# 업로드한 파일이 이미 존재하는 경우 해당 함수를 실행하지 않음
@st.cache_resource(show_spinner="Embedding file...")
def embedding_file(file):
    # setting path
    file_path = "./.cache/files"
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    cache_path = "./.cache/embeddings"
    if not os.path.exists(cache_path):
        os.makedirs(cache_path)

    # 업로드한 파일 저장
    file_content = uploaded_file.read()
    with open(f"{file_path}/{uploaded_file.name}", "wb") as f:
        f.write(file_content)

    # set up the embedding process
    cache_dir = LocalFileStore(f"{cache_path}/{uploaded_file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    embeddings = OpenAIEmbeddings()

    # file embedding
    file = UnstructuredFileLoader(f"./.cache/files/{uploaded_file.name}")
    docs = file.load_and_split(text_splitter=splitter)
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        embeddings, cache_dir
    )
    vector_store = FAISS.from_documents(docs, cached_embeddings)
    retriver = vector_store.as_retriever()

    return retriver


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# ai response
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a helpful assistant. Answer quetions using"
                " only the following context. If you don't know the"
                " answer just say you don't know, don't make it"
                " up:\n\n"
                "Context:\n{context}\n\n"
                "Previous conversation:\n{history}\n\n"
            ),
        ),
        ("human", "{question}"),
    ]
)


def respond(message, retriver):
    # chain은 하나의 string(message)을 받는다
    chain = (
        {
            # retriver는 string(message)을 받아 List of Document를 chain의 두 번째 component(RunnableLambda)에 전달
            # RunnableLambda는 List of Document를 받아 함수(format_docs)를 실행
            # prompt의 context에 들어갈 string을 생성해 반환해줌
            "context": retriver | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
            "history": chatbot_session.load_memory,
        }
        | prompt
        | chat
    )

    response = chain.invoke(message).content
    chatbot_session.save_memory(message, response)


### Main
if not st.session_state.get("api_key"):
    st.markdown(
        """
        ## Enter your OpenAI API key for using this app.
        """
    )
    st.link_button(
        "Go to Home",
        "/",
        help="You can enter your OpenAI API key on the Home page.",
    )

else:
    chat = ChatOpenAI(
        temperature=0.1, streaming=True, callbacks=[chat_handler]
    )

    # File upload widget
    with st.sidebar:
        uploaded_file = st.file_uploader(
            "Upload a .pdf .txt or .docx file", type=["pdf", "txt", "docx"]
        )

    # File이 업로드 되면, File embedding 및 챗봇 시작
    if uploaded_file:
        retriver = embedding_file(uploaded_file)
        chatbot_session.send_message(
            "File uploaded and embedded successfully!",
            "ai",
            save=False,
        )

        # 채팅 히스토리
        chatbot_session.paint_messages()

        # 메세지 입력창
        message = st.chat_input(
            "Ask me anything about the uploaded file!",
        )

        # 채팅
        if message:
            chatbot_session.send_message(message, "human")
            with st.chat_message("ai"):
                respond(message, retriver)

    else:
        st.title("DocumentGPT")
        st.markdown(
            """
            ## Upload a document to ask questions about its content.
            """,
        )
