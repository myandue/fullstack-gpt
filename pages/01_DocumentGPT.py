import streamlit as st
import openai

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


# chat streaming
class ChatCallbackHandler(BaseCallbackHandler):
    ai_message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_new_token(self, token, *args, **kwargs):
        self.ai_message += token
        self.message_box.write(self.ai_message)

    def on_llm_end(self, *args, **kwargs):
        save_message(self.ai_message, "ai")


# chat
chat = None


# memory
if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationBufferMemory(
        return_messages=True,
        memory_key="history",
    )
memory = st.session_state["memory"]


### Functions
# file
# 업로드한 파일이 이미 존재하는 경우 해당 함수를 실행하지 않음
@st.cache_resource(show_spinner="Embedding file...")
def embedding_file(file):
    # 업로드한 파일 저장
    file_content = uploaded_file.read()
    file_path = f"./.cache/files/{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    # set up the embedding process
    cache_dir = LocalFileStore(f"./.cache/embeddings/{uploaded_file.name}")
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


# chat
def save_message(message, role):
    st.session_state.messages.append({"role": role, "message": message})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.write(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False,
        )


# ai
def load_memory(_):
    history = memory.load_memory_variables({})["history"]
    return history


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
            "history": load_memory,
        }
        | prompt
        | chat
    )

    response = chain.invoke(message).content
    memory.save_context({"input": message}, {"output": response})


### Main
# api key widget
if not st.session_state.get("api_key"):
    st.markdown(
        """
        ## Please enter your OpenAI API key<br>on the sidebar to use DocumentGPT.
        """,
        unsafe_allow_html=True,
    )
    with st.sidebar:
        st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="sk-...",
            key="api_key",
        )


else:
    # Chat
    openai.api_key = st.session_state["api_key"]

    if chat is None:
        chat = ChatOpenAI(
            temperature=0.1, streaming=True, callbacks=[ChatCallbackHandler()]
        )

    # File upload widget
    st.title("DocumentGPT")
    st.markdown(
        """
        ## Upload a document to ask questions about its content.
        """,
    )
    with st.sidebar:
        uploaded_file = st.file_uploader(
            "Upload a .pdf .txt or .docx file", type=["pdf", "txt", "docx"]
        )

    # File이 업로드 되면, File embedding 및 챗봇 시작
    if uploaded_file:
        retriver = embedding_file(uploaded_file)
        send_message(
            "File uploaded and embedded successfully!",
            "ai",
            save=False,
        )

        # 채팅 히스토리
        paint_history()

        # 메세지 입력창
        message = st.chat_input(
            "Ask me anything about the uploaded file!",
        )

        # 채팅
        if message:
            send_message(message, "human")
            with st.chat_message("ai"):
                respond(message, retriver)

    else:
        st.session_state["messages"] = []
