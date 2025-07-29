import streamlit as st

# utils
from utils.chat_callback_handler import ChatCallbackHandler
from utils.chatbot_session import ChatBotSession
from utils.docs_handler import DocsHandler

# LangChain - Chain
from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda

# LangChain - Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory

### Settings
# a. Set the page configuration
st.set_page_config(
    page_title="DocumentGPT",
    page_icon=":page_facing_up:",
)

# b. Initialization
# b-1. docs
docs_handler = DocsHandler()
docs_handler.splitter = CharacterTextSplitter.from_tiktoken_encoder(
    separator="\n",
    chunk_size=600,
    chunk_overlap=100,
)

# b-2. chat
chatbot_session = ChatBotSession("document")
chat_handler = ChatCallbackHandler(chatbot_session)

# b-3. memory
if chatbot_session.memory is None:
    chatbot_session.set_memory(
        ConversationBufferMemory(
            return_messages=True,
            memory_key="history",
        )
    )


### Functions
def respond_to_question(question, file_path):
    llm = ChatOpenAI(temperature=0.1, streaming=True, callbacks=[chat_handler])

    retriever = docs_handler.embedding_n_return_retriever(file_path)

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

    # chain은 하나의 string(message)을 받는다
    chain = (
        {
            # retriver는 string(message)을 받아 List of Document를 chain의 두 번째 component(RunnableLambda)에 전달
            # RunnableLambda는 List of Document를 받아 함수(format_docs)를 실행
            # prompt의 context에 들어갈 string을 생성해 반환해줌
            "context": retriever | RunnableLambda(docs_handler.format_docs),
            "question": RunnablePassthrough(),
            "history": chatbot_session.load_memory,
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    answer = chain.invoke(question)
    chatbot_session.save_memory(question, answer)


### Main
if not st.session_state.get("api_key"):
    st.markdown(
        """
            ## Enter your OpenAI API key for using this app.
        """
    )
    if st.button(
        label="Go to Home",
        help="You can enter your OpenAI API key on the Home page.",
    ):
        st.switch_page("Home.py")

else:
    # File upload widget
    with st.sidebar:
        uploaded_file = st.file_uploader(
            "Upload a .pdf .txt or .docx file", type=["pdf", "txt", "docx"]
        )

    # File이 업로드 되면, File embedding 및 챗봇 시작
    if uploaded_file:
        text_path = docs_handler.save_text_file(uploaded_file)

        # 채팅 히스토리
        chatbot_session.paint_messages()

        # 첫 안내
        chatbot_session.send_info(
            "File uploaded and embedded successfully!",
        )

        # 질문 입력창
        question = st.chat_input(
            "Ask me anything about the uploaded file!",
        )

        # 채팅
        if question:
            chatbot_session.send_message(question, "human")
            with st.chat_message("ai"):
                respond_to_question(question, text_path)

    else:
        st.title("DocumentGPT")
        st.markdown(
            """
                ## Upload a document to ask questions about its content.
            """,
        )
