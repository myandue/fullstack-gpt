import streamlit as st
import os

# Site Load
from langchain_community.document_loaders import SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Chat
from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.base import BaseCallbackHandler

### Settings
# a. Set the page configuration
st.set_page_config(page_title="SiteGPT", page_icon="🔍")


# b. initilization
# b-1-1. chat
chat = None


# b-1-2. chat streaming
class ChatCallbackHandler(BaseCallbackHandler):
    ai_message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_new_token(self, token, *args, **kwargs):
        self.ai_message += token
        self.message_box.write(self.ai_message)

    def on_llm_end(self, *args, **kwargs):
        if self.ai_message:
            save_message(self.ai_message, "ai")


# b-2. memory
if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationBufferMemory(
        return_messages=True, memory_key="history"
    )
memory = st.session_state["memory"]


### Functions
# a. site
def parse_page(soup):
    header = soup.find("header")
    footer = soup.find("footer")

    if header:
        header.decompose()
    if footer:
        footer.decompose()

    return str(soup.get_text()).replace("\n", " ")


@st.cache_resource(show_spinner="Loading site...")
def load_site(url):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=200
    )
    loader = SitemapLoader(url, parsing_function=parse_page)
    loader.requests_per_second = 2

    docs = loader.load_and_split(text_splitter=splitter)
    vector_store = FAISS.from_documents(docs, OpenAIEmbeddings())

    return vector_store.as_retriever()


# b. chat
def save_message(message, role):
    st.session_state.messages.append({"role": role, "message": message})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.write(message)
    if save:
        save_message(message, role)


# c. chat history
def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)


# d. memory
def load_memory(_):
    return memory.load_memory_variables({})["history"]


search_history_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
    If a similar question exists in the following context('Previous conversation'), return the same answer.
    Else, just return a word 'False'

    Previous conversation:\n{history}\n\n
    """,
        ),
        ("human", "{question}"),
    ]
)


def search_history(inputs):
    chat.streaming = False

    search_history_chain = (
        {"history": load_memory, "question": RunnablePassthrough()}
        | search_history_prompt
        | chat
    )
    inputs["answer"] = search_history_chain.invoke(inputs["question"]).content

    return inputs


# e. answers
answers_prompt = ChatPromptTemplate.from_template(
    """
        Using ONLY the following context answer the user's question. If you can't, just say you don't know. Don't make anything up.
        Then, give a score to the answer between 0 and 5.
        If the answer answers the user's question, the score should be high. Else it should be low.
        Make sure to always include the answer's score even if it's 0.
        
        Context: {context}
        
        Examples:
        
        Question: How far away is the moon?
        Answer: The moon is 384,400 km away.
        Score: 5
        
        Question: How far away is the sun?
        Answer: I don't know.
        Score: 0
        
        Your turn!
        
        Question: {question}
    """
)


def get_answers(inputs):
    chat.streaming = False
    answers_chain = answers_prompt | chat

    docs = inputs["docs"]
    question = inputs["question"]

    return {
        "question": question,
        "answers": [
            {
                "answer": (
                    answers_chain.invoke(
                        {"context": doc.page_content, "question": question}
                    ).content
                ),
                "source": doc.metadata["source"],
            }
            for doc in docs
        ],
    }


choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
    Use ONLY the following pre-existing answers to answer user's question.
    Use the answers that have the highest score (more helpful) and favor the most recent ones.
    Cite sources and return the sources of the answers as they are, do not change them.

    Please follow the following answer's format.

    {{The answer you chose}}
    [{{The source of the answer you chose}}]

    Answers: {answers}
    """,
        ),
        ("human", "{question}"),
    ]
)


def choose_answer(inputs):
    chat.streaming = True
    choose_chain = choose_prompt | chat

    question = inputs["question"]
    answers = inputs["answers"]

    condensed = "\n\n".join(
        f"\n{answer['answer']}\nSource: {answer['source']}\n"
        for answer in answers
    )

    with st.chat_message("ai"):
        response = choose_chain.invoke(
            {"answers": condensed, "question": question}
        ).content
        memory.save_context({"input": question}, {"output": response})


def new_answer(inputs):
    if inputs["answer"] == "False":
        retriever = inputs["retriever"]
        question = inputs["question"]

        new_answer_chain = (
            {
                "docs": retriever,
                "question": RunnablePassthrough(),
            }
            | RunnableLambda(get_answers)
            | RunnableLambda(choose_answer)
        )

        new_answer_chain.invoke(question)

    else:
        send_message(inputs["answer"], "ai")


# f. main - ai's response
def respond(message, retriever):
    chain = search_history | RunnableLambda(new_answer)

    chain.invoke({"question": message, "retriever": retriever})


### Main
# api key widget
with st.sidebar:
    st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-...",
        key="api_key",
        disabled=bool(st.session_state.get("api_key")),
    )

    if st.session_state.get("api_key"):
        st.success("✅ API Key registered")


if not st.session_state.get("api_key"):
    st.markdown(
        """
        ## Please enter your OpenAI API key<br>on the sidebar to use DocumentGPT.
        """,
        unsafe_allow_html=True,
    )

else:
    os.environ["OPENAI_API_KEY"] = st.session_state["api_key"]

    if chat is None:
        chat = ChatOpenAI(
            temperature=0.1, streaming=True, callbacks=[ChatCallbackHandler()]
        )

    with st.sidebar:
        # url widget
        url = st.text_input(
            "Write down a URL", placeholder="https://example.com"
        )

    if url:
        if ".xml" not in url:
            with st.sidebar:
                st.error("")
        else:
            retriever = load_site(url)
            send_message("Website loaded successfully!", "ai", save=False)

            # 채팅 히스토리
            paint_history()

            # 메세지 입력창
            message = st.chat_input(
                "Ask me anything about the website you write down."
            )

            # 채팅
            if message:
                send_message(message, "human")
                respond(message, retriever)

    else:
        st.title("SiteGPT")
        st.markdown(
            """
        ## Write down an URL to ask questions about its content.
        """
        )
        st.session_state["messages"] = []
