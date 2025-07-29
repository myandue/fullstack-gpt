import streamlit as st

# utils
from utils.chat_callback_handler import ChatCallbackHandler
from utils.chatbot_session import ChatBotSession

# LangChain - Chain
from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough

# LangChain - Document, Site Load
from langchain_community.document_loaders import SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

### Settings
# a. Set the page configuration
st.set_page_config(page_title="SiteGPT", page_icon="🔍")


# b. Initilization
# b-1. chat
chatbot_session = ChatBotSession("site")
chat_handler = ChatCallbackHandler(chatbot_session)


# b-2. memory
if chatbot_session.memory is None:
    chatbot_session.set_memory(
        ConversationBufferMemory(
            return_messages=True,
            memory_key="history",
        )
    )


### Functions
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


def search_history(inputs):
    llm = ChatOpenAI(temperature=0.1)

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

    search_history_chain = (
        {
            "history": chatbot_session.load_memory,
            "question": RunnablePassthrough(),
        }
        | search_history_prompt
        | llm
        | StrOutputParser()
    )

    inputs["answer"] = search_history_chain.invoke(inputs["question"])

    return inputs


def get_answers(inputs):
    llm = ChatOpenAI(temperature=0.1)

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

    answers_chain = answers_prompt | llm | StrOutputParser()

    docs = inputs["docs"]
    question = inputs["question"]

    return {
        "question": question,
        "answers": [
            {
                "answer": answers_chain.invoke(
                    {"context": doc.page_content, "question": question}
                ),
                "source": doc.metadata["source"],
            }
            for doc in docs
        ],
    }


def choose_answer(inputs):
    llm = ChatOpenAI(temperature=0.1, streaming=True, callbacks=[chat_handler])

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

    choose_chain = choose_prompt | llm | StrOutputParser()

    question = inputs["question"]
    answers = inputs["answers"]

    condensed = "\n\n".join(
        f"\n{answer['answer']}\nSource: {answer['source']}\n"
        for answer in answers
    )

    with st.chat_message("ai"):
        response = choose_chain.invoke(
            {"answers": condensed, "question": question}
        )
        chatbot_session.save_memory(question, response)


def get_answer(inputs):
    # history에 없을 경우 get new answers
    if inputs["answer"] == "False":
        retriever = inputs["retriever"]
        question = inputs["question"]

        get_answer_chain = (
            {
                "docs": retriever,
                "question": RunnablePassthrough(),
            }
            | RunnableLambda(get_answers)
            | RunnableLambda(choose_answer)
        )

        get_answer_chain.invoke(question)

    # history에 있을 경우, history에서 답변을 불러옴
    else:
        chatbot_session.send_message(inputs["answer"], "ai")


def respond_to_question(question, url):
    retriever = load_site(url)

    chain = search_history | RunnableLambda(get_answer)

    chain.invoke({"question": question, "retriever": retriever})


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
    with st.sidebar:
        # url widget
        url = st.text_input(
            "Write down a URL", placeholder="https://example.com"
        )

    if url:
        if ".xml" not in url:
            with st.sidebar:
                st.error("Please write down a valid sitemap URL.")

        else:
            # 채팅 히스토리
            chatbot_session.paint_messages()

            # 첫 안내
            chatbot_session.send_info("Website loaded successfully!")

            # 질문 입력창
            question = st.chat_input(
                "Ask me anything about the website you write down."
            )

            # 채팅
            if question:
                chatbot_session.send_message(question, "human")
                with st.chat_message("ai"):
                    respond_to_question(question, url)

    else:
        st.title("SiteGPT")
        st.markdown(
            """
                ## Write down an URL to ask questions about its content.
            """
        )
