import streamlit as st
import json

# utils
from utils.docs_handler import DocsHandler

# LangChain - Chain
from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableMap

# LangChain - Wikipedia
from langchain_community.retrievers import WikipediaRetriever

### Settings
# a. Set the page configuration
st.set_page_config(
    page_title="QuizGPT",
    page_icon=":question:",
)

# b. Initialization
docs_handler = DocsHandler()

docs = None
success_count = 0

function = {
    "name": "make_quiz",
    "description": (
        """
            function that takes a list of questions and answers and returns a quiz
        """
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "answers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "answer": {"type": "string"},
                                    "correct": {"type": "boolean"},
                                },
                                "required": ["answer", "correct"],
                            },
                        },
                    },
                    "required": ["question", "answers"],
                },
            }
        },
        "required": ["questions"],
    },
}


### Functions
def load_file(file):
    text_path = docs_handler.save_text_file(file)

    return docs_handler.load_file(text_path)


@st.cache_resource(show_spinner="Searching Wikipedia...")
def search_wikipedia(topic):
    retriever = WikipediaRetriever(top_k_results=5)

    return retriever.get_relevant_documents(topic)


### caching 에 대한 기록
# @st.cache_resource는 함수의 매개변수를 해싱하고, 그것을 기반으로 캐시를 저장한다.
# 하지만 매개변수가 해싱할 수 없는 형태일 수 있다. 그럴 경우 streamlit에게 해당 매개변수를 해싱하지 않도록 알려주어야한다.
# 매개변수 앞에 underscore(_)를 붙이면 streamlit은 해당 매개변수를 해싱하지 않는다.
# 그렇게 될 경우, docs가 변화하더라도 make_quiz 함수는 캐싱된 같은 결과만을 반환한다.
# 때문에, 사용하지 않더라도 topic 매개변수를 추가해준 것이다.
# make_quiz 함수를 호출 할 때, 파일업로드할 경우에는 uploaded_file.name을, 위키피디아 검색을 할 경우에는 검색한 topic을 넘겨준다.
# 위 수단을 추가해줌으로써 docs가 변화한 것을 인지할 수 있고, 새로운 quiz를 생성할 수 있다.


@st.cache_resource(show_spinner="Creating quiz...")
def make_quiz(_docs, topic, level):
    llm = ChatOpenAI(temperature=0.1)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
             You are a helpful assistant that creates quizzes from documents.
             Based ONLY the content of the provided documents, create a quiz with 10 questions to test the user's knowledge about the text.
             Questions' difficulty should be around {level}.
             Each question should have 4 multiple choice answers, with one correct answer and three of them must be wrong.

             Context: {context}
             """,
            )
        ]
    )

    chain = (
        RunnableMap(
            {
                "context": lambda x: docs_handler.format_docs(x["docs"]),
                "level": lambda x: x["level"],
            }
        )
        | prompt
        | llm.bind(function_call={"name": "make_quiz"}, functions=[function])
    )

    return json.loads(
        chain.invoke({"docs": _docs, "level": level}).additional_kwargs[
            "function_call"
        ]["arguments"]
    )


## Main
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
        # for quiz
        mode = st.selectbox(
            "Select a mode", options=["File", "Wikipedia Article"]
        )

        if mode == "File":
            uploaded_file = st.file_uploader(
                "Upload a .pdf .txt or .docx file", type=["pdf", "txt", "docx"]
            )

            if uploaded_file:
                docs = load_file(uploaded_file)

        else:
            topic = st.text_input("Search Wikipedia...")

            if topic:
                docs = search_wikipedia(topic)


if not docs:
    st.markdown(
        """
            # Welcome to QuizGPT!

            I will make a quiz from Wikipedia articles or files you upload to test your knowledge and help you learn.

            First, please enter your OpenAI API key in the sidebar to use QuizGPT.

            And then, select a mode from the sidebar and provide the necessary input.
        """
    )
else:
    # difficulty
    level = st.selectbox(
        "Select a difficulty level",
        options=["Easy", "Normal", "Hard"],
        index=None,
    )

    if level:
        # make quiz
        quiz = make_quiz(
            docs,
            topic if mode == "Wikipedia Article" else uploaded_file.name,
            level,
        )

        # display quiz
        with st.form("quiz_form"):
            for idx, question in enumerate(quiz["questions"], 1):
                st.markdown(f"**{idx}. {question['question']}**")
                value = st.radio(
                    "-options-",
                    options=[
                        answer["answer"] for answer in question["answers"]
                    ],
                    index=None,
                )
                st.markdown("<br>", unsafe_allow_html=True)

                if {"answer": value, "correct": True} in question["answers"]:
                    success_count += 1
                    st.success("Correct!")
                elif value is not None:  # 답을 선택했을 때만 피드백
                    st.error("Incorrect. Try again!")

            st.form_submit_button("Submit")

        if success_count == len(quiz["questions"]):
            st.balloons()
