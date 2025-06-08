import streamlit as st
import os
import json

# File
from langchain_community.document_loaders import UnstructuredFileLoader

# Wikipedia
from langchain_community.retrievers import WikipediaRetriever

# Chat
from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableMap
from langchain.schema import BaseOutputParser

### Settings
# Set the page configuration
st.set_page_config(
    page_title="QuizGPT",
    page_icon=":question:",
)

# initialization
chat = None
docs = None
success_count = 0


### Functions
# file
@st.cache_resource(show_spinner="Loading file...")
def load_file(file):
    # setting path
    file_path = "./.cache/quiz_files"
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    # 업로드한 파일 저장
    file_content = uploaded_file.read()
    with open(f"{file_path}/{uploaded_file.name}", "wb") as f:
        f.write(file_content)

    # load
    file = UnstructuredFileLoader(f"{file_path}/{uploaded_file.name}")

    return file.load()


# document list -> string
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# wikipedia
@st.cache_resource(show_spinner="Searching Wikipedia...")
def search_wikipedia(topic):
    retriever = WikipediaRetriever(top_k_results=5)

    return retriever.get_relevant_documents(topic)


# output parser
class QuizOutputParser(BaseOutputParser):
    def parse(self, text):
        text = text.replace("```", "").replace("json", "").strip()
        return json.loads(text)


output_parser = QuizOutputParser()


# prompts
question_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
         You are a helpful assistant that creates quizzes from documents.
         Based ONLY the content of the provided documents, create a quiz with 10 questions to test the user's knowledge about the text.
         Questions' difficulty should be around {level}.
         Each question should have 4 multiple choice answers, with one correct answer and three of them must be wrong.
         Use (o) for the correct answer.

         Question examples:

         Question: What is the capital of France?
         Answers: Paris (o) | Seoul | London | New York

         Question: What is the largest planet in our solar system?
         Answers: Earth | Mars | Jupiter (o) | Saturn

         Question: What is the main ingredient in guacamole?
         Answers: Tomato | Avocado (o) | Potato | Onion

         Your turn!

         Context: {context}
         """,
        )
    ]
)

formatting_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
        You are a helpful assistant that formats quizzes.
        Format the quiz into JSON format.
        Answers with (o) are correct answers.

        Example Input:

        Question: What is the capital of France?
        Answers: Paris (o) | Seoul | London | New York

        Question: What is the largest planet in our solar system?
        Answers: Earth | Mars | Jupiter (o) | Saturn

        Question: What is the main ingredient in guacamole?
        Answers: Tomato | Avocado (o) | Potato | Onion

        Example Output:
        ```json
        {{
            "questions": [
                {{
                    "question": "What is the capital of France?",
                    "answers": [
                        {{"answer": "Paris", "correct": true}},
                        {{"answer": "Seoul", "correct": false}},
                        {{"answer": "London", "correct": false}},
                        {{"answer": "New York", "correct": false}}
                    ]
                }},
                {{
                    "question": "What is the largest planet in our solar system?",
                    "answers": [
                        {{"answer": "Earth", "correct": false}},
                        {{"answer": "Mars", "correct": false}},
                        {{"answer": "Jupiter", "correct": true}},
                        {{"answer": "Saturn", "correct": false}}
                    ]
                }},
                {{
                    "question": "What is the main ingredient in guacamole?",
                    "answers": [
                        {{"answer": "Tomato", "correct": false}},
                        {{"answer": "Avocado", "correct": true}},
                        {{"answer": "Potato", "correct": false}},
                        {{"answer": "Onion", "correct": false}}
                    ]
                }}
            ]
        }}
        ```

        Your turn!
        
        Questions: {context}
        """,
        ),
    ]
)


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
    # chain
    question_chain = (
        RunnableMap(
            {
                "context": lambda x: format_docs(x["docs"]),
                "level": lambda x: x["level"],
            }
        )
        | question_prompt
        | chat
    )
    formatting_chain = formatting_prompt | chat
    final_chain = (
        RunnableMap(
            {"docs": lambda x: x["docs"], "level": lambda x: x["level"]}
        )
        | {"context": question_chain}
        | formatting_chain
        | output_parser
    )

    return final_chain.invoke({"docs": _docs, "level": level})


## Main
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
    pass

else:
    # chat
    os.environ["OPENAI_API_KEY"] = st.session_state["api_key"]
    if chat is None:
        chat = ChatOpenAI(
            temperature=0.1,
        )

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
