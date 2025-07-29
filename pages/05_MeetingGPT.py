from dotenv import load_dotenv
import streamlit as st
from streamlit_float import *
import os
import math

# utils
from utils import common
from utils.chat_callback_handler import ChatCallbackHandler
from utils.chatbot_session import ChatBotSession
from utils.docs_handler import DocsHandler

# File
from pydub import AudioSegment
import subprocess
import glob

# OpenAI
from openai import OpenAI

# LangChain - Chain
from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda

# LangChain - Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory

### Settings
load_dotenv()

# Set the page configuration
st.set_page_config(
    page_title="MeetingGPT",
    page_icon=":video_camera:",
)
float_init(theme=True)

# Initialize
openai = OpenAI()

docs_handler = DocsHandler(
    RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=800, chunk_overlap=100
    )
)

chatbot_session = ChatBotSession("meeting")
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
@st.cache_resource(show_spinner="Saving video...")
def save_video(video):
    # setting path
    video_folder = "./.cache/videos"
    common.check_dir(video_folder)

    # save uploaded video
    video_content = video.read()
    with open(f"{video_folder}/{video.name}", "wb") as f:
        f.write(video_content)

    return f"{video_folder}/{video.name}"


@st.cache_resource(show_spinner="Extracting audio...")
def extract_audio(video_path):
    audio_path = video_path.replace("videos", "audios").replace(".mp4", ".mp3")
    common.check_dir(os.path.dirname(audio_path))

    # -y: overwrite output files without asking
    # -i: input file
    # -vn: 비디오 인코딩 비활성화
    command = ["ffmpeg", "-y", "-i", video_path, "-vn", audio_path]

    # subprocess: 파이썬에서 명령어를 실행할 수 있게 해주는 모듈
    subprocess.run(command)

    return audio_path


@st.cache_resource(show_spinner="Splitting audio...")
def split_audio(audio_path, segment_size):
    # extracting file name
    audio_name = os.path.splitext(os.path.basename(audio_path))[0]

    # setting path
    segment_folder = "./.cache/audios/segments"
    common.check_dir(segment_folder)

    specific_segment_folder = f"{segment_folder}/{audio_name}"
    common.check_dir(specific_segment_folder)

    audio = AudioSegment.from_file(audio_path)
    segment_count = math.ceil(len(audio) / (segment_size * 1000))

    max_digits = math.ceil(segment_count / 10)
    for i in range(segment_count):
        index_digits = math.ceil(i / 10)
        if i % 10 == 0:
            index_digits += 1
        num = "0"
        for n in range(max_digits - index_digits):
            num += "0"
        num += str(i)

        start_time = i * segment_size * 1000
        end_time = start_time + (segment_size * 1000)
        segment = audio[start_time:end_time]

        segment.export(
            f"{specific_segment_folder}/segment_{num}.mp3",
            format="mp3",
        )

    return specific_segment_folder


@st.cache_resource(show_spinner="Transcribing audio...")
def transcribe_audio(audio_segments_folder, filename):
    text_folder = "./.cache/texts"
    common.check_dir(text_folder)

    text_path = f"{text_folder}/{filename}.txt"

    audio_files = glob.glob(f"{audio_segments_folder}/*.mp3")
    audio_files.sort()

    for audio_file in audio_files:
        with open(audio_file, "rb") as audio_file, open(
            text_path, "a"
        ) as text_file:
            transcription = openai.audio.transcriptions.create(
                model="gpt-4o-transcribe",
                file=audio_file,
                response_format="text",
            )
            text_file.write(transcription + " ")

    return text_path


@st.cache_resource(show_spinner="Creating summary...")
def generate_summary(text_path):
    llm = ChatOpenAI(temperature=0.1)

    docs = docs_handler.split_n_return_docs(text_path)

    initial_prompt = PromptTemplate.from_template(
        """
        Write a concise summary of the following.
        The summary should be in the language of the text.
        ------------
        {context}
        ------------
        Use the language of the text.
        """
    )
    initial_chain = initial_prompt | llm | StrOutputParser()
    summary = initial_chain.invoke({"context": docs[0].page_content})

    summary_prompt = PromptTemplate.from_template(
        """
        Product a final summary.
        The summary should be in the language of the text.

        Existing summary up to this point:
        {previous_summary}

        New context:
        ------------
        {context}
        ------------

        Given the new context, refine the original summary.
        """
    )
    summary_chain = summary_prompt | llm | StrOutputParser()

    for doc in docs[1:]:
        summary = summary_chain.invoke(
            {"previous_summary": summary, "context": doc.page_content}
        )

    return summary


def answer_question(question, file_path):
    llm = ChatOpenAI(temperature=0.1, streaming=True, callbacks=[chat_handler])

    retriever = docs_handler.embedding_n_return_retriever(file_path)

    qna_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                    You are a helpful assistant. Answer questions using only the following context.
                    If you don't know the answer, just say you don't know. Don't make it up.


                    Context:
                    {context}

                    Previous conversation:
                    {history}
                    """,
            ),
            ("human", "{question}"),
        ]
    )

    qna_chain = (
        (
            {
                "context": (
                    retriever | RunnableLambda(docs_handler.format_docs)
                ),
                "question": RunnablePassthrough(),
                "history": chatbot_session.load_memory,
            }
        )
        | qna_prompt
        | llm
        | StrOutputParser()
    )

    answer = qna_chain.invoke(question)
    chatbot_session.save_memory(question, answer)


### Main
transcription_tab, summary_tab, qna_tab = st.tabs(
    ["Transcription", "Summary", "Q&A"]
)

# File upload widget
with st.sidebar:
    uploaded_video = st.file_uploader(
        "Upload a video file (MP4 format)",
        type=["mp4"],
    )

if uploaded_video:
    video_path = save_video(uploaded_video)
    audio_path = extract_audio(video_path)
    specific_segment_folder = split_audio(
        audio_path, segment_size=30
    )  # 30 seconds
    text_path = transcribe_audio(
        specific_segment_folder, os.path.splitext(uploaded_video.name)[0]
    )

    with transcription_tab:
        transcription = open(text_path, "r").read()
        st.write(transcription)

    with summary_tab:
        if st.button("Generate Summary"):
            summary_text = generate_summary(text_path)
            st.write(summary_text)

    with qna_tab:
        # 채팅 히스토리
        chatbot_session.paint_messages()

        # 첫 안내
        chatbot_session.send_info(
            "You can ask questions about the uploaded video."
        )

        with st.container():
            # 질문 입력창
            question = st.chat_input(
                "Ask me anything about the uploaded video!"
            )

            # chat_input은 기본적으로 하단 고정이다. 하지만 tab, column 등에서는 그것이 적용되지 않는다.
            # 해서, container를 새로 만들어주고, 아래 css를 이용해 해당 container의 위치를 고정시키는 방식을 적용한다.
            # width는 50%를 했을 때 tab의 width와 사이즈가 일치했고, bottom은 숫자를 바꿔가며 직접 확인했는데 2rem 정도가 적당해 보였다.
            float_parent(
                css=float_css_helper(bottom="2rem", width="50%", transition=0)
            )

        # 채팅
        if question:
            chatbot_session.send_message(question, "human")
            with st.chat_message("ai"):
                answer_question(question, text_path)
