import streamlit as st

import subprocess
from pydub import AudioSegment
import math
import os
import glob
from openai import OpenAI

st.title("MeetingGPT")
openai = OpenAI()


def check_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


@st.cache_resource(show_spinner="Saving video...")
def save_video(video):
    # setting path
    video_folder = "./.cache/videos"
    check_dir(video_folder)

    # save uploaded video
    video_content = video.read()
    with open(f"{video_folder}/{video.name}", "wb") as f:
        f.write(video_content)

    return f"{video_folder}/{video.name}"


@st.cache_resource(show_spinner="Extracting audio...")
def extract_audio(video_path):
    audio_path = video_path.replace("videos", "audios").replace(".mp4", ".mp3")
    check_dir(os.path.dirname(audio_path))

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
    check_dir(segment_folder)

    specific_segment_folder = f"{segment_folder}/{audio_name}"
    check_dir(specific_segment_folder)

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
    check_dir(text_folder)

    text_path = f"{text_folder}/{filename}.txt"

    audio_files = glob.glob(f"{audio_segments_folder}/*.mp3")
    audio_files.sort()
    print(audio_files)
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


### Main

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

    transcription = open(text_path, "r").read()
    st.write(transcription)
