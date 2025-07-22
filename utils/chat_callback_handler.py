import streamlit as st

# utils
from utils.chatbot_session import ChatBotSession

# LangChain
from langchain.callbacks.base import BaseCallbackHandler


class ChatCallbackHandler(BaseCallbackHandler):
    def __init__(self, session: ChatBotSession):
        self.session = session
        self.ai_message = ""
        self.message_box = None

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_new_token(self, token, *args, **kwargs):
        self.ai_message += token
        self.message_box.write(self.ai_message)

    def on_llm_end(self, *args, **kwargs):
        self.session.save_message(self.ai_message, "ai")
