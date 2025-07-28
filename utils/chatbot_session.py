import streamlit as st


class ChatBotSession:
    def __init__(self, session_key: str):
        self.key = session_key
        if self.key not in st.session_state:
            st.session_state[self.key] = {"messages": [], "memory": None}

    def save_message(self, message: str, role: str):
        st.session_state[self.key]["messages"].append(
            {"role": role, "message": message}
        )

    def send_message(self, message: str, role: str, save: bool = True):
        with st.chat_message(role):
            st.write(message)
        if save:
            self.save_message(message, role)

    def send_info(self, message: str):
        if len(st.session_state[self.key]["messages"]) == 0:
            self.send_message(message, "ai")

    def paint_messages(self):
        for message in st.session_state[self.key]["messages"]:
            self.send_message(message["message"], message["role"], save=False)

    def set_memory(self, memory_object):
        st.session_state[self.key]["memory"] = memory_object

    def save_memory(self, input_message, output_message):
        if self.memory:
            self.memory.save_context(
                {"input": input_message}, {"output": output_message}
            )

    def load_memory(self, _):
        if self.memory:
            return self.memory.load_memory_variables({})["history"]
        return []

    @property
    def memory(self):
        if self.key not in st.session_state:
            st.session_state[self.key] = {"messages": [], "memory": None}
        return st.session_state[self.key]["memory"]
