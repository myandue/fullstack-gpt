import os
import streamlit as st

### Settings
# a. Set the page configuration
st.set_page_config(
    page_title="Home",
    page_icon=":house:",
)

# b. css
st.markdown(
    """
        <style>
        div.stHorizontalBlock div.stButton > button {
            border-radius: 12px;
            padding: 30px;
            height: 140px;
            width: 100%;
            text-align:center;
            background-color: #f0f2f6;
            transition: 0.2s;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }
        
        onmouseover: "this.style.backgroundColor='#e0e0e0'"
        onmouseout: "this.style.backgroundColor='#f0f2f6'"
        
        div.stHorizontalBlock div.stButton > button :hover {
            background-color: #105d8c;
        }
        </style>
    """,
    unsafe_allow_html=True,
)


### Functions
def render_page(page):
    st.session_state.target_page = page


### Main
st.markdown(
    """
    # Hello!

    ## Welcome to my FullstackGPT Portfolio App!
    """
)

if "access_token" not in st.session_state:
    st.session_state.access_token = None


if "api_key" not in st.session_state:
    st.session_state.api_key = None

if not st.session_state.get("access_token"):
    st.markdown(
        """
        ### Please login to use various functions.<br>You can use various functions after login.<br>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Sign Up"):
        st.switch_page("pages/sign_up.py")
    if st.button("Login"):
        st.switch_page("pages/login.py")


elif not st.session_state.get("api_key"):
    st.markdown(
        """
        ### Please enter your OpenAI API key on the sidebar.<br>You can use various functions.<br>
        """,
        unsafe_allow_html=True,
    )
    st.session_state.api_key = ""

    with st.sidebar:
        st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="sk-...",
            key="api_key_input",
            disabled=bool(st.session_state.get("api_key")),
        )

        if st.button("Register Key"):
            if st.session_state.api_key_input.startswith("sk-"):
                st.session_state.api_key = st.session_state.api_key_input
                st.success("✅ API Key registered")
            else:
                st.warning("❗Invalid API Key format")

else:
    os.environ["OPENAI_API_KEY"] = st.session_state["api_key"]


# TODO: 각 기능들 설명 추가
titles_and_paths = [
    ("DocumentGPT", "pages/01_DocumentGPT.py"),
    ("PrivateGPT", "pages/02_PrivateGPT.py"),
    ("QuizGPT", "pages/03_QuizGPT.py"),
    ("SiteGPT", "pages/04_SiteGPT.py"),
    ("MeetingGPT", "pages/05_MeetingGPT.py"),
    ("InvestorGPT", "pages/06_InvestorGPT.py"),
]

row1 = st.columns(3)
row2 = st.columns(3)
grids = row1 + row2

for i, (title, path) in enumerate(titles_and_paths):
    with grids[i]:
        # on_click은 button을 눌렀을 때, 호출되는 콜백함수이다.
        # st.switch_page는 내부에 st.rerun()을 포함하고 있으며 이 때문에 콜백 함수로서 호출하게 되면 오류가 발생한다.
        # 따라서, st.session_state에 target_page를 저장하고, 이후에 st.switch_page를 호출한다.
        st.button(title, on_click=render_page, args=(path,))

if st.session_state.get("target_page"):
    # session_state의 target_page를 초기화해주지 않으면,
    # 페이지 이동 후, 다시 Home.py로 돌아왔을 때, 이전에 클릭했던 페이지로 이동하게 된다.
    target_page = st.session_state.target_page
    st.session_state.target_page = None
    st.switch_page(target_page)
