import streamlit as st

st.set_page_config(
    page_title="Home",
    page_icon=":house:",
)

st.markdown(
    """
    # Hello!

    ## Welcome to my FullstackGPT Portfolio App!
    """
)


if "api_key" not in st.session_state:
    st.session_state.api_key = ""

if not st.session_state.get("api_key"):
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


def render_clickable_card(title, path):
    st.markdown(
        f"""
    <a href="{path}" target="_self" style="text-decoration: none;">
        <div style="
            border-radius: 12px;
            padding: 30ps;
            height: 140px;
            width: 100%;
            text-align:center;
            background-color: #f0f2f6;
            transition: 0.2s;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        " onmouseover="this.style.backgroundColor='#e0e0e0'" onmouseout="this.style.backgroundColor='#f0f2f6'">
            <h3 style="color: #333;">{title}<br></h3>
        </div>
    </a>
    """,
        unsafe_allow_html=True,
    )


row1 = st.columns(3)
st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
row2 = st.columns(3)

# TODO: 각 기능들 설명 추가
titles_and_paths = [
    ("DocumentGPT", "/DocumentGPT"),
    ("PrivateGPT", "/PrivateGPT"),
    ("QuizGPT", "/QuizGPT"),
    ("SiteGPT", "/SiteGPT"),
    ("MeetingGPT", "/MeetingGPT"),
    ("InvestorGPT", "/InvestorGPT"),
]

grids = row1 + row2
for i, (title, path) in enumerate(titles_and_paths):
    with grids[i]:
        render_clickable_card(title, path)
