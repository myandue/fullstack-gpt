import streamlit as st

from api.user import login

st.set_page_config(
    page_title="Login",
    page_icon=":key:",
)

username = st.text_input("Username")
password = st.text_input("Password", type="password")
if st.button("Login"):
    try:
        response = login(username, password)
        st.session_state.access_token = response["access_token"]
        st.success("Login successful!")
    except Exception as e:
        st.error(f"Error: {e}")

if st.button("Sign Up"):
    st.switch_page("pages/sign_up.py")
