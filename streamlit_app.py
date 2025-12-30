import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/chat"

st.set_page_config(page_title="Python Chatbot", layout="centered")
st.title("🧠 Python Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
user_input = st.chat_input("Ask something about Python...")

if user_input:
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.spinner("Thinking..."):
        response = requests.post(
            API_URL,
            json={"message": user_input},
            timeout=120
        )

    if response.status_code == 200:
        assistant_reply = response.json()["reply"]
    else:
        assistant_reply = "⚠️ Error: Backend not responding."

    st.session_state.messages.append(
        {"role": "assistant", "content": assistant_reply}
    )
    with st.chat_message("assistant"):
        st.markdown(assistant_reply)
