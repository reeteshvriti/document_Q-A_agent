import streamlit as st
import requests

st.title("DOC Q&A Agent")

# Input field for the question
question = st.text_input("Ask a question:")

# Button to trigger the query
if st.button("Submit") and question:
    try:
        response = requests.get(
            "http://127.0.0.1:9100/ask",
            params={"question": question}
        )
        result = response.json()
        answer = result.get("answer", "No answer returned")
        st.subheader("Answer:")
        st.write(answer)
    except Exception as e:
        st.error(f"Error: {e}")
