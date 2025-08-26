import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="DOC Q&A Agent", layout="wide")
st.title(" DOC Q&A Agent")

# Sidebar for upload & delete
st.sidebar.header("📂 Document Management")

# Keep a session state for uploaded docs
if "docs" not in st.session_state:
    st.session_state.docs = {}

# ---- Upload PDF ----
uploaded_file = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])
if uploaded_file is not None:
    if st.sidebar.button("Submit File"):
        try:
            files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
            response = requests.post(f"{API_URL}/upload", files=files)
            if response.status_code == 200:
                result = response.json()
                doc_id = result.get("doc_id")
                filename = result.get("filename")
                st.session_state.docs[doc_id] = filename
                st.sidebar.success(f"✅ Uploaded: {filename}")
            else:
                st.sidebar.error(f"Upload failed: {response.text}")
        except Exception as e:
            st.sidebar.error(f"Error: {e}")

# ---- Delete Document ----
if st.session_state.docs:
    delete_doc_id = st.sidebar.selectbox(
        "Select document to delete",
        options=list(st.session_state.docs.keys()),
        format_func=lambda x: st.session_state.docs[x]
    )
    if st.sidebar.button("Delete Document"):
        try:
            response = requests.delete(f"{API_URL}/delete/{delete_doc_id}")
            if response.status_code == 200:
                deleted_name = st.session_state.docs.pop(delete_doc_id)
                st.sidebar.success(f"🗑️ Deleted: {deleted_name}")
            else:
                st.sidebar.error(f"Delete failed: {response.text}")
        except Exception as e:
            st.sidebar.error(f"Error: {e}")

# ---- Q&A Section ----
st.subheader(" Ask a Question from your uploaded Docs")
question = st.text_input("Enter your question:")

if st.button("Get Answer") and question:
    try:
        response = requests.get(f"{API_URL}/ask", params={"question": question})
        if response.status_code == 200:
            result = response.json()
            answer = result.get("answer", "⚠️ No answer returned")
            st.markdown("### 📝 Answer")
            st.write(answer)
        else:
            st.error(f"Ask failed: {response.text}")
    except Exception as e:
        st.error(f"Error: {e}")
