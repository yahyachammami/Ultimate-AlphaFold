# app.py
import streamlit as st
import time
import base64
import os
import streamlit.components.v1 as components  
from vectors import EmbeddingsManager  
from chatbot import ChatbotManager      

# Function to display the PDF of a given file
def displayPDF(file):
    base64_pdf = base64.b64encode(file.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

# Initialize session_state variables
if 'temp_pdf_path' not in st.session_state:
    st.session_state['temp_pdf_path'] = None
if 'chatbot_manager' not in st.session_state:
    st.session_state['chatbot_manager'] = None
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

st.set_page_config(page_title="Document Buddy App", layout="wide")

with st.sidebar:
    st.image("logo.png", use_container_width=True)
    st.markdown("### 📚 Your Personal Document Assistant")
    menu = ["🏠 Home", "🤖 Chatbot", "📧 Contact"]
    choice = st.selectbox("Navigate", menu)

if choice == "🏠 Home":
    st.title("📄 Document Buddy App")
    st.markdown("Welcome to Document Buddy! Upload, summarize, and chat with your documents.")

elif choice == "🤖 Chatbot":
    st.title("🤖 Chatbot Interface (Multimodal)")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.header("📂 Upload Document")
        uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
        if uploaded_file:
            st.success("📄 File Uploaded Successfully!")
            st.markdown(f"**Filename:** {uploaded_file.name}")
            st.markdown(f"**File Size:** {uploaded_file.size} bytes")
            displayPDF(uploaded_file)
            temp_pdf_path = "temp.pdf"
            with open(temp_pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.session_state['temp_pdf_path'] = temp_pdf_path

    with col2:
        st.header("🧠 Embeddings")
        if st.button("Create Embeddings"):
            if not st.session_state['temp_pdf_path']:
                st.warning("⚠️ Please upload a PDF first.")
            else:
                embeddings_manager = EmbeddingsManager(
                    model_name="BAAI/bge-small-en",
                    device="cpu",
                    encode_kwargs={"normalize_embeddings": True},
                    qdrant_url="http://localhost:6333",
                    collection_name="vector_db"
                )
                with st.spinner("🔄 Creating embeddings..."):
                    result = embeddings_manager.create_embeddings(st.session_state['temp_pdf_path'])
                    time.sleep(1)
                st.success(result)
                if not st.session_state['chatbot_manager']:
                    st.session_state['chatbot_manager'] = ChatbotManager()

    with col3:
        st.header("💬 Chat with Document")
        uploaded_image = st.file_uploader("Upload an image (optional)", type=["jpg", "png", "jpeg"])
        image_path = None
        if uploaded_image:
            image_path = f"temp_{uploaded_image.name}"
            with open(image_path, "wb") as f:
                f.write(uploaded_image.getbuffer())
            st.image(image_path, caption="Uploaded Image", use_column_width=True)

        if st.session_state['chatbot_manager']:
            for msg in st.session_state['messages']:
                st.chat_message(msg['role']).markdown(msg['content'])
            user_input = st.chat_input("Type your message here...")
            if user_input:
                st.chat_message("user").markdown(user_input)
                st.session_state['messages'].append({"role": "user", "content": user_input})
                with st.spinner("🤖 Responding..."):
                    answer = st.session_state['chatbot_manager'].get_response(user_input, image_path)
                st.chat_message("assistant").markdown(answer)
                st.session_state['messages'].append({"role": "assistant", "content": answer})

# Contact Page

elif choice == "📧 Contact":
    st.title("📬 Contact Us")
    
    # Full path to your HTML file
    html_file_path = "combined_interaction.html"
    
    # Check if the file exists
    if os.path.exists(html_file_path):
        # Read and display the HTML file using components.html()
        with open(html_file_path, "r", encoding="utf-8") as file:
            html_content = file.read()
            components.html(html_content, height=800, scrolling=True)
    else:
        st.error("⚠️ The contact HTML file could not be found.")
