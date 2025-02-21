# app.py

import streamlit as st
from streamlit import session_state
import time
import base64
import os
import streamlit.components.v1 as components  
from vectors import EmbeddingsManager  # Import the EmbeddingsManager class
from chatbot import ChatbotManager     # Import the ChatbotManager class

# Function to display the PDF of a given file
def displayPDF(file):
    # Reading the uploaded file
    base64_pdf = base64.b64encode(file.read()).decode('utf-8')

    # Embedding PDF in HTML
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'

    # Displaying the PDF
    st.markdown(pdf_display, unsafe_allow_html=True)

# Initialize session_state variables if not already present
if 'temp_pdf_path' not in st.session_state:
    st.session_state['temp_pdf_path'] = None

if 'chatbot_manager' not in st.session_state:
    st.session_state['chatbot_manager'] = None

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

if 'embeddings_created' not in st.session_state:
    st.session_state['embeddings_created'] = False

if 'uploaded_image_path' not in st.session_state:
    st.session_state['uploaded_image_path'] = None

# Set the page configuration to wide layout and add a title
st.set_page_config(
    page_title="Document Buddy App",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar
with st.sidebar:
    # You can replace the URL below with your own logo URL or local image path
    st.image("C:/Users/yahya/Desktop/last version/Ultimate-AlphaFold/loogo.png", use_container_width=True)

    st.markdown("### 📚 Drug Interaction")
    st.markdown("---")
    
    # Navigation Menu
    menu = ["🏠 Home", "🤖 Chatbot", "📽️​ Vizualisation"]
    choice = st.selectbox("Navigate", menu)

# Custom Styling
st.markdown(
    """
    <style>
        .title {
            font-size: 2.5em;
            font-weight: bold;
            text-align: center;
            color: #4CAF50;
        }
        .subtitle {
            font-size: 1.5em;
            text-align: center;
            color: #666;
        }
        .highlight {
            background-color: #f9f9f9;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 3px 3px 12px rgba(0, 0, 0, 0.1);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Home Page
if choice == "🏠 Home":
    st.markdown('<p class="title">📄 Ultimate AlphaFold</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Explaining Drug Interactions With Multimodal RAG 🚀</p>', unsafe_allow_html=True)
    
    st.markdown(
        """
        <div class="highlight">
        Welcome to <b>Ultimate AlphaFold</b>, your AI-powered document assistant! 😊
        
        <b>Built using Open Source Stack:</b>
        - 🦙 Llama 3.2 Vision
        - 🧠 BGE Embeddings
        - 📦 Qdrant (running locally within a Docker Container)
        
        **Core Features:**
        - 📂 <b>Upload Documents</b>: Easily upload your PDF documents.
        - ✨ <b>Summarize</b>: Get concise, AI-generated summaries of your documents.
        - 💬 <b>Chat</b>: Interact with your documents through our intelligent chatbot.
        
        Experience the future of **Drug Discovery** with **Ultimate AlphaFold**! 🚀
        </div>
        """,
        unsafe_allow_html=True,
    )
# Chatbot Page
elif choice == "🤖 Chatbot":
    st.title("🤖 Ask the PDFs about the reason of the interaction.")
    st.markdown("---")
    
    # Create three columns
    col1, col2 = st.columns(2)

    # Column 1: File Uploader and Preview
    with col1:
        st.header("📂 Upload Document")
        uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
        if uploaded_file is not None:
            st.success("📄 File Uploaded Successfully!")
            # Display file name and size
            st.markdown(f"**Filename:** {uploaded_file.name}")
            st.markdown(f"**File Size:** {uploaded_file.size} bytes")
            
            # Display PDF preview using displayPDF function
            st.markdown("### 📖 PDF Preview")
            displayPDF(uploaded_file)
            
            # Save the uploaded file to a temporary location
            temp_pdf_path = "temp.pdf"
            with open(temp_pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Store the temp_pdf_path in session_state
            st.session_state['temp_pdf_path'] = temp_pdf_path

    # Column 2: Create Embeddings
    with col2:
        st.header("🧠 Embeddings")
        create_embeddings = st.checkbox("✅ Create Embeddings")
        if create_embeddings:
            if st.session_state['temp_pdf_path'] is None:
                st.warning("⚠️ Please upload a PDF first.")
            else:
                try:
                    # Initialize the EmbeddingsManager
                    embeddings_manager = EmbeddingsManager(
                        model_name="BAAI/bge-small-en",
                        device="cpu",
                        encode_kwargs={"normalize_embeddings": True},
                        qdrant_url="http://localhost:6333",
                        collection_name="vector_db",
                        #Images_collection_name="Images_vector_db"
                    )
                    
                    with st.spinner("🔄 Embeddings are in process..."):
                        # Create embeddings
                        result = embeddings_manager.create_embeddings(st.session_state['temp_pdf_path'])
                        time.sleep(1)  # Optional: To show spinner for a bit longer
                    st.success(result)
                    
                    # Set embeddings_created to True
                    st.session_state['embeddings_created'] = True
                    
                    # Initialize the ChatbotManager after embeddings are created
                    if st.session_state['chatbot_manager'] is None:
                        st.session_state['chatbot_manager'] = ChatbotManager(
                            model_name="BAAI/bge-small-en",
                            device="cpu",
                            encode_kwargs={"normalize_embeddings": True},
                            llm_model="llama-3.2-11b-vision-preview",
                            llm_temperature=0.7,
                            qdrant_url="http://localhost:6333",
                            collection_name="vector_db"
                        )
                    
                except FileNotFoundError as fnf_error:
                    st.error(fnf_error)
                except ValueError as val_error:
                    st.error(val_error)
                except ConnectionError as conn_error:
                    st.error(conn_error)
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")

    # Column 3: Chatbot Interface

    st.header("💬 Chat with Document")
    
    if st.session_state['chatbot_manager'] is None:
        st.info("🤖 Please upload a PDF and create embeddings to start chatting.")
    else:
        # Display existing messages
        for msg in st.session_state['messages']:
            st.chat_message(msg['role']).markdown(msg['content'])

        # Image upload (manual upload before chat)
        if st.session_state['uploaded_image_path'] is None:
            uploaded_image = st.file_uploader("Upload an image of the Drug interactions", type=["jpg", "jpeg", "png"])
            if uploaded_image is not None:
                st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)
            
                st.success("🖼️ Image uploaded successfully!")

        # User input
        if user_input := st.chat_input("Type your message here..."):
            # Display user message
            st.chat_message("user").markdown(user_input)
            st.session_state['messages'].append({"role": "user", "content": user_input})

            with st.spinner("🤖 Responding..."):
                try:
                    # Get the chatbot response using the ChatbotManager
                    answer = st.session_state['chatbot_manager'].get_response(
                        user_input, 
                        image_path=st.session_state['uploaded_image_path']
                    )
                    time.sleep(1)  # Simulate processing time
                except Exception as e:
                    answer = f"⚠️ An error occurred while processing your request: {e}"
            
            # Display chatbot message
            st.chat_message("assistant").markdown(answer)
            st.session_state['messages'].append({"role": "assistant", "content": answer})

# Contact Page

elif choice == "📽️​ Vizualisation":
    st.title("Protein and Drug Interaction Visualization")
    
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