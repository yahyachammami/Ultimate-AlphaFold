# chatbot.py
import os
import ollama
import base64
from PIL import Image
import io
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from langchain.chains import RetrievalQA
from langchain import PromptTemplate

class ChatbotManager:
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en",
        device: str = "cpu",
        encode_kwargs: dict = {"normalize_embeddings": True},
        llm_model: str = "llama3.2-vision",
        llm_temperature: float = 0.7,
        qdrant_url: str = "http://localhost:6333",
        collection_name: str = "vector_db",
    ):
        """
        Initializes the ChatbotManager with embedding models, retrieval, and vector store.
        """
        self.model_name = model_name
        self.device = device
        self.encode_kwargs = encode_kwargs
        self.llm_model = llm_model
        self.llm_temperature = llm_temperature
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name

        # Initialize Embeddings
        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": self.device},
            encode_kwargs=self.encode_kwargs,
        )

        # Initialize Qdrant client
        self.client = QdrantClient(url=self.qdrant_url, prefer_grpc=False)

        # Initialize the Qdrant vector store
        self.db = Qdrant(
            client=self.client,
            embeddings=self.embeddings,
            collection_name=self.collection_name
        )

        # Initialize the retriever
        self.retriever = self.db.as_retriever(search_kwargs={"k": 3})  # Retrieve top 3 relevant docs

        # Define the prompt template
        self.prompt_template = """Use the following retrieved context to answer the user's question. If you don't know, just say so.

Context: {context}
Question: {question}

Helpful Answer:
"""

        # Initialize the prompt
        self.prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=['context', 'question']
        )

    def preprocess_image(self, image_path: str) -> str:
        """
        Converts the image to Base64 format for compatibility with llama3.2-vision.
        """
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")
        return base64_image

    def get_response(self, query: str, image_path: str = None) -> str:
        """
        Processes the user's query using retrieval + LLM and returns a response.
        Handles both text from the document and image input for multimodal interaction.
        """
        try:
            # Retrieve relevant documents from the vector store
            retrieved_docs = self.retriever.get_relevant_documents(query)
            context = "\n".join([doc.page_content for doc in retrieved_docs])

            # Prepare messages for LLM
            messages = [{
                'role': 'user',
                'content': self.prompt.format(context=context, question=query),
            }]

            if image_path and os.path.exists(image_path):
                # Convert image to Base64 before sending it to the model
                base64_image = self.preprocess_image(image_path)
                messages[0]['images'] = [base64_image]

            # Send to Ollama LLM (text + image)
            response = ollama.chat(
                model=self.llm_model,
                messages=messages
            )
            return response.get("message", {}).get("content", "⚠️ No response received.")

        except Exception as e:
            return f"⚠️ An error occurred while processing your request: {e}"
