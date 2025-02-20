import os
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_core.runnables import RunnableLambda
import streamlit as st
from groq import Groq  # Using the official Groq Python library
 
class ChatbotManager:
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en",
        device: str = "cpu",
        encode_kwargs: dict = {"normalize_embeddings": True},
        llm_model: str = "llama-3.2-11b-vision-preview",  # Updated model per documentation
        llm_temperature: float = 0.7,
        qdrant_url: str = "http://localhost:6333",
        collection_name: str = "vector_db",
    ):
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
        self.client = QdrantClient(
            url=self.qdrant_url, prefer_grpc=False
        )
 
        # Initialize the Qdrant vector store
        self.db = Qdrant(
            client=self.client,
            embeddings=self.embeddings,
            collection_name=self.collection_name
        )
 
        # Define the prompt template
        self.prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
 
Context: {context}
Question: {question}
 
Only return the helpful answer. Answer must be detailed and well explained.
Helpful answer:
"""
 
        # Initialize the prompt
        self.prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=['context', 'question']
        )
 
        # Initialize the retriever
        self.retriever = self.db.as_retriever(search_kwargs={"k": 1})
 
        # Define chain type kwargs
        self.chain_type_kwargs = {"prompt": self.prompt}
 
        # Create a Runnable wrapper for the Groq LLM
        groq_llm = RunnableLambda(self._groq_llm)
 
        # Initialize the RetrievalQA chain with return_source_documents=False
        self.qa = RetrievalQA.from_chain_type(
            llm=groq_llm,  # Using the wrapped Groq API-based LLM
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=False,
            chain_type_kwargs=self.chain_type_kwargs,
            verbose=False
        )
 
    def _groq_llm(self, prompt: str, **kwargs) -> str:
        """
        Calls the Groq API to generate a response using the Groq Python library.
        """
        from groq import Groq  # Ensure the import is here or at the top of your file
        # Convert prompt to a string if it's not already
        prompt_text = str(prompt)
    
        # Initialize the Groq client using the API key from environment variables.
        client = Groq(api_key="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")
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt_text}  # Use the string version
            ],
            model=self.llm_model,
            temperature=self.llm_temperature,
        )
        return chat_completion.choices[0].message.content.strip()
 
    def get_response(self, query: str) -> str:
        """
        Processes the user's query and returns the chatbot's response.
        """
        try:
            response = self.qa.run(query)
            return response  # 'response' is a string containing the result
        except Exception as e:
            st.error(f"⚠️ An error occurred while processing your request: {e}")
            return "⚠️ Sorry, I couldn't process your request at the moment."