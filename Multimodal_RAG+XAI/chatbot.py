import os
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_core.runnables import RunnableLambda
import streamlit as st
import requests

class ChatbotManager:
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en",
        device: str = "cpu",
        encode_kwargs: dict = {"normalize_embeddings": True},
        llm_url: str = "https://api.groq.com/openai/v1/chat/completions",  # Groq API URL
        llm_token: str = "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",  # Replace with your Groq API key
        llm_model: str = "llama-3.2-11b-vision-preview",  # Example Groq model
        llm_temperature: float = 0.7,
        qdrant_url: str = "http://localhost:6333",
        collection_name: str = "vector_db",
    ):
        self.model_name = model_name
        self.device = device
        self.encode_kwargs = encode_kwargs
        self.llm_url = llm_url
        self.llm_token = llm_token
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
            llm=groq_llm,  # Use the wrapped Groq API-based LLM
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=False,
            chain_type_kwargs=self.chain_type_kwargs,
            verbose=False
        )

    def _groq_llm(self, prompt: str, **kwargs) -> str:
        """
        Calls the Groq API to generate a response.
        """
        headers = {
            'Authorization': f'Bearer {self.llm_token}',
            'Content-Type': 'application/json'
        }

        if hasattr(prompt, 'text'):
            prompt_text = prompt.text
        else:
            prompt_text = str(prompt)

        payload = {
            "model": self.llm_model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt_text}
            ],
            "temperature": self.llm_temperature,
            "max_tokens": 5000
        }

        try:
            response = requests.post(self.llm_url, headers=headers, json=payload)
            response.raise_for_status()
            response_json = response.json()
            response_text = response_json.get("choices", [{}])[0].get("message", {}).get("content", "")
            return response_text.strip()
        except requests.exceptions.HTTPError as e:
            st.error(f"⚠️ HTTP Error {response.status_code} from Groq API: {e}")
            st.error(f"Response details: {response.text}")
            return f"⚠️ API Error: {response.status_code} - {response.text}"
        except requests.exceptions.RequestException as e:
            st.error(f"⚠️ Connection Error while calling Groq API: {e}")
            return "⚠️ Connection Error: Could not reach the API server."
        except Exception as e:
            st.error(f"⚠️ Unexpected error while processing API response: {e}")
            return "⚠️ Unexpected Error: Please try again later."

    def get_response(self, query: str) -> str:
        """
        Processes the user's query and returns the chatbot's response.
        """
        try:
            response = self.qa.run(query)
            return response  # 'response' is now a string containing only the 'result'
        except Exception as e:
            st.error(f"⚠️ An error occurred while processing your request: {e}")
            return "⚠️ Sorry, I couldn't process your request at the moment."
