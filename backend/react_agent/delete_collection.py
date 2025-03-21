import os
from dotenv import load_dotenv
from langchain.agents import initialize_agent, Tool
from langchain.agents.react.agent import create_react_agent
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_core.vectorstores import VectorStore
from langchain_qdrant.qdrant import QdrantVectorStore
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from qdrant_client import QdrantClient
from tavily import TavilyClient
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
import logging
import prompts
from langchain_core.prompts import PromptTemplate
from cxstore.context_vector_store import KXVectorStore
from react_agent.state import InputState, State

# Load environment variables
load_dotenv()

# Configuration
QDRANT_URL = os.getenv("QDRANT_URL", "http://172.20.0.30:6333")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "kx_embeddings")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://172.20.0.50:11434")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Initialize logging
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def delete_collection(collection_name: str) -> None:
    assert collection_name != None
    try:
        vs = KXVectorStore(
            url=QDRANT_URL,
            collection=QDRANT_COLLECTION_NAME,
        )
        response = vs.delete_collection(collection_name)
        response = vs._qdrant_client.delete_collection(collection_name)
        print(f"response to deletion of collection is {response}")
    except Exception as e:
        logger.error(f"Error while deleting collection - {e}")
        raise


# Main loop
def main():
    delete_collection("kx_collection")


# Run the application
if __name__ == "__main__":
    main()
