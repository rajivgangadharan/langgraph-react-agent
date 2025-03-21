"""This module provides example tools for web scraping and search functionality.

It includes a basic Tavily search function (as an example)

These tools are intended as free examples to get started. For production use,
consider implementing more robust and specialized tools tailored to your needs.
"""

from typing import Any, Callable, List, Optional, Dict, cast

from langchain_community.tools.tavily_search import TavilySearchResults
from tavily import TavilyClient
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg
from typing_extensions import Annotated

from react_agent.configuration import Configuration

from langchain.tools import tool
from langchain.agents import initialize_agent, Tool
from langchain.agents.react.agent import create_react_agent
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_qdrant.qdrant import QdrantVectorStore
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from qdrant_client import QdrantClient
import os
import prompts
import logging
from dotenv import load_dotenv
from langchain.agents.agent import AgentExecutor
from langchain_core.runnables.config import RunnableConfig
from cxstore.context_vector_store import KXVectorStore

load_dotenv()
# Configuration
QDRANT_URL = os.getenv("QDRANT_URL", "http://172.20.0.30:6333")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "kx_embeddings")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://172.20.0.50:11434")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Initialize logging
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Initialize Qdrant client
# qdrant_client = QdrantClient(url=QDRANT_URL)
# embedding_model = OllamaEmbeddings(base_url=OLLAMA_URL, model="smollm2")
# vector_store = QdrantVectorStore(
#    client=qdrant_client,
#    collection_name=QDRANT_COLLECTION_NAME,
#    embedding=embedding_model,
# )
#
# Initialize LLM
llm = OllamaLLM(base_url=OLLAMA_URL, model="smollm2")


async def search(
    query: str, *, config: Annotated[RunnableConfig, InjectedToolArg]
) -> Dict[str, list[dict[str, Any]]]:
    """Search for general web results.

    This function performs a search using the Tavily search engine, which is designed
    to provide comprehensive, accurate, and trusted results. It's particularly useful
    for answering questions about current events.
    """
    configuration = Configuration.from_runnable_config(config)
    wrapped = TavilySearchResults(max_results=configuration.max_search_results)
    result = await wrapped.ainvoke({"query": query})

    return {"results": cast(list[dict[str, Any]], result)}


def retrieve_context(
    topic: str, difficulty: str, board: str, grade: str, context: str
) -> Dict[str, str]:
    """
    Generate a question based on the content in the PDF and internet search.

    Args:
        topic (str): The topic selected by the student.

    Returns:
        str: A context generated from the PDF content.
    """
    vs = KXVectorStore(
        url=QDRANT_URL,
        collection=QDRANT_COLLECTION_NAME,
    )

    query = f"Topic {topic} difficulty {difficulty} board {board} grade {grade} context {context}"
    # Retrieve relevant chunks from Qdrant
    documents = vs.get_docs(query)
    context = "\n".join([document.page_content for document in documents])
    logger.info(f"Context: {context}")

    return {"context": context}


def generate_question(
    topic: str, difficulty: str, board: str, grade: str, context: str
) -> Dict[str, str]:
    # Retrieve relevant chunks from Qdrant

    context = (
        context
        + f"""
        Topic: {topic} Difficulty: {difficulty} Board: {board} Grade: {grade} Context: {context}
    """
    )

    # Generate a question using the LLM
    prompt = PromptTemplate(
        template=prompts.question_generation,
        input_variables=["topic", "difficulty", "board", "grade", "context"],
    )
    question = llm(
        prompt.format(
            topic=topic,
            difficulty=difficulty,
            board=board,
            grade=grade,
            context=context,
        )
    )
    logger.info(f"Question: {question}")

    return {"question": question}


def evaluate_answer(question: str, answer: str) -> str:
    """
    Evaluate the student's answer and provide feedback.

    Args:
        question (str): The question asked.
        answer (str): The student's answer.

    Returns:
        dict: A dictionary containing the grade and feedback.
    """
    # Evaluate the answer using the LLM
    prompt = PromptTemplate(
        template=prompts.answer_evaluation,
        input_variables=["question", "answer"],
    )
    evaluation = llm(prompt.format(question=question, answer=answer))
    logger.info("Evaluation is {evaluation}")

    return evaluation


TOOLS: Dict[str, Callable] = {
    "retrieve_context": retrieve_context,
    "search": search,
    "generate_question": generate_question,
    "evaluate_answer": evaluate_answer,
}
