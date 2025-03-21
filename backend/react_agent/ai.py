import os
import sys
from dotenv import load_dotenv
from langchain.tools import tool
from langchain.agents import initialize_agent, Tool
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_core.vectorstores import VectorStore
from langchain_qdrant.qdrant import QdrantVectorStore
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaLLM, ChatOllama
from langgraph.graph import StateGraph, END
from qdrant_client import QdrantClient
from tavily import TavilyClient
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
import logging
import prompts
from tools import retrieve_context, search, generate_question
from langchain_core.prompts import PromptTemplate
from cxstore.context_vector_store import KXVectorStore
from react_agent.state import InputState, State
from langgraph.prebuilt import create_react_agent
from langchain.tools import BaseTool
from pydantic import BaseModel
from langchain.agents import BaseSingleActionAgent
from langchain.schema import SystemMessage, AgentAction, AgentFinish
from typing import Optional, Union, TypedDict, Annotated
from typing import List, Tuple, Optional, Union, Any, Set, Sequence
from langchain.callbacks.base import BaseCallbackHandler, BaseCallbackManager
import asyncio
from langgraph.prebuilt import ToolNode
from langchain_anthropic import ChatAnthropic
import pprint as pp
from langchain.agents import initialize_agent, AgentType

# Load environment variables
load_dotenv()

# Configuration
QDRANT_URL = os.getenv("QDRANT_URL", "http://172.20.0.30:6333")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "doc_embeddings")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://172.20.0.50:11434")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
MODEL_ID = os.getenv("MODEL_ID", "smollm2")

# Initialize logging
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


question_generation_prompt_template = """
You are a tutor. Generate a {difficulty} level question for {grade} grade students
on the topic {topic} in the subject {subject}, following the {board} syllabus.
""".strip()

question_generation_prompt = PromptTemplate(
    template=question_generation_prompt_template,
    input_variables=[
        "subject",
        "topic",
        "difficulty",
        "board",
        "grade",
    ],
)


# Step 3: Define Tools for Question Generation and Evaluation


@tool
def generate_question(
    subject: str, topic: str, difficulty: str, grade: str, board: str
) -> str:
    """Generate the question based on the subject, topic, difficulty, grade and board of education"""
    return f"Generated question for {subject} - {topic} at {difficulty} level for {grade} grade ({board} board)."


@tool
def retrieve_context(query):
    """Fetch context from vector database"""
    vc = KXVectorStore(url=QDRANT_URL, collection=QDRANT_COLLECTION_NAME)
    docs = vc.get_docs(query_str=query, limit=1)
    context = "\n".join([doc.page_content for doc in docs])
    logger.info(f"Returning context {context}")
    return context if context else "No relevant context found."


llm = OllamaLLM(
    base_url=OLLAMA_URL,
    model=MODEL_ID,
)


class AgentState(TypedDict):
    subject: str
    topic: str
    difficulty: str
    grade: str
    board: str
    retrieved_context: str
    generated_question: str
    student_answer: str
    evaluation: str
    thought: str


def generate_question_step(state: AgentState) -> AgentState:
    """Generates a question based on retrieved context, subject, topic, and difficulty."""
    prompt_template = PromptTemplate(
        input_variables=["retrieved_context"],
        template="""
            Based on this context:{retrieved_context}
            Generate a {difficulty} level question for {grade} 
            grade on {topic}.""",
    )
    prompt = prompt_template.format(
        retrieved_context=state["retrieved_context"],
        difficulty=state["difficulty"],
        grade=state["grade"],
        topic=state["topic"],
    )

    generated_question = llm.invoke(prompt).strip()
    return {**state, "generated_question": generated_question}


def evaluate_answer_step(state: AgentState) -> AgentState:
    """Evaluate the answer by the student"""
    print("=" * 50)

    print(f"Question: {state['generated_question']}")
    student_answer = input("Your Answer: ")  # Get user input
    state["student_answer"] = student_answer  # Store the answer in the state
    print("-" * 50)
    prompt_template = prompts.answer_evaluation
    prompt = prompt_template.format(
        question=state["generated_question"], answer=state["student_answer"]
    )
    feedback = llm.invoke(prompt).strip()
    return {**state, "evaluation": feedback}


def retrieve_context_step(state: AgentState) -> AgentState:
    """Fetches relevant document context based on subject, topic, and grade."""
    query: str = (
        f"{state['subject']} {state['topic']} {state['difficulty']} {state['grade']} {state['board']}"
    )
    logging.info(f"retrieve_context_step() - Constructed query {query}")
    response = retrieve_context.invoke(query)

    return {**state, "retrieved_context": response}


def reasoning_step(state: AgentState) -> AgentState:
    """Thinks about the best way to answer using retrieved context."""
    prompt = ""
    try:
        print(f"STATE is {state}")
        if state.get("generated_question", "") == "":
            logger.info("The Question is not yet generated!")
            prompt = f"Context: {state['retrieved_context']}\nQuestion: \nHow should I proceed to generate a question for a student ?"
        elif state.get("student_answer", "") != "":
            logger.info(
                f"The Student has provided an answer {state['student_answer']}!"
            )
            prompt = f"Context: {state['retrieved_context']}\nThought: \nHow should I proceed to evaluate the answer now that I have the answer to the question?"
        else:
            logger.info("Something else... let me reason...")
            prompt = f"Context: {state['retrieved_context']}\nThought: \nHow should I proceed now, given the current context?"
    except KeyError as ke:
        logger.error(f"reasoning_step() - Caught {ke}")
        sys.exit(100)
    except Exception as e:
        logger.error(f"reasoning_step() - Caught {e}")
        sys.exit(100)

    thought = llm.invoke(prompt).strip()

    return {**state, "thought": thought}


# Main loop
def main():
    print("Welcome to the AI Tutor!")
    subject = "Physics"
    # topic = input("Enter the topic you want to learn about: ")
    topic = "Pulleys"
    # difficulty = input("Enter the difficulty level (easy, medium, hard): ")
    difficulty = "Hard"
    # grade = input("Enter your grade level (e.g., 5th, 6th, 7th): ")
    grade = "10th"
    # board = input("Enter your board (ICSE, CBSE): ")
    board = "ICSE"
    context = ""

    history = []
    question = None
    input = {
        "subject": subject,
        "topic": topic,
        "difficulty": difficulty,
        "board": board,
        "grade": grade,
    }

    workflow = StateGraph(AgentState)

    workflow.add_node("retrieve_context", retrieve_context_step)
    workflow.add_node("generate_question", generate_question_step)
    workflow.add_node("reasoning", reasoning_step)
    workflow.add_node("evaluate_answer", evaluate_answer_step)

    workflow.set_entry_point("retrieve_context")

    workflow.add_edge("retrieve_context", "reasoning")
    workflow.add_edge("reasoning", "generate_question")
    workflow.add_edge("generate_question", "evaluate_answer")
    workflow.add_edge("evaluate_answer", END)

    graph = workflow.compile()

    state: AgentState = {
        "subject": input["subject"],
        "topic": input["topic"],
        "difficulty": input["difficulty"],
        "board": input["board"],
        "grade": input["grade"],
        "retrieved_context": "",
        "generated_question": "",
        "student_answer": "",
        "evaluation": "",
        "thought": "",
    }

    output = dict()
    for output in graph.stream(state):
        print(f"OUTPUT = {output}")

    # question = output.get("generate_question", {}).get("generated_question", {})


# Run the application
if __name__ == "__main__":
    main()
