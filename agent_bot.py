from typing import TypedDict, List, Optional
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")

class AgentState(TypedDict):
    message: List[HumanMessage]
    response: Optional[str]

# Define the LLM via LangChain but point it to Groq
llm = ChatOpenAI(
    # model="llama-3-8b-8192",
    model="openai/gpt-oss-120b",
    # model="deepseek-r1-distill-llama-70b",
    api_key=API_KEY,  # type: ignore
    base_url="https://api.groq.com/openai/v1",
)


def process(state: AgentState) -> AgentState:
    response = llm.invoke(state["message"])
    print(f"AI: {response.content}")
    state["response"] = response.content #type: ignore
    return state


graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)

agent = graph.compile()

# Run
user_input = input("Enter: ")
agent.invoke({"message": [HumanMessage(content=user_input)]}) # type: ignore
