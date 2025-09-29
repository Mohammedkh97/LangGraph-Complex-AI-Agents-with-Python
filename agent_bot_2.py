from typing import TypedDict, List, Union
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")


class AgentState(TypedDict):
    message: List[Union[HumanMessage, AIMessage]]
    # message_ai: List[AIMessage]
    # response: Optional[str]


# Define the LLM
llm = ChatOpenAI(
    # model="llama-3-8b-8192",
    model="openai/gpt-oss-120b",
    # model="deepseek-r1-distill-llama-70b",
    # model="llama3-8b-8192",
    api_key=API_KEY,  # type: ignore
    base_url="https://api.groq.com/openai/v1",
)


def process(state: AgentState) -> AgentState:
    """Node that handles the user message with the LLM"""
    response = llm.invoke(state["message"])
    state["message"].append(AIMessage(content=response.content))
    print(f"AI: {response.content}")
    print("CURRENT STATE: ", state["message"])
    # The state is modified in-place and returned
    return state


# --- Utility to save & load conversations ---
LOG_FILE = "logs/logging.txt"


def save_conversation(conversation: List[Union[HumanMessage, AIMessage]]):
    os.makedirs("logs", exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write("\n=== New Conversation ===\n")
        for message in conversation:
            if isinstance(message, HumanMessage):
                f.write(f"YOU: {message.content}\n")
            elif isinstance(message, AIMessage):
                f.write(f"AI: {message.content}\n")
        f.write("--- End of Conversation ---\n")


def load_last_conversation() -> List[Union[HumanMessage, AIMessage]]:
    if not os.path.exists(LOG_FILE):
        return []

    conversation: List[Union[HumanMessage, AIMessage]] = []
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Parse only the last conversation block
    if "=== New Conversation ===" in "".join(lines):
        last_block = "".join(lines).split("=== New Conversation ===")[-1]
        for line in last_block.splitlines():
            if line.startswith("YOU: "):
                conversation.append(
                    HumanMessage(content=line.replace("YOU: ", "").strip())
                )
            elif line.startswith("AI: "):
                conversation.append(AIMessage(content=line.replace("AI: ", "").strip()))
    return conversation


def main():
    # Build the graph
    graph = StateGraph(AgentState)
    graph.add_node("process", process)
    graph.add_edge(START, "process")
    graph.add_edge("process", END)
    agent = graph.compile()

    # Load last session’s memory (but keep hidden)
    conversation_history = load_last_conversation()

    print("✨ Welcome to the AI Agent!")
    print("Type 'exit' to quit.")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Exiting conversation.")
            save_conversation(conversation_history)
            break

        conversation_history.append(HumanMessage(content=user_input))
        result = agent.invoke({"message": conversation_history})

        # Update memory
        conversation_history = result["message"]


if __name__ == "__main__":
    main()
