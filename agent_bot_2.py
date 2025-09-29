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


def main():
    # Build the graph
    graph = StateGraph(AgentState)
    graph.add_node("process", process)
    graph.add_edge(START, "process")
    graph.add_edge("process", END)
    agent = graph.compile()

    # Conversation loop
    conversation_history: List[Union[HumanMessage, AIMessage]] = []
    print("Welcome to the AI Agent! Type 'exit' to quit.")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Exiting conversation.")
            break

        conversation_history.append(HumanMessage(content=user_input))
        result = agent.invoke({"message": conversation_history})

        # keep memory
        conversation_history = result["message"]

    # Save conversation to file (append mode)
    os.makedirs("logs", exist_ok=True)
    with open("logs/logging.txt", "a", encoding="utf-8") as f:
        # count conversations by adding a header each time
        f.write("\n=== New Conversation ===\n")
        for message in conversation_history:
            if isinstance(message, HumanMessage):
                f.write(f"You: {message.content}\n")
            elif isinstance(message, AIMessage):
                f.write(f"AI: {message.content}\n")
        f.write("--- End of Conversation ---\n")

    print("Conversation history appended to logs/logging.txt")


if __name__ == "__main__":
    main()
