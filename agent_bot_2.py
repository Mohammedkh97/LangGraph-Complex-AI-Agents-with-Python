from typing import TypedDict, List, Union
import os
from typing import List, TypedDict, Union

from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph import END, StateGraph, START
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")


class AgentState(TypedDict):
    message: List[Union[HumanMessage, AIMessage]]
    # message_ai: List[AIMessage]
    # response: Optional[str]


# Define the LLM via LangChain but point it to Groq
llm = ChatOpenAI(
    # model="llama-3-8b-8192",
    model="openai/gpt-oss-120b",
    # model="deepseek-r1-distill-llama-70b",
    # model="llama3-8b-8192",
    api_key=API_KEY,  # type: ignore
    base_url="https://api.groq.com/openai/v1",
)


def process(state: AgentState) -> AgentState:
    """THis node handles the requests you request"""
    """This node invokes the LLM to get a response."""
    response = llm.invoke(state["message"])
    state["message"].append(AIMessage(content=response.content))
    # print(f"AI: {response.content}")
    print("CURRENT STATE: ", state["message"])
    print("CURRENT STATE: ", state["message"])
    # The state is modified in-place and returned
    return state


graph = StateGraph(AgentState)


def main():
    """Main function to run the conversational agent."""
    graph = StateGraph(AgentState)
    graph.add_node("process", process)
    graph.add_edge(START, "process")
    graph.add_edge("process", END)
    agent = graph.compile()


graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)

agent = graph.compile()
conversation_history: List[Union[HumanMessage, AIMessage]] = []

conversation_history = []
print("Welcome to the AI Agent! Type 'exit' to quit.")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Exiting conversation.")
        break

user_input = input("Enter: ")
while user_input != "exit":
    conversation_history.append(HumanMessage(content=user_input))
    result = agent.invoke({"message": conversation_history})
    ai_reply = result["message"][-1].content
    print(f"AI: {ai_reply}")
    conversation_history.append(HumanMessage(content=user_input))

    # show only AI's reply, clean
    ai_reply = result["message"][-1].content
    print(f"AI: {ai_reply}")
    conversation_history = result["message"]
    # keep memory
    conversation_history = result["message"]
    user_input = input("Enter: ")

    # The agent will receive the full history and append the AI's response
    result = agent.invoke({"message": conversation_history})

    # The result contains the updated history, so we just update our local copy
    conversation_history = result["message"]
    ai_reply = conversation_history[-1].content
    print(f"AI: {ai_reply}")


if __name__ == "__main__":
    main()
