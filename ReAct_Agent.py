from typing import TypedDict, List, Union, Annotated, Sequence
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    BaseMessage,
    SystemMessage,
    ToolMessage,
)

from langchain_groq import ChatGroq
from langchain_core.tools import tool, BaseTool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv
import os
import sys

# Ensure UTF-8 encoding for Windows console
if sys.platform == "win32":
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


@tool
def add(a: int, b: int) -> int:
    """This is an addition function that adds 2 numbers together"""

    return a + b


@tool
# @tool(description="Subtract two numbers")
def subtract(a: int, b: int) -> int:
    """This is an subtraction function that subtracts 2 numbers together"""  # docstring is important to llm to understand what the function does. Function must have a docstring if description not provided.
    return a - b


@tool
def multiply(a: int, b: int) -> int:
    """This is an multiplication function that multiplies 2 numbers together"""
    return a * b


@tool
def divide(a: float, b: float) -> float:
    """This is a division function that divides two numbers. It handles float results."""
    if b == 0:
        return "Error: Division by zero"
    return a / b


@tool
def power(a: float, b: float) -> float:
    """This is a power function that raises a number to the power of another."""
    return a**b


tools = [add, subtract, multiply, divide, power]


llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=API_KEY,  # type: ignore
).bind_tools(tools)


def llm_model(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(
        content=(
            "You are a helpful AI assistant with access to mathematical tools. "
            "Solve the user's query step-by-step using the provided tools. "
            "Always explain your reasoning before using a tool."
        )
    )

    messages = state["messages"]
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [system_prompt] + messages  # type: ignore

    response = llm.invoke(messages)
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    """Determines whether to continue to call tools or end the process."""
    messages = state["messages"]
    last_message = messages[-1]
    # If the LLM didn't call a tool, we end the conversation
    if not last_message.tool_calls:
        return "end"
    # Otherwise, we continue to the tool node
    else:
        return "continue"


graph = StateGraph(AgentState)
tool_node = ToolNode(tools)

graph.add_node("agent", llm_model)
graph.add_node("tools", tool_node)

graph.set_entry_point("agent")
# graph.add_edge(START, "agent")
graph.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",
        "end": END,
    },
)
graph.add_edge("tools", "agent")

agent = graph.compile()

# Run
# def run_agent(stream):
#     # Example of how to run it
#     # inputs = {"messages": [HumanMessage(content="what is 5 plus 5  and 10 minus 5?")]}
#     # for output in agent.stream(inputs):
#     #     # stream() yields dictionaries with output from each step
#     #     for key, value in output.items():
#     #         print(f"Output from node '{key}':")
#     #         print("---")
#     #         print(value)
#     #     print("\n---\n")
#     for s in stream:
#         print(s)
#         if isinstance(s, dict) and "agent" in s:
#             messages = s["agent"]["messages"]
#             for msg in messages:
#                 if isinstance(msg, AIMessage):
#                     print(f"AI: {msg.content}")
#                 elif isinstance(msg, ToolMessage):
#                     print(f"Tool Call: {msg.tool_calls}")
#         print("\n---\n")


# if __name__ == "__main__":
#     run_agent(agent.stream({"messages": [HumanMessage(content="what is 5 plus 5  and 10 minus 5?")]}) )


def run(stream):
    print("\n" + "=" * 50)
    print("🚀 STARTING AGENT SESSION")
    print("=" * 50 + "\n")

    for s in stream:
        # Check if we are in 'updates' mode (dict of node names)
        if isinstance(s, dict):
            for node_name, state in s.items():
                print(f"[{node_name.upper()} NODE]")

                messages = state.get("messages", [])
                for message in messages:
                    if isinstance(message, AIMessage):
                        if message.content:
                            print(f"🤖 Agent Reasoning:\n{message.content.strip()}")

                        if message.tool_calls:
                            for tool_call in message.tool_calls:
                                print(
                                    f"🛠️ Tool Call: {tool_call['name']}({tool_call['args']})"
                                )

                    elif isinstance(message, ToolMessage):
                        print(f"🔍 Observation: {message.content}")

                    elif isinstance(message, HumanMessage):
                        print(f"👤 User: {message.content}")

                print("-" * 30)
        else:
            # If stream_mode="values", s is the full state
            print("Full State Update:", s)

    print("\n" + "=" * 50)
    print("🏁 SESSION COMPLETE")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    inputs = {
        "messages": [
            HumanMessage(
                content="what is 5 plus 5, then multiply the result by 23 and subtract 7?"
            )
        ]
    }

    # We use stream_mode="updates" for clear node-by-node feedback
    run(agent.stream(inputs, stream_mode="updates"))
