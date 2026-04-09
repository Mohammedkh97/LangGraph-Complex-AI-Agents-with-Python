from typing import TypedDict, List, Union, Annotated, Sequence
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    BaseMessage,
    SystemMessage,
    ToolMessage,
)

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool, BaseTool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv
import os


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


tools = [add, subtract, multiply]


llm = ChatOpenAI(
    # model="llama-3-8b-8192",
    model="openai/gpt-oss-120b",
    # model="deepseek-r1-distill-llama-70b",
    # model="llama3-8b-8192",
    api_key=API_KEY,  # type: ignore
    base_url="https://api.groq.com/openai/v1",
).bind_tools(tools)


def llm_model(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(
        content="You are a helpful AI assistant. You have access to the following tools, so please answer my query to the best of your ability."
    )

    # The `add_messages` in the state automatically appends new messages
    # We can add the system prompt if it's the first message
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
    for s in stream:
        print(s)  # debugging: see what node gave output
        # get the node name (first and only key in dict)
        node_name = list(s.keys())[0]
        state = s[node_name]
        message = state["messages"][-1]

        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()
        print("\n---\n")


# inputs = {"messages": [HumanMessage(content="what is 5 plus 5  and 23 minus 7?")]}
# run(agent.stream(inputs, stream_node="values"))
if __name__ == "__main__":
    run(
        agent.stream(
            {"messages": [HumanMessage(content="what is 5 plus 5  and 23 minus 7?")]},
            stream_node="values",
        )
    )
