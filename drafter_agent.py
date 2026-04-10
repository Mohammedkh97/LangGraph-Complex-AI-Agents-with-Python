from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
    SystemMessage,
)
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from os import getenv
import sys

# Ensure UTF-8 encoding for Windows console
if sys.platform == "win32":
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

load_dotenv()

API_KEY = getenv("GROQ_API_KEY")
# This is the global variable to store document content
document_content = ""


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


@tool
def update(content: str) -> str:
    """Updates the document with the provided content."""
    global document_content
    document_content = content
    return f"Document has been updated successfully! The current content is:\n{document_content}"


@tool
def save(filename: str) -> str:
    """Save the current document to a text file and finish the process.

    Args:
        filename: Name for the text file.
    """

    global document_content

    if not filename.endswith(".txt"):
        filename = f"{filename}.txt"

    try:
        with open(filename, "w") as file:
            file.write(document_content)
        print(f"\n💾 Document has been saved to: {filename}")
        return f"Document has been saved successfully to '{filename}'."

    except Exception as e:
        return f"Error saving document: {str(e)}"


tools = [update, save]

model = ChatGroq(model="llama-3.3-70b-versatile", api_key=API_KEY).bind_tools(tools)


def our_agent(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(
        content=f"""
You are Drafter, a helpful expert writing assistant. You help the user draft, update, and modify documents.

Available tools:
1. 'update': Use this tool to modify or create the document content. Always provide the COMPLETE updated content.
2. 'save': Use this tool to save the draft to a file. Provide a filename.

CRITICAL INSTRUCTIONS:
- If the user wants to update or modify content, use the 'update' tool with the complete updated content.
- If the user wants to save and finish, you need to use the 'save' tool.
- Make sure to always show the current document state after modifications.

Current document content:
{document_content if document_content else "[Empty Document]"}
"""
    )

    messages = state["messages"]
    last_message = messages[-1] if messages else None
    new_messages = []

    # Decide whether to ask the user for input
    # Ask if: 1. No messages yet (start) OR 2. Last message was from AI without tool calls (AI is waiting for user)
    if not messages:
        print(
            "\n🤖 AI: I'm Drafter, ready to help you update a document. What would you like to create?"
        )
        user_input = input("\n👤 USER: ")
        user_message = HumanMessage(content=user_input)
        new_messages.append(user_message)
    elif isinstance(last_message, AIMessage) and not last_message.tool_calls:
        # AI finished its thought, now we need user input
        user_input = input("\n👤 USER: ")
        user_message = HumanMessage(content=user_input)
        new_messages.append(user_message)
    # If last message was a ToolMessage, we DON'T ask for human input. We let the AI process the tool result.

    all_messages = [system_prompt] + list(messages) + new_messages

    response = model.invoke(all_messages)

    if response.content:
        print(f"\n🤖 AI: {response.content}")
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"🔧 USING TOOLS: {[tc['name'] for tc in response.tool_calls]}")

    return {"messages": new_messages + [response]}


def should_continue(state: AgentState) -> str:
    """Determine if we should continue or end the conversation."""

    messages = state["messages"]

    if not messages:
        return "continue"

    # This looks for the most recent tool message....
    for message in reversed(messages):
        # ... and checks if this is a ToolMessage resulting from save
        if (
            isinstance(message, ToolMessage)
            and "saved" in message.content.lower()
            and "document" in message.content.lower()
        ):
            return "end"  # goes to the end edge which leads to the endpoint

    return "continue"


def print_messages(messages):
    """Function I made to print the messages in a more readable format"""
    if not messages:
        return

    for message in messages[-3:]:
        if isinstance(message, ToolMessage):
            print(f"\n🛠️ TOOL RESULT: {message.content}")


graph = StateGraph(AgentState)

graph.add_node("agent", our_agent)
graph.add_node("tools", ToolNode(tools))

graph.set_entry_point("agent")

graph.add_edge("agent", "tools")


graph.add_conditional_edges(
    "tools",
    should_continue,
    {
        "continue": "agent",
        "end": END,
    },
)

app = graph.compile()


def run_document_agent():
    print("\n ===== DRAFTER =====")

    state = {"messages": []}

    for step in app.stream(state, stream_mode="values"):
        if "messages" in step:
            print_messages(step["messages"])

    print("\n ===== DRAFTER FINISHED =====")


if __name__ == "__main__":
    run_document_agent()
