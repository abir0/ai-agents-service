from datetime import datetime

from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.runnables import (
    RunnableConfig,
    RunnableLambda,
    RunnableSerializable,
)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from agents.llm import get_model, settings
from agents.tools import get_retriever_tool


class AgentState(MessagesState, total=False):
    """`total=False` is PEP589 specs.

    documentation: https://typing.readthedocs.io/en/latest/spec/typeddict.html#totality
    """


# Add the tools
retriever_tool = get_retriever_tool()
web_search = DuckDuckGoSearchResults(name="web_search")
tools = [retriever_tool, web_search]


# System message
current_date = datetime.now().strftime("%B %d, %Y")
instructions = f"""
    You are a helpful customer support assistant with the ability
    to use tools and retrieve information and assist the user.
    Today's date is {current_date}.

    NOTE: THE USER CAN'T SEE THE TOOL RESPONSE.

    A few things to remember:
    - Use the provided tools to search data for user queries.
    - When searching web, be persistent. Expand your query bounds if the first search returns no results.
    - If a search comes up empty, expand your search before giving up
    - Please include markdown-formatted links to any citations used in your response. Only include one
    or two citations per response unless more are needed. ONLY USE LINKS RETURNED BY THE TOOLS.
    """


def wrap_model(model: BaseChatModel) -> RunnableSerializable[AgentState, AIMessage]:
    model = model.bind_tools(tools)
    preprocessor = RunnableLambda(
        lambda state: [SystemMessage(content=instructions)] + state["messages"],
        name="StateModifier",
    )
    return preprocessor | model


async def acall_model(state: AgentState, config: RunnableConfig) -> AgentState:
    m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    model_runnable = wrap_model(m)
    response = await model_runnable.ainvoke(state, config)

    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


# Agent functions
def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }


def create_tool_node_with_fallback(tools: list) -> dict:
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )


# Define the graph
workflow = StateGraph(AgentState)

# Define the nodes
workflow.add_node("agent", acall_model)
workflow.add_node("tools", create_tool_node_with_fallback(tools))

# Define the edges
workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent",
    tools_condition,
)
workflow.add_edge("tools", "agent")
workflow.add_edge("agent", END)

rag_agent = workflow.compile(checkpointer=MemorySaver())
