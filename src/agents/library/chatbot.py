from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import (
    RunnableConfig,
    RunnableLambda,
    RunnableSerializable,
)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph

from agents.library.llama_guard import LlamaGuard, LlamaGuardOutput, SafetyAssessment
from agents.llm import get_model, settings


class AgentState(MessagesState, total=False):
    """`total=False` is PEP589 specs.

    documentation: https://typing.readthedocs.io/en/latest/spec/typeddict.html#totality
    """

    safety: LlamaGuardOutput


def wrap_model(model: BaseChatModel) -> RunnableSerializable[AgentState, AIMessage]:
    preprocessor = RunnableLambda(
        lambda state: state["messages"],
        name="StateModifier",
    )
    return preprocessor | model


def format_safety_message(safety: LlamaGuardOutput) -> AIMessage:
    content = f"This conversation was flagged for unsafe content: {', '.join(safety.unsafe_categories)}"
    return AIMessage(content=content)


async def acall_model(state: AgentState, config: RunnableConfig) -> AgentState:
    m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    model_runnable = wrap_model(m)
    response = await model_runnable.ainvoke(state, config)

    # Run llama guard check here to avoid returning the message if it's unsafe
    llama_guard = LlamaGuard()
    safety_output = await llama_guard.ainvoke("Agent", state["messages"] + [response])
    if safety_output.safety_assessment == SafetyAssessment.UNSAFE:
        return {
            "messages": [format_safety_message(safety_output)],
            "safety": safety_output,
        }

    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


# Agent functions
async def llama_guard_input(state: AgentState, config: RunnableConfig) -> AgentState:
    llama_guard = LlamaGuard()
    safety_output = await llama_guard.ainvoke("User", state["messages"])
    return {"safety": safety_output}


async def block_unsafe_content(state: AgentState, config: RunnableConfig) -> AgentState:
    safety: LlamaGuardOutput = state["safety"]
    return {"messages": [format_safety_message(safety)]}


def check_safety(state: AgentState) -> Literal["unsafe", "safe"]:
    safety: LlamaGuardOutput = state["safety"]
    match safety.safety_assessment:
        case SafetyAssessment.UNSAFE:
            return "unsafe"
        case _:
            return "safe"


# Define the graph
workflow = StateGraph(AgentState)
workflow.add_node("model", acall_model)
workflow.add_node("guard_input", llama_guard_input)
workflow.add_node("block_unsafe_content", block_unsafe_content)

# Define edges
workflow.set_entry_point("guard_input")
workflow.add_conditional_edges(
    "guard_input",
    check_safety,
    {"unsafe": "block_unsafe_content", "safe": "model"},
)
workflow.add_edge("block_unsafe_content", END)
workflow.add_edge("model", END)

chatbot = workflow.compile(
    checkpointer=MemorySaver(),
)
