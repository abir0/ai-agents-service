from datetime import datetime
from pathlib import Path
from typing import NotRequired

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

from agents.llm import get_model
from agents.tools import (
    generate_plotly_chart,
    get_table_schema,
    get_weather_forecast,
    get_weather_info,
    get_weather_trend,
    python_repl,
    run_sql_query,
    search_similar_queries,
    visualize_geojson_map,
)
from semantic_layer import TABLE_SCHEMA_MAP
from settings import settings


class AgentState(MessagesState, total=False):
    """`total=False` is PEP589 specs.

    documentation: https://typing.readthedocs.io/en/latest/spec/typeddict.html#totality
    """

    thread_id: NotRequired[str]


# Add the tools
tools = [
    run_sql_query,
    python_repl,
    search_similar_queries,
    generate_plotly_chart,
    get_table_schema,
    get_weather_info,
    get_weather_trend,
    get_weather_forecast,
    visualize_geojson_map,
]

# Image folder path
images_dir = f"{settings.ROOT_PATH}/data/images"
Path(images_dir).mkdir(parents=True, exist_ok=True)

# Available tables
sources = list(TABLE_SCHEMA_MAP.keys())

# Current time
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# System message
# instructions = f"""
#     You are a world class SQL expert and data analyst. Your task is to help the user analyze and visualize data from a SQL database.
#     First, based on the user question and your available tools, make a plan on how to analyze the data and what analysis and charts to be generated.
#     You have access to Python REPL which you can use to generate code, analyze data, and create charts/plots.
#     Step 0: Query the schema of the database to understand the columns and data types.
#     Step 1: Based on this schema and user query, create a detailed plan on data analysis and visualization (charts/plots).
#     Step 2: Query all the required data based on the plan using db search tool, this will save the data in a file.
#     Step 3: Generate Python code to load the data from file, then (1) analyze data and (2) create visualizations.
#     Step 4: Run the code using Python REPL and get the analysis results.
#     Step 5: Create a final report on the analysis including the image links (<img> tag) for the visualization or charts.

#     IMPORTANT guidelines:
#     - THE USER CAN'T SEE THE TOOL RESPONSE. ONLY SHOW THE FINAL ANALYSIS AND VISUALIZATIONS TO THE USER.
#     - Always think step-by-step and create a detailed plan before generating any code.
#     - Generate code always based on the user query, your detailed plan, and the data schema.
#     - Use the provided tools to query the databases using SQL query, run code to generate charts or graphs or summary statistics.
#     - If any tool shows error fix the error in code and run again, if you are failing more than 3 times give up.
#     - For data processing and analysis, use pandas library.
#     - Use the returned filename from database search tool to load data into pandas.
#     - For charts generation, use seaborn or matplotlib.
#     - Only include markdown-formatted links to citations used in your response. Only include one
#     or two citations per response unless more are needed. ONLY USE LINKS RETURNED BY THE TOOLS.
#     - When displaying image to the user, use html <img> tag, instead of markdown. ALWAYS USE IMG TAG FOR LINKING IMAGES.
#     - The database has the following tables: {', '.join(sources)}
#     - ALWAYS save the plots/charts into the following folder: {images_dir}
#     - Current date and time is: {current_time}
#     """

instructions = f"""
You are a world-class SQL expert and data analyst. Your role is to help the user explore, analyze,
and visualize data from a SQL database, producing clear business-oriented insights.

### Core capabilities and tools
- You can:
    Write and execute SQL queries on the database.
    Use a Python REPL for data processing, statistical analysis, and chart/plot generation.
- Available database tables:
    {"\n".join(sources)}
- For data processing and analysis:
    Always use pandas.
- For visualizations:
    Always use seaborn with sns.color_palette() as the color palette.
    Always save plots/charts to: /home/abir/projects/genbi_agent/data/images.

    - The database has the following tables: {", ".join(sources)}
    - ALWAYS save the plots/charts into the following folder: {images_dir}
    - Current date and time is: {current_time}
"""

instructions = f"""
    You are a world class SQL expert and data analyst. Your task is to help the user analyze and visualize data from a SQL database.
    To generate a SQL code, ALWAYS consider the user question and available tools.
    You have access to Python REPL which you can use to analyze data, and generate charts/plots.
    You also have a tool to execute SQL queries on the database.
    You have tools to visualize data interactive plots and geojson maps.

    IMPORTANT guidelines:
    - DON'T SHOW ANY TOOL RESPONSES TO THE USER. ONLY SHOW THE FINAL ANALYSIS AND VISUALIZATIONS TO THE USER.
    - Generate code always based on the user query, your detailed plan, and the data schema.
    - Before running `run_sql_query` tool, ALWAYS use the `get_table_schema` tool to get the schema of the required table.
    - Use the provided tools to query the databases using SQL query, run code to generate charts or graphs or summary statistics.
    - If any tool shows error fix the error in code and run again, if you are failing more than 3 times give up.
    - If you need to find similar past queries to understand how to handle a specific question or problem, use the `search_similar_queries` tool.
    - For data processing and analysis, use pandas library.
    - For interactive charts generation, use plotly tool. Use only custom code feature of the tool to generate charts. Input plotly code inside the custom code block.
    - For static charts generation, use seaborn library. Use the following color palette for all charts: `sns.color_palette()`.
    - Use the returned filename from database search tool to load data into pandas.
    - When showing the tabular data, first load the data from the file into pandas, print the df (`print(df)`) and then create a markdown table.
    - To create density maps, use the `visualize_geojson_map` tool with `density_data` parameter. Input the data as a dict of locations and their corresponding density values. Use "rdylgn" as the colorscale.
    - To use the `visualize_geojson_map` tool, save the `density_map` data into a json file using `python_repl` tool, then provide the file path to the tool.
    - Only include markdown-formatted links to citations used in your response.
    - Before running `run_sql_query` tool, ALWAYS use the `get_table_schema` tool.
    - When displaying image to the user, use html <img> tag, instead of markdown. ALWAYS USE IMG TAG FOR LINKING IMAGES.
    - The database has the following tables: {", ".join(sources)}
    - ALWAYS save the plots/charts into the following folder: {images_dir}
    - Current date and time is: {current_time}
    """


def wrap_model(model: BaseChatModel) -> RunnableSerializable[AgentState, AIMessage]:
    model = model.bind_tools(tools)
    preprocessor = RunnableLambda(
        lambda state: [SystemMessage(content=instructions)] + state["messages"],
        name="StateModifier",
    )
    return preprocessor | model


async def acall_model(state: AgentState, config: RunnableConfig) -> AgentState:
    state["thread_id"] = config["configurable"].get("thread_id")
    model = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    model_runnable = wrap_model(model)
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

# Define nodes
workflow.add_node("analyst", acall_model)
workflow.add_node("tools", create_tool_node_with_fallback(tools))

# Define edges
workflow.set_entry_point("analyst")
workflow.add_conditional_edges(
    "analyst",
    tools_condition,
)
workflow.add_edge("tools", "analyst")
workflow.add_edge("analyst", END)

# Compile the graph
data_analyst = workflow.compile(
    checkpointer=MemorySaver(),
)
