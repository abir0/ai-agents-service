from typing import Any, Literal, NotRequired, Optional, Union

from pydantic import BaseModel, Field, SerializeAsAny
from typing_extensions import TypedDict

from schema import AllModelEnum, OpenAIModelName


class AgentInfo(BaseModel):
    """Info about an available agent."""

    key: str = Field(
        description="Agent key.",
        examples=["research-assistant"],
    )
    description: str = Field(
        description="Description of the agent.",
        examples=["A research assistant for generating research papers."],
    )


class ServiceMetadata(BaseModel):
    """Metadata about the service including available agents and models."""

    agents: list[AgentInfo] = Field(
        description="List of available agents.",
    )
    models: list[AllModelEnum] = Field(
        description="List of available LLMs.",
    )
    default_agent: str = Field(
        description="Default agent used when none is specified.",
        examples=["research-assistant"],
    )
    default_model: AllModelEnum = Field(
        description="Default model used when none is specified.",
    )


class UserInput(BaseModel):
    """Basic user input for the agent."""

    message: str = Field(
        description="User input to the agent.",
        examples=["What is the weather in Tokyo?"],
    )
    model: SerializeAsAny[AllModelEnum] | None = Field(
        title="Model",
        description="LLM Model to use for the agent.",
        default=OpenAIModelName.GPT_4O,
        examples=[OpenAIModelName.GPT_4O_MINI, OpenAIModelName.GPT_4O],
    )
    thread_id: str | None = Field(
        description="Thread ID to persist and continue a multi-turn conversation.",
        default=None,
        examples=["847c6285-8fc9-4560-a83f-4e6285809254"],
    )
    agent_config: dict[str, Any] = Field(
        description="Additional configuration to pass through to the agent",
        default={},
        examples=[{"spicy_level": 0.8}],
    )


class StreamInput(UserInput):
    """User input for streaming the agent's response."""

    stream_tokens: bool = Field(
        description="Whether to stream LLM tokens to the client.",
        default=True,
    )


class ToolCall(TypedDict):
    """Represents a request to call a tool."""

    name: str
    """The name of the tool to be called."""
    args: dict[str, Any]
    """The arguments to the tool call."""
    id: str | None
    """An identifier associated with the tool call."""
    type: NotRequired[Literal["tool_call"]]


class ChatMessage(BaseModel):
    """Message in a chat."""

    type: Literal["human", "ai", "tool", "custom"] = Field(
        description="Role of the message.",
        examples=["human", "ai", "tool", "custom"],
    )
    content: str = Field(
        description="Content of the message.",
        examples=["Hello, world!"],
    )
    tool_calls: list[ToolCall] = Field(
        description="Tool calls in the message.",
        default=[],
    )
    tool_call_id: str | None = Field(
        description="Tool call that this message is responding to.",
        default=None,
        examples=["call_Jja7J89XsjrOLA5r!MEOW!SL"],
    )
    run_id: str | None = Field(
        description="Run ID of the message.",
        default=None,
        examples=["847c6285-8fc9-4560-a83f-4e6285809254"],
    )
    response_metadata: dict[str, Any] = Field(
        description="Response metadata. For example: response headers, logprobs, token counts.",
        default={},
    )
    custom_data: dict[str, Any] = Field(
        description="Custom message data.",
        default={},
    )
    plotly_data: dict[str, Any] | None = Field(
        description="Plotly figure data for chart visualization.",
        default=None,
    )

    def pretty_repr(self) -> str:
        """Get a pretty representation of the message."""
        base_title = self.type.title() + " Message"
        padded = " " + base_title + " "
        sep_len = (80 - len(padded)) // 2
        sep = "=" * sep_len
        second_sep = sep + "=" if len(padded) % 2 else sep
        title = f"{sep}{padded}{second_sep}"
        return f"{title}\n\n{self.content}"

    def pretty_print(self) -> None:
        print(self.pretty_repr())  # noqa: T201


class Feedback(BaseModel):
    """Feedback for a run, to record to LangSmith."""

    run_id: str = Field(
        description="Run ID to record feedback for.",
        examples=["847c6285-8fc9-4560-a83f-4e6285809254"],
    )
    key: str = Field(
        description="Feedback key.",
        examples=["human-feedback-stars"],
    )
    score: float = Field(
        description="Feedback score.",
        examples=[0.8],
    )
    kwargs: dict[str, Any] = Field(
        description="Additional feedback kwargs, passed to LangSmith.",
        default={},
        examples=[{"comment": "In-line human feedback"}],
    )


class FeedbackResponse(BaseModel):
    status: Literal["success"] = "success"


class ChatHistoryInput(BaseModel):
    """Input for retrieving chat history."""

    thread_id: str = Field(
        description="Thread ID to persist and continue a multi-turn conversation.",
        examples=["847c6285-8fc9-4560-a83f-4e6285809254"],
    )


class ChatHistory(BaseModel):
    messages: list[ChatMessage]


class PostgresDBSearchInput(BaseModel):
    """Schema for PostgreSQL search input parameters."""

    query: str = Field(
        ...,
        description="SQL query string in PostgreSQL syntax. Examples: "
        "'SELECT * FROM table_name' for all records, or "
        "'SELECT * FROM table_name WHERE category = :category' "
        "for parameterized queries",
    )
    # table_name: str = Field(
    #     ...,
    #     description="Name of the PostgreSQL table to query",
    # )
    parameters: Optional[dict] = Field(
        None,
        description="Optional parameters for parameterized queries as a dictionary. "
        "Use in queries like: 'WHERE category = :category'",
        examples=[{"category": "books", "author": "Jane Doe"}],
    )


class PostgresDBSearchOutput(BaseModel):
    """Schema for PostgreSQL search results."""

    snippet: Optional[Any] = Field(..., description="A partial snippet of the data.")
    # db_schema: Optional[dict] = Field(
    #     ..., description="Schema of the output in standard format."
    # )
    count: int = Field(..., description="Number of documents found")
    status: str = Field(..., description="Status of the search operation")
    file: Optional[str] = Field(..., description="Path of the file containing output.")


class DBSQLExecuteInput(BaseModel):
    """Schema for SQL execution input parameters."""

    query: str = Field(
        ...,
        description="Raw SQL query to execute against the database",
        examples=["SELECT * FROM users WHERE status = 'active'"],
    )
    app: str = Field(
        default="arrow",
        description="Application name for database context",
        examples=["arrow", "biarrow"],
    )


class DBSQLExecuteOutput(BaseModel):
    """Schema for SQL execution results."""

    query: str = Field(..., description="The SQL query that was executed")
    status: str = Field(..., description="Status of the query execution")
    message: Optional[str] = Field(
        None, description="Additional message or error details"
    )
    data: Optional[Union[dict, str]] = Field(None, description="Query result data")
    data_path: Optional[str] = Field(
        None,
        description="Path to the file where the result data is stored",
    )
    row_count: int = Field(..., description="Number of rows returned")
    is_error: bool = Field(
        ..., description="Whether an error occurred during execution"
    )


class VectorDBSearchInput(BaseModel):
    """Schema for vector database search input parameters."""

    query: str = Field(
        ...,
        description="Search query text to find semantically similar items",
        examples=["What is the average sales revenue?"],
    )
    collection_name: str = Field(
        default="biarrow_sample_questions",
        description="Name of the vector database collection to search",
        examples=["biarrow_sample_questions", "documents", "faqs"],
    )
    top_k: int = Field(
        default=3,
        ge=1,
        le=100,
        description="Number of top similar results to return",
        examples=[3, 5, 10],
    )


class VectorDBSearchOutput(BaseModel):
    """Schema for vector database search results."""

    query: str = Field(..., description="The search query that was executed")
    collection_name: str = Field(..., description="The collection that was searched")
    results: list[dict] = Field(..., description="List of similar items found")
    result_count: int = Field(..., description="Number of results returned")
    status: str = Field(..., description="Status of the search operation")


class WeatherTrendOutput(BaseModel):
    """Schema for weather trend analysis results."""

    location_name: str = Field(..., description="Name of the location")
    start_date: str = Field(..., description="Start date of the trend analysis")
    end_date: str = Field(..., description="End date of the trend analysis")
    summary: str = Field(
        ..., description="User-friendly text summary of weather trends"
    )
    data_path: Optional[str] = Field(
        None,
        description="Path to the file where the detailed trend data is stored",
    )
    days_analyzed: int = Field(
        ..., description="Number of days included in the analysis"
    )
    status: str = Field(..., description="Status of the operation")
    is_error: bool = Field(default=False, description="Whether an error occurred")


class WeatherForecastOutput(BaseModel):
    """Schema for weather forecast results."""

    location_name: str = Field(..., description="Name of the location")
    forecast_days: int = Field(..., description="Number of days in the forecast")
    summary: str = Field(
        ..., description="User-friendly text summary of weather forecast"
    )
    data_path: Optional[str] = Field(
        None,
        description="Path to the file where the detailed forecast data is stored",
    )
    status: str = Field(..., description="Status of the operation")
    is_error: bool = Field(default=False, description="Whether an error occurred")


class PlotlyChartOutput(BaseModel):
    """Schema for Plotly chart generation results."""

    chart_type: str = Field(
        ..., description="Type of chart (e.g., 'line', 'bar', 'scatter', 'pie')"
    )
    title: str = Field(..., description="Title of the chart")
    description: str = Field(..., description="Description of what the chart shows")
    plotly_json: dict[str, Any] = Field(
        ..., description="Plotly figure as JSON-serializable dictionary"
    )
    data_path: Optional[str] = Field(
        None,
        description="Path to the file where the source data is stored",
    )
    status: str = Field(..., description="Status of the operation")
    is_error: bool = Field(default=False, description="Whether an error occurred")
