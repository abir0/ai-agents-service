import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Annotated, List, Optional

import requests
from genson import SchemaBuilder
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL

from database import AsyncPostgresManager
from semantic_layer import TABLE_SCHEMA_MAP
from schema import (
    PostgresDBSearchInput,
    PostgresDBSearchOutput,
    DBSQLExecuteOutput,
    VectorDBSearchInput,
    VectorDBSearchOutput,
)
from settings import settings


def json_snippet(
    data: List[dict], max_length: int = 2000, format_type: str = "json"
) -> str:
    """
    Convert a JSON data object into a partial snippet.

    Args:
        data (List[dict]): Array of JSON objects in Python data type
        max_length (int): Maximum length of the output snippet
        format_type (str): 'json' for JSON format

    Returns:
        str: A truncated JSON snippet or text view
    """
    if format_type == "json":
        if len(data) <= 5:
            return json.dumps(data, indent=2)
        json_str = json.dumps(data[:5], indent=2)
        return json_str[:max_length] + ("..." if len(json_str) > max_length else "")


def generate_schema(data: List[dict]) -> dict:
    builder = SchemaBuilder()
    builder.add_object(data)
    return builder.to_schema()


@tool(args_schema=PostgresDBSearchInput)
def postgres_db_search(
    query: str,
    parameters: Optional[List[dict]] = None,
) -> PostgresDBSearchOutput:
    """
    Searches and queries data from PostgreSQL using SQL syntax.
    This tool helps retrieve data from PostgreSQL tables based on user queries.
    It returns a dict containing the filepath of the full data and a textual snippet
    of the data.

    Useful for when you need to:
    - Search for specific items in the database
    - Query data using SQL syntax
    - Filter and retrieve records based on conditions

    The query should be in PostgreSQL SQL syntax. Example queries:
    - "SELECT * FROM table_name"
    - "SELECT data->>'field' FROM table_name WHERE data->>'category' = 'books'"

    Args:
        query (str): SQL query to execute
        parameters (Optional[List[dict]]): Optional query parameters in sqlialchemy format

    Returns:
        PostgresDBSearchOutput: A structured response containing a snippet of query results,
        the filename where the full results are saved, the count of items, and the status.
    """
    try:
        db_url = settings.DATABASE_URL

        if not db_url:
            raise ValueError(
                "PostgreSQL database URL not found in environment variables"
            )

        # Ensure parameters is either None or a valid list of parameter dictionaries
        if parameters is not None and not isinstance(parameters, list):
            parameters = None

        async def execute_search():
            async with AsyncPostgresManager(db_url) as manager:
                results = await manager.query_data(
                    query=query,
                    parameters=parameters or [],  # Pass empty list if parameters is None
                )
                return results

        results = asyncio.run(execute_search())

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        filename = f"{settings.ROOT_PATH}/data/postgres/results_{timestamp}.json"
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)

        snippet = json_snippet(results) if results else ""
        # schema = generate_schema(results)
        return PostgresDBSearchOutput(
            snippet=snippet,
            # db_schema=schema,
            file=filename,
            count=len(results),
            status="success",
        )

    except Exception as e:
        return PostgresDBSearchOutput(
            snippet=None, db_schema=None, file=None, count=0, status=f"error: {str(e)}"
        )


@tool
def python_repl(code: Annotated[str, "Python code or filename to read the code from"]):
    """Use this tool to execute python code. Make sure that you input the code correctly.
    Either input actual code or filename of the code. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user.
    """

    try:
        result = PythonREPL().run(code)
        print("RESULT CODE EXECUTION:", result)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    return f"Executed:\n```python\n{code}\n```\nStdout: {result}"


@tool
def run_sql_query(
    query: str,
    app: str,
    config: RunnableConfig,
) -> DBSQLExecuteOutput:
    """Use this tool to execute a SQL query against the database.
    
    This tool executes raw SQL queries directly on the configured database.
    Useful for data retrieval, analysis, and reporting tasks.
    
    Args:
        query (str): Raw SQL query to execute
        app (str): Application name (default: "arrow")
        config (RunnableConfig): set by runtime environment to pass thread_id

    Returns:
        DBSQLExecuteOutput: A structured response containing query results,
        status, message, and error information.
    """
    try:
        if config:
            thread_id = config["configurable"].get("thread_id")
        else:
            thread_id = None
        print(thread_id)
        print(settings.GENBI_API_KEY)

        payload = {
            "uid": thread_id if thread_id else f"agent_req_{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
            "app": app if app else "arrow",
            "sql": query,
        }

        result = requests.post(
            f"{settings.GENBI_API_URL}/api/query/execute",
            json=payload,
            headers={"Authorization": f"Bearer {settings.GENBI_API_KEY}"},
        ).json()
        print(result)
        
        if result.get("is_error"):
            return DBSQLExecuteOutput(
                query=query,
                status=result.get("status", "error"),
                message=result.get("message", "Unknown error"),
                data=None,
                data_path=None,
                row_count=0,
                is_error=True,
            )
        
        data = result.get("data", [])
        status = result.get("status", "success")
        message = result.get("message", "")

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        filename = f"{settings.ROOT_PATH}/data/sql/data_{timestamp}.json"
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

        snippet = json_snippet(data) if data else ""
        
        return DBSQLExecuteOutput(
            query=query,
            status=status,
            message=message if message else None,
            data=snippet,
            data_path=filename,
            row_count=len(data),
            is_error=False,
        )
    except Exception as e:
        return DBSQLExecuteOutput(
            query=query,
            status="error",
            message=f"Failed to execute SQL query. Error: {repr(e)}",
            data=None,
            data_path=None,
            row_count=0,
            is_error=True,
        )


@tool()
def get_table_schema(
    table_name: str,
) -> str:
    """Use this tool to get the schema of a PostgreSQL table.
    
    This tool retrieves the schema of a specified table in the PostgreSQL database.
    Useful for understanding the structure of the data and available columns.
    
    Args:
        table_name (str): Name of the table to retrieve schema for
        app (str): Application name (default: "arrow")
        
    Returns:
        str: The schema of the specified table in markdown format.
    """
    try:
        return TABLE_SCHEMA_MAP[table_name]
    except Exception as e:
        raise ValueError(f"Failed to retrieve schema for table '{table_name}': {repr(e)}")


@tool(args_schema=VectorDBSearchInput)
def search_similar_queries(
    query: str,
    collection_name: str = "biarrow_sample_questions",
    top_k: int = 3,
) -> VectorDBSearchOutput:
    """Use this tool to search for similar queries in the vector database.
    
    This tool finds semantically similar items based on the input query using
    vector similarity search. Useful for finding related questions, similar 
    documents, or retrieving relevant context.
    
    Args:
        query (str): The search query text
        collection_name (str): Name of the vector database collection (default: "biarrow_sample_questions")
        top_k (int): Number of top similar results to return (default: 3)
        
    Returns:
        VectorDBSearchOutput: A structured response containing the query,
        collection name, results, result count, and status.
    """
    try:
        payload = {
            "query": query,
            "collection_name": collection_name,
            "top_k": top_k,
        }
        
        result = requests.post(
            f"{settings.GENBI_API_URL}/api/vectordb/search",
            json=payload,
            headers={"Authorization": f"Bearer {settings.GENBI_API_KEY}"},
        ).json()
        
        results = result.get("results", [])
        
        return VectorDBSearchOutput(
            query=query,
            collection_name=collection_name,
            results=results,
            result_count=len(results),
            status="success",
        )
    except Exception as e:
        return VectorDBSearchOutput(
            query=query,
            collection_name=collection_name,
            results=[],
            result_count=0,
            status=f"error: {repr(e)}",
        )
