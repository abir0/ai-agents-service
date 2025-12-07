import asyncio
import csv
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
from schema import (
    DBSQLExecuteOutput,
    PlotlyChartOutput,
    PostgresDBSearchInput,
    PostgresDBSearchOutput,
    VectorDBSearchInput,
    VectorDBSearchOutput,
    WeatherForecastOutput,
    WeatherTrendOutput,
)
from semantic_layer import TABLE_SCHEMA_MAP
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


def convert_numpy_to_list(obj):
    """
    Recursively convert numpy arrays to lists in nested dictionaries and lists.

    Args:
        obj: Object that may contain numpy arrays

    Returns:
        Object with all numpy arrays converted to lists
    """
    import numpy as np

    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_list(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_list(item) for item in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    else:
        return obj


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
                    parameters=parameters
                    or [],  # Pass empty list if parameters is None
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
            "uid": thread_id
            if thread_id
            else f"agent_req_{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
            "app": app if app else "arrow",
            "sql": query,
        }

        result = requests.post(
            f"{settings.GENBI_API_URL}/api/query/execute",
            json=payload,
            headers={"Authorization": f"Bearer {settings.GENBI_API_KEY}"},
        )
        print(result.text)

        result = result.json()
        # print(result)
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
        raise ValueError(
            f"Failed to retrieve schema for table '{table_name}': {repr(e)}"
        )


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


# Load lat-long coordinates
LATLNG_MAP = {}
with open(f"{settings.ROOT_PATH}/data/latlng.csv", mode="r") as file:
    reader = csv.DictReader(file)
    for row in reader:
        LATLNG_MAP[row["name"]] = (float(row["lat"]), float(row["lng"]))


@tool
def get_weather_info(location_name: str, location_type: Optional[str] = None) -> str:
    """Use this tool to get the current weather information for a specified location.

    This tool retrieves real-time weather data including temperature, humidity,
    and weather conditions from OpenWeatherMap API for the given location.

    Args:
        location_name (str): Name of the location to get weather information for (e.g., "DHAKA CIRCLE", "CHATTOGRAM CENTRAL", etc.)
        location_type (Optional[str]): Type of the location ("circle", "region", "cluster", "territory").
    Returns:
        str: A formatted string containing the current weather information.
    """
    try:
        # Get latitude and longitude from LATLNG_MAP
        location_name_upper = location_name.upper()
        if location_name_upper in LATLNG_MAP:
            lat, lng = LATLNG_MAP[location_name_upper]
        else:
            return f"Location '{location_name}' not found in latitude/longitude map."

        # Check if OpenWeatherMap API key is configured
        if not settings.OPENWEATHERMAP_API_KEY:
            return "OpenWeatherMap API key is not configured. Please set OPENWEATHERMAP_API_KEY in environment variables."

        # Call OpenWeatherMap API
        api_url = "http://api.openweathermap.org/data/2.5/weather"
        params = {
            "lat": lat,
            "lon": lng,
            "appid": settings.OPENWEATHERMAP_API_KEY.get_secret_value(),
            "units": "metric",  # For temperature in Celsius
            "lang": "en",
            "mode": "json",
        }

        response = requests.get(api_url, params=params, timeout=10)
        response.raise_for_status()

        weather_data = response.json()

        # Extract weather information
        main = weather_data.get("main", {})
        weather_list = weather_data.get("weather", [])
        weather_dict = weather_list[0] if weather_list else {}
        wind = weather_data.get("wind", {})
        clouds = weather_data.get("clouds", {})

        temperature = main.get("temp")
        feels_like = main.get("feels_like")
        temp_min = main.get("temp_min")
        temp_max = main.get("temp_max")
        pressure = main.get("pressure")
        humidity = main.get("humidity")

        condition = weather_dict.get("main")
        description = weather_dict.get("description")

        wind_speed = wind.get("speed")
        cloudiness = clouds.get("all")

        # Format the output
        weather_info = f"Current weather for {location_name}"
        if location_type:
            weather_info += f" ({location_type})"
        weather_info += ":\n"

        if temperature is not None:
            weather_info += f"- Temperature: {temperature}°C"
            if feels_like is not None:
                weather_info += f" (feels like {feels_like}°C)"
            weather_info += "\n"

        if temp_min is not None and temp_max is not None:
            weather_info += f"- Temperature Range: {temp_min}°C - {temp_max}°C\n"

        if humidity is not None:
            weather_info += f"- Humidity: {humidity}%\n"

        if condition:
            weather_info += f"- Condition: {condition}"
            if description:
                weather_info += f" ({description})"
            weather_info += "\n"

        if pressure is not None:
            weather_info += f"- Pressure: {pressure} hPa\n"

        if wind_speed is not None:
            weather_info += f"- Wind Speed: {wind_speed} m/s\n"

        if cloudiness is not None:
            weather_info += f"- Cloudiness: {cloudiness}%\n"

        return weather_info.strip()

    except requests.exceptions.RequestException as e:
        return f"Failed to retrieve weather information from OpenWeatherMap API. Error: {repr(e)}"
    except Exception as e:
        return f"Failed to retrieve weather information. Error: {repr(e)}"


@tool
def get_weather_trend(
    location_name: str,
    start_date: str,
    end_date: str,
    location_type: Optional[str] = None,
) -> WeatherTrendOutput:
    """Use this tool to get weather trend data for a location over a specific date range.

    This tool retrieves historical weather data including temperature trends, humidity patterns,
    and weather conditions from OpenWeatherMap API for the given location and date range.
    Note: Historical weather data is available for dates in the past. For future forecasts,
    use the weather forecast API instead.

    Args:
        location_name (str): Name of the location (e.g., "DHAKA CIRCLE", "CHATTOGRAM CENTRAL", etc.)
        start_date (str): Start date in format 'YYYY-MM-DD' (e.g., "2025-12-01")
        end_date (str): End date in format 'YYYY-MM-DD' (e.g., "2025-12-07")
        location_type (Optional[str]): Type of the location ("circle", "region", "cluster", "territory")

    Returns:
        WeatherTrendOutput: A structured response containing weather trend summary,
        file path to detailed data, and analysis statistics.
    """
    try:
        # Get latitude and longitude from LATLNG_MAP
        location_name_upper = location_name.upper()
        if location_name_upper in LATLNG_MAP:
            lat, lng = LATLNG_MAP[location_name_upper]
        else:
            return f"Location '{location_name}' not found in latitude/longitude map."

        # Check if OpenWeatherMap API key is configured
        if not settings.OPENWEATHERMAP_API_KEY:
            return "OpenWeatherMap API key is not configured. Please set OPENWEATHERMAP_API_KEY in environment variables."

        # Parse dates
        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")

            if start_dt > end_dt:
                return WeatherTrendOutput(
                    location_name=location_name,
                    start_date=start_date,
                    end_date=end_date,
                    summary="Start date must be before or equal to end date.",
                    data_path=None,
                    days_analyzed=0,
                    status="error",
                    is_error=True,
                )

            # Convert to Unix timestamps
            start_timestamp = int(start_dt.timestamp())
            end_timestamp = int(end_dt.timestamp())

        except ValueError as e:
            return WeatherTrendOutput(
                location_name=location_name,
                start_date=start_date,
                end_date=end_date,
                summary=f"Invalid date format. Please use 'YYYY-MM-DD' format. Error: {repr(e)}",
                data_path=None,
                days_analyzed=0,
                status="error",
                is_error=True,
            )

        # Call OpenWeatherMap History API (requires subscription) or use Daily Aggregation API
        # For demonstration, we'll use the One Call API 3.0 timemachine endpoint
        # Note: This requires a paid subscription for historical data

        weather_records = []
        current_timestamp = start_timestamp

        # Collect daily data points
        while current_timestamp <= end_timestamp:
            api_url = "http://api.openweathermap.org/data/3.0/onecall/timemachine"
            params = {
                "lat": lat,
                "lon": lng,
                "dt": current_timestamp,
                "appid": settings.OPENWEATHERMAP_API_KEY.get_secret_value(),
                "units": "metric",
            }

            try:
                response = requests.get(api_url, params=params, timeout=10)
                response.raise_for_status()
                weather_data = response.json()

                # Extract data from the response
                if "data" in weather_data and len(weather_data["data"]) > 0:
                    day_data = weather_data["data"][0]
                    weather_records.append(
                        {
                            "date": datetime.fromtimestamp(current_timestamp).strftime(
                                "%Y-%m-%d"
                            ),
                            "temp": day_data.get("temp"),
                            "humidity": day_data.get("humidity"),
                            "pressure": day_data.get("pressure"),
                            "wind_speed": day_data.get("wind_speed"),
                            "clouds": day_data.get("clouds"),
                            "weather": day_data.get("weather", [{}])[0].get(
                                "main", "N/A"
                            ),
                        }
                    )
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 401:
                    return WeatherTrendOutput(
                        location_name=location_name,
                        start_date=start_date,
                        end_date=end_date,
                        summary="OpenWeatherMap API authentication failed. Historical weather data requires a paid subscription (One Call API 3.0).",
                        data_path=None,
                        days_analyzed=0,
                        status="error",
                        is_error=True,
                    )
                elif e.response.status_code == 403:
                    return WeatherTrendOutput(
                        location_name=location_name,
                        start_date=start_date,
                        end_date=end_date,
                        summary="Access forbidden. Historical weather data requires a paid subscription to OpenWeatherMap One Call API 3.0.",
                        data_path=None,
                        days_analyzed=0,
                        status="error",
                        is_error=True,
                    )
            except Exception as e:
                print(
                    f"Error fetching data for timestamp {current_timestamp}: {repr(e)}"
                )

            # Move to next day (86400 seconds = 24 hours)
            current_timestamp += 86400

        if not weather_records:
            return WeatherTrendOutput(
                location_name=location_name,
                start_date=start_date,
                end_date=end_date,
                summary=(
                    "Unable to retrieve historical weather data. "
                    "Note: Historical weather data requires OpenWeatherMap One Call API 3.0 subscription. "
                    "Alternatively, you can query historical data from your database if available."
                ),
                data_path=None,
                days_analyzed=0,
                status="error",
                is_error=True,
            )

        # Calculate statistics
        temps = [r["temp"] for r in weather_records if r["temp"] is not None]
        humidities = [
            r["humidity"] for r in weather_records if r["humidity"] is not None
        ]

        if not temps:
            return WeatherTrendOutput(
                location_name=location_name,
                start_date=start_date,
                end_date=end_date,
                summary="No valid weather data found for the specified date range.",
                data_path=None,
                days_analyzed=0,
                status="error",
                is_error=True,
            )

        avg_temp = sum(temps) / len(temps)
        min_temp = min(temps)
        max_temp = max(temps)
        avg_humidity = sum(humidities) / len(humidities) if humidities else None

        # Count weather conditions
        conditions = {}
        for record in weather_records:
            condition = record["weather"]
            conditions[condition] = conditions.get(condition, 0) + 1

        # Format the output
        trend_info = f"Weather trend for {location_name}"
        if location_type:
            trend_info += f" ({location_type})"
        trend_info += f"\nPeriod: {start_date} to {end_date}\n"
        trend_info += f"Total days analyzed: {len(weather_records)}\n\n"

        trend_info += "Temperature Statistics:\n"
        trend_info += f"- Average: {avg_temp:.1f}°C\n"
        trend_info += f"- Minimum: {min_temp:.1f}°C\n"
        trend_info += f"- Maximum: {max_temp:.1f}°C\n"
        trend_info += f"- Range: {max_temp - min_temp:.1f}°C\n\n"

        if avg_humidity is not None:
            trend_info += f"Average Humidity: {avg_humidity:.1f}%\n\n"

        trend_info += "Weather Conditions Distribution:\n"
        for condition, count in sorted(
            conditions.items(), key=lambda x: x[1], reverse=True
        ):
            percentage = (count / len(weather_records)) * 100
            trend_info += f"- {condition}: {count} days ({percentage:.1f}%)\n"

        trend_info += "\nDaily Breakdown:\n"
        for record in weather_records[-7:]:  # Show last 7 days or all if fewer
            trend_info += (
                f"- {record['date']}: {record['temp']:.1f}°C, "
                f"{record['humidity']}% humidity, {record['weather']}\n"
            )

        if len(weather_records) > 7:
            trend_info += f"... and {len(weather_records) - 7} more days\n"

        # Save detailed data to JSON file
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        filename = f"{settings.ROOT_PATH}/data/weather/trend_{timestamp}.json"
        Path(filename).parent.mkdir(parents=True, exist_ok=True)

        detailed_data = {
            "location_name": location_name,
            "location_type": location_type,
            "start_date": start_date,
            "end_date": end_date,
            "days_analyzed": len(weather_records),
            "statistics": {
                "temperature": {
                    "average": avg_temp,
                    "minimum": min_temp,
                    "maximum": max_temp,
                    "range": max_temp - min_temp,
                },
                "humidity": {"average": avg_humidity} if avg_humidity else None,
                "conditions": conditions,
            },
            "daily_records": weather_records,
        }

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(detailed_data, f, indent=4)

        # Truncate summary to 500 characters to avoid context overload
        summary_text = trend_info.strip()
        if len(summary_text) > 500:
            summary_text = summary_text[:500] + "..."

        return WeatherTrendOutput(
            location_name=location_name,
            start_date=start_date,
            end_date=end_date,
            summary=trend_info.strip(),
            data_path=filename,
            days_analyzed=len(weather_records),
            status="success",
            is_error=False,
        )

    except requests.exceptions.RequestException as e:
        return WeatherTrendOutput(
            location_name=location_name,
            start_date=start_date,
            end_date=end_date,
            summary=f"Failed to retrieve weather trend data from OpenWeatherMap API. Error: {repr(e)}",
            data_path=None,
            days_analyzed=0,
            status="error",
            is_error=True,
        )
    except Exception as e:
        return WeatherTrendOutput(
            location_name=location_name,
            start_date=start_date,
            end_date=end_date,
            summary=f"Failed to retrieve weather trend data. Error: {repr(e)}",
            data_path=None,
            days_analyzed=0,
            status="error",
            is_error=True,
        )


@tool
def get_weather_forecast(
    location_name: str, days: int = 5, location_type: Optional[str] = None
) -> WeatherForecastOutput:
    """Use this tool to get weather forecast for a location for upcoming days.

    This tool retrieves weather forecast data including predicted temperature, humidity,
    precipitation probability, and weather conditions from OpenWeatherMap API.
    The forecast includes hourly predictions for the next 48 hours and daily predictions
    for up to 7-8 days (depending on API subscription).

    Args:
        location_name (str): Name of the location (e.g., "DHAKA CIRCLE", "CHATTOGRAM CENTRAL", etc.)
        days (int): Number of days to forecast (1-8 days). Default is 5 days. Free tier supports up to 5 days.
        location_type (Optional[str]): Type of the location ("circle", "region", "cluster", "territory")

    Returns:
        WeatherForecastOutput: A structured response containing weather forecast summary,
        file path to detailed data, and prediction statistics.
    """
    try:
        # Get latitude and longitude from LATLNG_MAP
        location_name_upper = location_name.upper()
        if location_name_upper in LATLNG_MAP:
            lat, lng = LATLNG_MAP[location_name_upper]
        else:
            return WeatherForecastOutput(
                location_name=location_name,
                forecast_days=days,
                summary=f"Location '{location_name}' not found in latitude/longitude map.",
                data_path=None,
                status="error",
                is_error=True,
            )

        # Check if OpenWeatherMap API key is configured
        if not settings.OPENWEATHERMAP_API_KEY:
            return WeatherForecastOutput(
                location_name=location_name,
                forecast_days=days,
                summary="OpenWeatherMap API key is not configured. Please set OPENWEATHERMAP_API_KEY in environment variables.",
                data_path=None,
                status="error",
                is_error=True,
            )

        # Validate days parameter
        if days < 1 or days > 8:
            return WeatherForecastOutput(
                location_name=location_name,
                forecast_days=days,
                summary="Days parameter must be between 1 and 8.",
                data_path=None,
                status="error",
                is_error=True,
            )

        # Call OpenWeatherMap 5 Day / 3 Hour Forecast API (free tier)
        # or One Call API 3.0 for more comprehensive forecasts
        api_url = "http://api.openweathermap.org/data/2.5/forecast"
        params = {
            "lat": lat,
            "lon": lng,
            "appid": settings.OPENWEATHERMAP_API_KEY.get_secret_value(),
            "units": "metric",
            "cnt": min(days * 8, 40),  # 8 forecasts per day (3-hour intervals), max 40
        }

        response = requests.get(api_url, params=params, timeout=10)
        response.raise_for_status()

        forecast_data = response.json()

        if "list" not in forecast_data or len(forecast_data["list"]) == 0:
            return WeatherForecastOutput(
                location_name=location_name,
                forecast_days=days,
                summary=f"No forecast data available for {location_name}.",
                data_path=None,
                status="error",
                is_error=True,
            )

        # Process forecast data - group by day
        daily_forecasts = {}

        for item in forecast_data["list"]:
            dt_timestamp = item.get("dt")
            if not dt_timestamp:
                continue

            forecast_datetime = datetime.fromtimestamp(dt_timestamp)
            date_key = forecast_datetime.strftime("%Y-%m-%d")

            if date_key not in daily_forecasts:
                daily_forecasts[date_key] = {
                    "date": date_key,
                    "day_name": forecast_datetime.strftime("%A"),
                    "temps": [],
                    "humidities": [],
                    "pressures": [],
                    "wind_speeds": [],
                    "conditions": [],
                    "descriptions": [],
                    "pop": [],  # Probability of precipitation
                    "rain": [],
                }

            main = item.get("main", {})
            weather_list = item.get("weather", [])
            weather_dict = weather_list[0] if weather_list else {}
            wind = item.get("wind", {})

            daily_forecasts[date_key]["temps"].append(main.get("temp"))
            daily_forecasts[date_key]["humidities"].append(main.get("humidity"))
            daily_forecasts[date_key]["pressures"].append(main.get("pressure"))
            daily_forecasts[date_key]["wind_speeds"].append(wind.get("speed"))
            daily_forecasts[date_key]["conditions"].append(weather_dict.get("main"))
            daily_forecasts[date_key]["descriptions"].append(
                weather_dict.get("description")
            )
            daily_forecasts[date_key]["pop"].append(
                item.get("pop", 0) * 100
            )  # Convert to percentage

            # Handle rain data
            rain = item.get("rain", {})
            rain_3h = rain.get("3h", 0)
            daily_forecasts[date_key]["rain"].append(rain_3h)

        # Calculate daily statistics
        daily_summaries = []
        for date_key in sorted(daily_forecasts.keys())[:days]:
            day_data = daily_forecasts[date_key]

            temps = [t for t in day_data["temps"] if t is not None]
            humidities = [h for h in day_data["humidities"] if h is not None]
            wind_speeds = [w for w in day_data["wind_speeds"] if w is not None]
            pop_values = [p for p in day_data["pop"] if p is not None]

            # Get most common condition
            conditions = [c for c in day_data["conditions"] if c]
            most_common_condition = (
                max(set(conditions), key=conditions.count) if conditions else "N/A"
            )

            # Get most common description
            descriptions = [d for d in day_data["descriptions"] if d]
            most_common_description = (
                max(set(descriptions), key=descriptions.count)
                if descriptions
                else "N/A"
            )

            daily_summaries.append(
                {
                    "date": day_data["date"],
                    "day_name": day_data["day_name"],
                    "temp_min": min(temps) if temps else None,
                    "temp_max": max(temps) if temps else None,
                    "temp_avg": sum(temps) / len(temps) if temps else None,
                    "humidity_avg": sum(humidities) / len(humidities)
                    if humidities
                    else None,
                    "wind_speed_avg": sum(wind_speeds) / len(wind_speeds)
                    if wind_speeds
                    else None,
                    "condition": most_common_condition,
                    "description": most_common_description,
                    "precipitation_prob": max(pop_values) if pop_values else 0,
                    "total_rain": sum(day_data["rain"]),
                }
            )

        if not daily_summaries:
            return WeatherForecastOutput(
                location_name=location_name,
                forecast_days=days,
                summary=f"No forecast data could be processed for {location_name}.",
                data_path=None,
                status="error",
                is_error=True,
            )

        # Format the output
        forecast_info = f"Weather forecast for {location_name}"
        if location_type:
            forecast_info += f" ({location_type})"
        forecast_info += f"\nForecast period: {days} days\n"
        forecast_info += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"

        for idx, day in enumerate(daily_summaries, 1):
            forecast_info += f"Day {idx} - {day['day_name']}, {day['date']}:\n"

            if day["temp_min"] is not None and day["temp_max"] is not None:
                forecast_info += (
                    f"  Temperature: {day['temp_min']:.1f}°C to {day['temp_max']:.1f}°C"
                )
                if day["temp_avg"] is not None:
                    forecast_info += f" (avg: {day['temp_avg']:.1f}°C)"
                forecast_info += "\n"

            if day["humidity_avg"] is not None:
                forecast_info += f"  Humidity: ~{day['humidity_avg']:.0f}%\n"

            if day["condition"]:
                forecast_info += f"  Condition: {day['condition']}"
                if day["description"]:
                    forecast_info += f" ({day['description']})"
                forecast_info += "\n"

            if day["precipitation_prob"] > 0:
                forecast_info += (
                    f"  Precipitation Probability: {day['precipitation_prob']:.0f}%\n"
                )

            if day["total_rain"] > 0:
                forecast_info += f"  Expected Rainfall: {day['total_rain']:.1f} mm\n"

            if day["wind_speed_avg"] is not None:
                forecast_info += f"  Wind Speed: ~{day['wind_speed_avg']:.1f} m/s\n"

            forecast_info += "\n"

        # Add summary
        all_temps = [
            day["temp_avg"] for day in daily_summaries if day["temp_avg"] is not None
        ]
        if all_temps:
            forecast_info += "Summary:\n"
            forecast_info += f"  Average temperature over period: {sum(all_temps) / len(all_temps):.1f}°C\n"

            rainy_days = sum(
                1 for day in daily_summaries if day["precipitation_prob"] > 50
            )
            if rainy_days > 0:
                forecast_info += f"  Days with high chance of rain: {rainy_days}\n"

            hot_days = sum(
                1 for day in daily_summaries if day["temp_max"] and day["temp_max"] > 30
            )
            if hot_days > 0:
                forecast_info += f"  Hot days (>30°C): {hot_days}\n"

        # Save detailed data to JSON file
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        filename = f"{settings.ROOT_PATH}/data/weather/forecast_{timestamp}.json"
        Path(filename).parent.mkdir(parents=True, exist_ok=True)

        detailed_data = {
            "location_name": location_name,
            "location_type": location_type,
            "forecast_days": days,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "daily_summaries": daily_summaries,
            "raw_forecast_data": forecast_data["list"],
        }

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(detailed_data, f, indent=4)

        # Truncate summary to 500 characters to avoid context overload
        summary_text = forecast_info.strip()
        if len(summary_text) > 500:
            summary_text = summary_text[:500] + "..."

        return WeatherForecastOutput(
            location_name=location_name,
            forecast_days=days,
            summary=forecast_info.strip(),
            data_path=filename,
            status="success",
            is_error=False,
        )

    except requests.exceptions.RequestException as e:
        return WeatherForecastOutput(
            location_name=location_name,
            forecast_days=days,
            summary=f"Failed to retrieve weather forecast from OpenWeatherMap API. Error: {repr(e)}",
            data_path=None,
            status="error",
            is_error=True,
        )
    except Exception as e:
        return WeatherForecastOutput(
            location_name=location_name,
            forecast_days=days,
            summary=f"Failed to retrieve weather forecast. Error: {repr(e)}",
            data_path=None,
            status="error",
            is_error=True,
        )


@tool
def visualize_geojson_map(
    location_type: str,
    location_names: Optional[List[str]] = None,
    output_filename: Optional[str] = None,
    center_lat: Optional[float] = None,
    center_lng: Optional[float] = None,
    zoom_start: int = 7,
) -> str:
    """Use this tool to load and visualize GeoJSON boundary data on an interactive map.

    This tool loads location hierarchy boundaries (Circle, Region, Cluster, or Territory)
    from GeoJSON files and creates an interactive Plotly map visualization with hover tooltips
    showing location details. You can optionally filter to show only specific boundaries.

    Args:
        location_type (str): Type of location boundaries to visualize.
                           Options: "circle", "region", "cluster", "territory"
        location_names (Optional[List[str]]): List of specific location names to display.
                                             If None, displays all boundaries.
                                             Names are matched case-insensitively against GeoJSON properties.
        output_filename (Optional[str]): Custom name for the output HTML map file (without extension).
                                        If not provided, auto-generates based on location type and timestamp.
        center_lat (Optional[float]): Latitude for map center. If None, auto-calculates from boundaries.
        center_lng (Optional[float]): Longitude for map center. If None, auto-calculates from boundaries.
        zoom_start (int): Initial zoom level (default: 7)

    Returns:
        str: JSON string containing the Plotly figure for rendering in the UI.
    """
    try:
        # Import plotly here to avoid loading it unless needed
        import plotly.graph_objects as go

        # Normalize location_type
        location_type_lower = location_type.lower()
        valid_types = ["circle", "region", "cluster", "territory"]

        if location_type_lower not in valid_types:
            return json.dumps(
                {
                    "error": f"Invalid location_type '{location_type}'. Must be one of: {', '.join(valid_types)}",
                    "status": "error",
                }
            )

        # Map location types to GeoJSON files
        geojson_files = {
            "circle": f"{settings.ROOT_PATH}/data/Circle_watermarked.geojson",
            "region": f"{settings.ROOT_PATH}/data/Region_watermarked.geojson",
            "cluster": f"{settings.ROOT_PATH}/data/Cluster_watermarked.geojson",
            "territory": f"{settings.ROOT_PATH}/data/Territory_watermarked 1.geojson",
        }

        geojson_path = geojson_files[location_type_lower]

        # Check if file exists
        if not Path(geojson_path).exists():
            return json.dumps(
                {"error": f"GeoJSON file not found: {geojson_path}", "status": "error"}
            )

        # Load GeoJSON data
        with open(geojson_path, "r", encoding="utf-8") as f:
            geojson_data = json.load(f)

        # Extract features
        all_features = geojson_data.get("features", [])
        if not all_features:
            return json.dumps(
                {"error": "No features found in GeoJSON file", "status": "error"}
            )

        # Filter features if location_names provided
        if location_names:
            # Normalize location names for case-insensitive matching
            location_names_lower = [name.lower() for name in location_names]
            filtered_features = []

            for feature in all_features:
                properties = feature.get("properties", {})
                # Check all property values for matches
                for prop_value in properties.values():
                    if (
                        isinstance(prop_value, str)
                        and prop_value.lower() in location_names_lower
                    ):
                        filtered_features.append(feature)
                        break

            if not filtered_features:
                return json.dumps(
                    {
                        "error": f"No boundaries found matching: {', '.join(location_names)}",
                        "status": "error",
                        "available_locations": [
                            feature.get("properties", {}).get("name", "Unknown")
                            for feature in all_features[
                                :10
                            ]  # Show first 10 as examples
                        ],
                    }
                )

            features = filtered_features
        else:
            features = all_features

        # Prepare hover text and IDs for each feature FIRST
        hover_texts = []
        feature_ids = []

        for idx, feature in enumerate(features):
            properties = feature.get("properties", {})
            hover_text = f"<b>{location_type.title()}</b><br>"
            for key, value in properties.items():
                if key != "_index":  # Skip internal index property
                    hover_text += f"{key}: {value}<br>"
            hover_texts.append(hover_text.rstrip("<br>"))

            # Add index-based ID to feature properties for mapping
            if "properties" not in feature:
                feature["properties"] = {}
            feature["properties"]["_index"] = idx
            feature_ids.append(idx)

        # Update geojson_data with features (now with _index properties)
        geojson_data = {"type": "FeatureCollection", "features": features}

        # Calculate center point from features if not provided
        if center_lat is None or center_lng is None:
            all_coords = []
            for feature in features:
                geometry = feature.get("geometry", {})
                if geometry.get("type") == "Polygon":
                    coords = geometry.get("coordinates", [[]])[0]
                    all_coords.extend(coords)
                elif geometry.get("type") == "MultiPolygon":
                    for polygon in geometry.get("coordinates", []):
                        coords = polygon[0] if polygon else []
                        all_coords.extend(coords)

            if all_coords:
                lats = [coord[1] for coord in all_coords]
                lngs = [coord[0] for coord in all_coords]
                calc_center_lat = sum(lats) / len(lats)
                calc_center_lng = sum(lngs) / len(lngs)
                center_lat = calc_center_lat if center_lat is None else center_lat
                center_lng = calc_center_lng if center_lng is None else center_lng

                # Auto-adjust zoom based on feature bounds
                if len(features) < len(all_features):
                    lat_range = max(lats) - min(lats)
                    lng_range = max(lngs) - min(lngs)
                    max_range = max(lat_range, lng_range)
                    # Adjust zoom: smaller range = higher zoom
                    if max_range < 0.5:
                        zoom_start = max(zoom_start, 10)
                    elif max_range < 1.0:
                        zoom_start = max(zoom_start, 9)
                    elif max_range < 2.0:
                        zoom_start = max(zoom_start, 8)
            else:
                # Default to Bangladesh center
                center_lat = 23.8103 if center_lat is None else center_lat
                center_lng = 90.4125 if center_lng is None else center_lng
        else:
            # Use provided values
            if center_lat is None:
                center_lat = 23.8103
            if center_lng is None:
                center_lng = 90.4125

        # Create the Plotly choropleth map
        fig = go.Figure(
            go.Choroplethmapbox(
                geojson=geojson_data,
                locations=feature_ids,
                z=list(range(len(features))),  # Different values for each feature
                featureidkey="properties._index",
                text=hover_texts,
                hovertemplate="%{text}<extra></extra>",
                colorscale=[[0, "lightblue"], [1, "blue"]],
                showscale=False,
                marker=dict(opacity=0.6, line=dict(color="darkblue", width=2)),
            )
        )

        # Update map layout
        map_title = f"{location_type.title()} Boundaries Map"

        fig.update_layout(
            mapbox=dict(
                style="open-street-map",
                center=dict(lat=center_lat, lon=center_lng),
                zoom=zoom_start,
            ),
            title=dict(text=map_title, x=0.5, xanchor="center"),
            margin=dict(l=0, r=0, t=40, b=0),
            height=600,
            hovermode="closest",
        )

        # Convert to dict and handle numpy arrays
        plotly_json = fig.to_dict()
        plotly_json = convert_numpy_to_list(plotly_json)

        # Save the figure data
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        if not output_filename:
            output_filename = f"{location_type_lower}_map_{timestamp}"

        output_dir = f"{settings.ROOT_PATH}/data/images"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_path = f"{output_dir}/{output_filename}.json"

        # Save as JSON
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(plotly_json, f, indent=2)

        # Build description
        description = f"Interactive map showing {len(features)} {location_type} "
        if location_names:
            if len(features) == 1:
                description += f"boundary: {location_names[0]}"
            else:
                description += f"boundaries (filtered from {len(all_features)} total)"
        else:
            description += "boundaries"

        # Return the result as JSON string
        result = {
            "chart_type": "geojson_map",
            "title": map_title,
            "description": description,
            "plotly_json": plotly_json,
            "data_path": output_path,
            "feature_count": len(features),
            "total_features": len(all_features),
            "filtered": location_names is not None,
            "status": "success",
            "is_error": False,
        }

        return json.dumps(result)

    except ImportError:
        return json.dumps(
            {
                "error": "Plotly library is not installed. Please install it with: pip install plotly",
                "status": "error",
            }
        )
    except json.JSONDecodeError as e:
        return json.dumps(
            {
                "error": f"Failed to parse GeoJSON file. Error: {repr(e)}",
                "status": "error",
            }
        )
    except Exception as e:
        return json.dumps(
            {
                "error": f"Failed to create map visualization. Error: {repr(e)}",
                "status": "error",
            }
        )


@tool
def generate_plotly_chart(
    data_path: str,
    custom_code: Optional[str] = None,
    chart_type: Optional[str] = "line",
    x_column: Optional[str] = None,
    y_columns: Optional[List[str]] = None,
    title: Optional[str] = None,
) -> str:
    """Use this tool to generate interactive Plotly charts from JSON data files.

    This tool loads data from a JSON file (typically generated by run_sql_query tool)
    and creates an interactive Plotly visualization. The chart is returned as JSON
    that can be rendered in the UI.

    Useful for creating data visualizations such as:
    - Line charts for trends over time
    - Bar charts for comparisons
    - Scatter plots for correlations
    - Pie charts for proportions
    - Area charts for cumulative data
    - Dual-axis charts
    - Custom Plotly visualizations

    Args:
        data_path (str): Path to JSON file containing the data to visualize
        chart_type (str): Type of chart to generate. Options: "line", "bar", "scatter",
                         "pie", "area", "histogram", "box", "heatmap", "custom" (default: "line")
        x_column (Optional[str]): Column name to use for x-axis. If None, uses first column.
        y_columns (Optional[List[str]]): Column names to use for y-axis. If None, uses all columns except x.
        title (Optional[str]): Custom title for the chart. If None, generates a default title.
        custom_code (Optional[str]): Custom Plotly code to execute. If provided and chart_type="custom",
                                     this code will be executed with 'data' and 'df' variables available.
                                     The code must create a 'fig' variable containing the Plotly figure.
                                     Example:
                                     ```
                                     import plotly.graph_objects as go
                                     from plotly.subplots import make_subplots

                                     fig = make_subplots(specs=[[{"secondary_y": True}]])
                                     fig.add_trace(go.Scatter(x=df['DAY_KEY'], y=df['TOTAL_HITS'], name='Hits'), secondary_y=False)
                                     fig.add_trace(go.Scatter(x=df['DAY_KEY'], y=df['TOTAL_AMT'], name='Amount'), secondary_y=True)
                                     fig.update_layout(title='Dual Axis Chart')
                                     ```

    Returns:
        str: JSON string containing the Plotly figure and chart metadata.
    """
    try:
        # Import plotly here to avoid loading it unless needed
        import pandas as pd
        import plotly.express as px
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        # Load data from JSON file
        if not Path(data_path).exists():
            result = PlotlyChartOutput(
                chart_type=chart_type,
                title="File Not Found",
                description=f"Data file not found: {data_path}",
                plotly_json={},
                data_path=data_path,
                status="error",
                is_error=True,
            )
            return result.model_dump_json()

        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not data:
            result = PlotlyChartOutput(
                chart_type=chart_type,
                title="No Data",
                description="Data file contains no data to visualize",
                plotly_json={},
                data_path=data_path,
                status="error",
                is_error=True,
            )
            return result.model_dump_json()

        # Convert to pandas DataFrame for easier manipulation
        df = pd.DataFrame(data)

        # Detect appropriate columns for x and y axes
        columns = df.columns.tolist()

        # Use provided columns or intelligently select defaults
        if x_column and x_column in columns:
            x_col = x_column
        else:
            x_col = columns[0] if len(columns) > 0 else None

        if y_columns:
            y_cols = [col for col in y_columns if col in columns]
        else:
            y_cols = [col for col in columns if col != x_col]

        # Generate the chart based on chart_type
        fig = None
        chart_title = title if title else f"{chart_type.title()} Chart"
        chart_description = (
            f"Visualization of data from {Path(data_path).name} as {chart_type} chart"
        )

        try:
            # Handle custom Plotly code
            if chart_type == "custom" and custom_code:
                try:
                    # Create a namespace with necessary imports and data
                    namespace = {
                        "data": data,
                        "df": df,
                        "pd": pd,
                        "px": px,
                        "go": go,
                        "make_subplots": make_subplots,
                    }

                    # Execute the custom code
                    exec(custom_code, namespace)

                    # Get the figure from the namespace
                    if "fig" not in namespace:
                        return PlotlyChartOutput(
                            chart_type=chart_type,
                            title="Custom Code Error",
                            description="Custom code must create a 'fig' variable containing the Plotly figure",
                            plotly_json={},
                            data_path=data_path,
                            status="error",
                            is_error=True,
                        ).model_dump_json()

                    fig = namespace["fig"]
                    chart_description = "Custom Plotly visualization"

                except Exception as custom_error:
                    return PlotlyChartOutput(
                        chart_type=chart_type,
                        title="Custom Code Execution Error",
                        description=f"Error executing custom code: {repr(custom_error)}",
                        plotly_json={},
                        data_path=data_path,
                        status="error",
                        is_error=True,
                    ).model_dump_json()

            elif chart_type == "line":
                if len(y_cols) == 1:
                    fig = px.line(df, x=x_col, y=y_cols[0], title=chart_title)
                else:
                    fig = px.line(df, x=x_col, y=y_cols, title=chart_title)
                chart_description = f"Line chart showing trends for {', '.join(y_cols)}"

            elif chart_type == "bar":
                if len(y_cols) == 1:
                    fig = px.bar(df, x=x_col, y=y_cols[0], title=chart_title)
                else:
                    fig = px.bar(df, x=x_col, y=y_cols, title=chart_title)
                chart_description = f"Bar chart comparing {', '.join(y_cols)}"

            elif chart_type == "scatter":
                y_col = y_cols[0] if len(y_cols) > 0 else None
                fig = px.scatter(df, x=x_col, y=y_col, title=chart_title)
                chart_description = f"Scatter plot of {x_col} vs {y_col}"

            elif chart_type == "pie":
                # Pie charts need values and labels
                values_col = y_cols[0] if len(y_cols) > 0 else None
                fig = px.pie(df, values=values_col, names=x_col, title=chart_title)
                chart_description = f"Pie chart showing distribution of {values_col}"

            elif chart_type == "area":
                if len(y_cols) == 1:
                    fig = px.area(df, x=x_col, y=y_cols[0], title=chart_title)
                else:
                    fig = px.area(df, x=x_col, y=y_cols, title=chart_title)
                chart_description = f"Area chart showing cumulative {', '.join(y_cols)}"

            elif chart_type == "histogram":
                col_to_plot = x_col if len(y_cols) == 0 else y_cols[0]
                fig = px.histogram(df, x=col_to_plot, title=chart_title)
                chart_description = f"Histogram showing distribution of {col_to_plot}"

            elif chart_type == "box":
                y_col = y_cols[0] if len(y_cols) > 0 else None
                fig = px.box(df, x=x_col, y=y_col, title=chart_title)
                chart_description = f"Box plot showing distribution of {y_col}"

            elif chart_type == "heatmap":
                # Create a pivot table for heatmap if possible
                if len(columns) >= 3:
                    pivot_df = df.pivot_table(
                        index=columns[0],
                        columns=columns[1],
                        values=columns[2],
                        aggfunc="mean",
                    )
                    fig = px.imshow(pivot_df, title=chart_title, aspect="auto")
                    chart_description = (
                        f"Heatmap of {columns[2]} across {columns[0]} and {columns[1]}"
                    )
                else:
                    # Fall back to correlation heatmap
                    corr_matrix = df.select_dtypes(include=["number"]).corr()
                    fig = px.imshow(
                        corr_matrix, title="Correlation Heatmap", aspect="auto"
                    )
                    chart_description = "Correlation heatmap of numeric columns"
            else:
                # Default to line chart
                fig = px.line(
                    df, x=x_col, y=y_cols[0] if y_cols else None, title=chart_title
                )
                chart_description = "Chart visualization of query results"

            if fig is None:
                result = PlotlyChartOutput(
                    chart_type=chart_type,
                    title="Error",
                    description=f"Could not generate {chart_type} chart from the data",
                    plotly_json={},
                    data_path=data_path,
                    status="error",
                    is_error=True,
                )
                return result.model_dump_json()

            # Update layout for better appearance
            fig.update_layout(
                template="plotly_white",
                hovermode="x unified",
                showlegend=True,
            )

            # Convert figure to JSON and handle numpy arrays
            plotly_json = fig.to_dict()
            plotly_json = convert_numpy_to_list(plotly_json)

            result = PlotlyChartOutput(
                chart_type=chart_type,
                title=chart_title,
                description=chart_description,
                plotly_json=plotly_json,
                data_path=data_path,
                status="success",
                is_error=False,
            )
            return result.model_dump_json()

        except Exception as chart_error:
            result = PlotlyChartOutput(
                chart_type=chart_type,
                title="Chart Generation Error",
                description=f"Error creating {chart_type} chart: {repr(chart_error)}",
                plotly_json={},
                data_path=data_path,
                status="error",
                is_error=True,
            )
            return result.model_dump_json()

    except ImportError:
        result = PlotlyChartOutput(
            chart_type=chart_type,
            title="Import Error",
            description="Plotly or pandas library is not installed. Please install with: pip install plotly pandas",
            plotly_json={},
            data_path=data_path if "data_path" in locals() else None,
            status="error",
            is_error=True,
        )
        return result.model_dump_json()
    except Exception as e:
        result = PlotlyChartOutput(
            chart_type=chart_type,
            title="Error",
            description=f"Failed to generate chart: {repr(e)}",
            plotly_json={},
            data_path=data_path if "data_path" in locals() else None,
            status="error",
            is_error=True,
        )
        return result.model_dump_json()
