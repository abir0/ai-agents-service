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
