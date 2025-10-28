import re
import yaml
from functools import lru_cache
from pathlib import Path
from typing import Dict

from semantic_layer.models import Configs
from settings import settings


def load_yml_config(path: str | Path) -> Configs:
    if isinstance(path, str):
        path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open() as f:
        raw = yaml.safe_load(f)

    config = Configs(**raw)

    # Resolve constraints inline in each table
    for table in config.tables.values():
        table.constraints = [config.constraints[c].sql for c in table.constraints]

    return config


CONFIG_PATH = Path(settings.ROOT_PATH) / "src" / "semantic_layer" / "configs.yaml"
CONFIG = load_yml_config(str(CONFIG_PATH))


@lru_cache()
def build_keyword_table_maps() -> Dict[str, Dict[str, str]]:
    """
    Builds a mapping of keywords to table names for each app.

    Returns:
        A dictionary where keys are app names and values are dictionaries
        mapping keywords to table names.
    """
    app_keyword_map: Dict[str, Dict[str, str]] = {}

    for app_name, app_config in CONFIG.apps.items():
        keyword_to_table = {}

        for table_name in app_config.tables:
            table_config = CONFIG.tables.get(table_name)
            if not table_config:
                continue

            for kpi_name in table_config.kpis:
                kpi_config = CONFIG.kpis.get(kpi_name)
                if not kpi_config:
                    continue

                for keyword in kpi_config.keywords:
                    keyword_to_table[keyword.lower()] = table_name

        app_keyword_map[app_name] = keyword_to_table

    return app_keyword_map


def select_table(query: str, app: str) -> str:
    """
    Selects the appropriate table based on keywords in the query for the given app.

    Args:
        query: The user's query string.
        app: The application identifier.

    Returns:
        The name of the table to use.
    """
    app_config = CONFIG.apps.get(app)
    if not app_config:
        raise ValueError(f"No config found for app '{app}'")

    keyword_table_map = build_keyword_table_maps().get(app, {})

    for keyword, table_name in keyword_table_map.items():
        if re.search(rf"\b{re.escape(keyword)}\b", query, re.IGNORECASE):
            return table_name

    if app_config.default_table:
        return app_config.default_table

    raise ValueError(
        f"No matching keywords found and no default_table set for app '{app}'"
    )


@lru_cache(maxsize=5)
def build_table_schema_map(schema_dir: Path) -> Dict[str, str]:
    table_schema_map = {}
    for md_file in schema_dir.glob("*.md"):
        table_name = md_file.stem  # filename without extension
        with md_file.open("r", encoding="utf-8") as f:
            table_schema_map[table_name] = f.read()
    return table_schema_map
