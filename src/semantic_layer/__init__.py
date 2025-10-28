from pathlib import Path

from semantic_layer.utils import build_table_schema_map
from settings import settings

# Build the Table Schema Map
print(f"Loading schemas from directory: {settings.ROOT_PATH}/src/semantic_layer/schemas")
SCHEMA_DIR = Path(settings.ROOT_PATH) / "src" / "semantic_layer" / "schemas"
TABLE_SCHEMA_MAP = build_table_schema_map(SCHEMA_DIR)
