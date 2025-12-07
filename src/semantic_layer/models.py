from typing import Dict, List, Optional

from pydantic import BaseModel


class DBConfig(BaseModel):
    autocommit: Optional[bool] = None
    use_pool: Optional[bool] = None
    pool_size: Optional[int] = None


class App(BaseModel):
    name: str
    tables: List[str]
    db_config: Optional[DBConfig] = None
    default_table: Optional[str] = None
    meta_table: Optional[str] = None
    constraints: List[str] = []
    description: Optional[str] = None


class Table(BaseModel):
    engine: str
    kpis: List[str]
    constraints: List[str] = []
    description: Optional[str] = None


class KPI(BaseModel):
    full_name: str
    keywords: List[str]
    description: Optional[str] = None


class Constraint(BaseModel):
    sql: str
    description: Optional[str] = None


class Configs(BaseModel):
    apps: Dict[str, App]
    tables: Dict[str, Table]
    kpis: Dict[str, KPI]
    constraints: Dict[str, Constraint]
