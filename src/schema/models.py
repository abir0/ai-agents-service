from enum import StrEnum, auto
from typing import TypeAlias


class Provider(StrEnum):
    FAKE = auto()
    OPENAI = auto()


class OpenAIModelName(StrEnum):
    """https://platform.openai.com/docs/models/gpt-4o"""

    GPT_4O_MINI = "gpt-4o-mini"
    GPT_4O = "gpt-4o"
    GPT_5_1 = "gpt-5.1"


class FakeModelName(StrEnum):
    """Fake model for testing."""

    FAKE = "fake"


AllModelEnum: TypeAlias = (
    OpenAIModelName
    | FakeModelName
)
