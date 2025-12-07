from functools import cache
from typing import TypeAlias

from langchain_community.chat_models import FakeListChatModel
from langchain_openai import AzureChatOpenAI, ChatOpenAI

from schema import (
    AllModelEnum,
    FakeModelName,
    OpenAIModelName,
)

_MODEL_TABLE = {
    FakeModelName.FAKE: "fake",
    OpenAIModelName.GPT_5_1: "gpt-5.1",
    OpenAIModelName.GPT_4O: "gpt-4o",
    OpenAIModelName.GPT_4O_MINI: "gpt-4o-mini",
}

ModelT: TypeAlias = (
    ChatOpenAI | AzureChatOpenAI
)


@cache
def get_model(model_name: AllModelEnum, /) -> ModelT:
    # NOTE: models with streaming=True will send tokens as they are generated
    # if the /stream endpoint is called with stream_tokens=True (the default)
    api_model_name = _MODEL_TABLE.get(model_name)
    if not api_model_name:
        raise ValueError(f"Unsupported model: {model_name}")

    if model_name in OpenAIModelName:
        return ChatOpenAI(model=api_model_name, temperature=0.5, streaming=True)
    if model_name in FakeModelName:
        return FakeListChatModel(
            responses=["This is a test response from the fake model."]
        )
