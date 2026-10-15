"""LLM factory. Returns an OpenAI-compatible chat model wired to OpenCode Go."""

import os
import uuid
from typing import Any

from langchain_openai import ChatOpenAI


_SESSION_ID = str(uuid.uuid4())


class OpenCodeChatOpenAI(ChatOpenAI):
    """ChatOpenAI wired to the OpenCode Go gateway.

    OpenCode Go rejects ``response_format: json_schema`` (HTTP 400), so
    structured output defaults to function calling. Its thinking mode also
    rejects forced ``tool_choice``, so defaults to ``"auto"``.
    """

    def with_structured_output(
        self, schema: Any, *, method: str | None = None, **kwargs: Any
    ) -> Any:
        if method is None:
            method = "function_calling"
        if method == "function_calling":
            kwargs.setdefault("tool_choice", "auto")
        return super().with_structured_output(schema, method=method, **kwargs)


def get_llm(temperature: float = 0.2) -> ChatOpenAI:
    api_key = os.environ.get("OPENCODE_API_KEY")
    if not api_key:
        raise RuntimeError("OPENCODE_API_KEY is not set — copy .env.example to .env")
    return OpenCodeChatOpenAI(
        model=os.environ.get("LLM_MODEL", "deepseek-v4.1-flash"),
        base_url=os.environ.get("LLM_BASE_URL", "https://opencode.ai/inference/go/v1"),
        api_key=api_key,
        temperature=temperature,
        default_headers={
            "User-Agent": "Day27-HITL-Lab/1.0",
            "x-opencode-session": _SESSION_ID,
        },
    )
