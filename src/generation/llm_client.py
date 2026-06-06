from typing import Optional, Type, TypeVar

from openai import OpenAI
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "EMPTY",
        temperature: float = 0.2,
        max_tokens: int = 64,
    ):
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, messages: list[dict], schema: Type[T]) -> Optional[T]:
        try:
            resp = self._client.beta.chat.completions.parse(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                messages=messages,
                response_format=schema,
            )
            return resp.choices[0].message.parsed
        except Exception:
            return None
