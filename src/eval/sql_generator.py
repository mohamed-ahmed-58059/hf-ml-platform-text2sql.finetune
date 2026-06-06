import threading
from typing import Optional

from openai import (APIConnectionError, APITimeoutError, InternalServerError,
                    OpenAI, RateLimitError)
from tenacity import (retry, retry_if_exception_type, stop_after_attempt,
                      wait_random_exponential)

from src.model.example import Example
from src.model.sql_output import SqlOutput

CHAT_SYSTEM = (
    "You are a text-to-SQL model. Given a table schema and a question, return one "
    "SQL query that answers it, using the exact table and column names from the schema."
)

RETRYABLE = (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)
_retry = retry(
    retry=retry_if_exception_type(RETRYABLE),
    wait=wait_random_exponential(multiplier=1, max=60),
    stop=stop_after_attempt(8),
    reraise=True,
)


class SqlGenerator:
    def __init__(self, base_url, model, api_key="EMPTY", mode="chat", max_tokens=256,
                 temperature=0.0, token_param="max_tokens", reasoning_effort=None):
        self._client = OpenAI(base_url=base_url, api_key=api_key, max_retries=0)
        self.model = model
        self.mode = mode
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.token_param = token_param
        self.reasoning_effort = reasoning_effort
        self._lock = threading.Lock()
        self.usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                      "cached_tokens": 0, "reasoning_tokens": 0, "failures": 0}

    def generate(self, example: Example) -> Optional[str]:
        return self._complete(example) if self.mode == "completion" else self._chat(example)

    def _fail(self) -> None:
        with self._lock:
            self.usage["failures"] += 1

    def _track(self, resp) -> None:
        u = getattr(resp, "usage", None)
        if not u:
            return
        pd = getattr(u, "prompt_tokens_details", None)
        cd = getattr(u, "completion_tokens_details", None)
        with self._lock:
            self.usage["calls"] += 1
            self.usage["prompt_tokens"] += u.prompt_tokens or 0
            self.usage["completion_tokens"] += u.completion_tokens or 0
            self.usage["cached_tokens"] += getattr(pd, "cached_tokens", 0) or 0
            self.usage["reasoning_tokens"] += getattr(cd, "reasoning_tokens", 0) or 0

    def _params(self) -> dict:
        params = {"model": self.model, self.token_param: self.max_tokens}
        if self.temperature is not None:
            params["temperature"] = self.temperature
        if self.reasoning_effort is not None:
            params["reasoning_effort"] = self.reasoning_effort
        return params

    @_retry
    def _chat_call(self, messages):
        return self._client.beta.chat.completions.parse(
            **self._params(), messages=messages, response_format=SqlOutput,
        )

    @_retry
    def _complete_call(self, prompt):
        return self._client.completions.create(
            **self._params(), prompt=prompt, stop=[";", "\n\n"],
        )

    def _chat(self, example: Example) -> Optional[str]:
        user = f"Schema:\n{example.schema_text()}\n\nQuestion: {example.question}"
        messages = [
            {"role": "system", "content": CHAT_SYSTEM},
            {"role": "user", "content": user},
        ]
        try:
            resp = self._chat_call(messages)
        except Exception:
            self._fail()
            return None
        self._track(resp)
        parsed = resp.choices[0].message.parsed
        return parsed.sql.strip() if parsed and parsed.sql.strip() else None

    def _complete(self, example: Example) -> Optional[str]:
        try:
            resp = self._complete_call(example.create_prompt())
        except Exception:
            self._fail()
            return None
        self._track(resp)
        text = resp.choices[0].text.strip()
        sql = text.splitlines()[0].strip() if text else ""
        return sql or None
