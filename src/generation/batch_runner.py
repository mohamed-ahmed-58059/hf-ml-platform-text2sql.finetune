from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional, Type, TypeVar

from pydantic import BaseModel
from tqdm import tqdm

from src.generation.llm_client import LLMClient


T = TypeVar("T", bound=BaseModel)


class BatchRunner:
    def __init__(self, client: LLMClient, max_retries: int = 5, concurrency: int = 16):
        self.client = client
        self.max_retries = max_retries
        self.concurrency = concurrency

    def run(
        self,
        items: list,
        messages_of: Callable[[object], list[dict]],
        schema: Type[T],
        on_success: Callable[[object, T], None],
        validate: Optional[Callable[[T], bool]] = None,
    ) -> list:
        pending = list(items)
        for rnd in range(self.max_retries + 1):
            if not pending:
                break
            retry = []
            with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
                futures = {
                    pool.submit(self.client.generate, messages_of(it), schema): it
                    for it in pending
                }
                for fut in tqdm(
                    as_completed(futures),
                    total=len(pending),
                    desc=f"round {rnd + 1}/{self.max_retries + 1}",
                    leave=False,
                ):
                    item = futures[fut]
                    result = fut.result()
                    # fire on_success the instant a result lands -> checkpoint writes live
                    if result is not None and (validate is None or validate(result)):
                        on_success(item, result)
                    else:
                        retry.append(item)
            print(f"round {rnd + 1}/{self.max_retries + 1}: "
                  f"{len(pending) - len(retry)} ok, {len(retry)} to retry")
            pending = retry

        return pending
