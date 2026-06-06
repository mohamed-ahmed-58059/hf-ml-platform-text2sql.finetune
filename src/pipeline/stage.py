from abc import ABC, abstractmethod

from src.pipeline.context import Context


class Stage(ABC):
    @abstractmethod
    def run(self, ctx: Context) -> Context:
        ...
