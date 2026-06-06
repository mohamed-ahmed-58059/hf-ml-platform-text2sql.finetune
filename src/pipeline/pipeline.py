from src.pipeline.context import Context
from src.pipeline.stage import Stage


class Pipeline:
    def __init__(self, stages: list[Stage]):
        self.stages = stages

    def run(self, ctx: Context) -> Context:
        for stage in self.stages:
            ctx = stage.run(ctx)
        return ctx
