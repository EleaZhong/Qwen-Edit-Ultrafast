from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field


def default_name():
    datetime_str = datetime.now().strftime("%m_%d_%H_%M_%S")
    prefix = "Experiment_"
    return prefix + datetime_str


class ExperimentConfig(BaseModel):
    name: str = Field(default_factory=default_name)
    report_dir: Path = Path("reports")
    iterations: int = 10
    cache_compiled: bool = True


class AbstractExperiment:
    def __init__(self, config: ExperimentConfig | None = None):
        self.config = config if config is not None else ExperimentConfig()
    
    def load(self):
        pass

    def optimize(self):
        pass

    def run_once(self):
        pass

    def run(self):
        pass

    def report(self):
        pass
    
    def cleanup(self):
        pass