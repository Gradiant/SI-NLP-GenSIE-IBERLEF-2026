
from abc import abstractmethod
from typing import Any, Dict
import tiktoken
from ..grad_llm import GradLLM
from ..utils import get_prompts
from abc import ABC

ESTIMATOR = tiktoken.get_encoding("cl100k_base")  # compatible con la mayoría de modelos



class StrategyV2(ABC):
    def __init__(self, llm: GradLLM):
        self.llm = llm
        self.prompts = get_prompts()
        self.encoder=tiktoken.get_encoding("cl100k_base")  # compatible con la mayoría de modelos

        self.exe_time = 0
        self.estimated_time = 0
        self.max_time = 0

        self.tokens = []
        self.times = []

    def use_model(self, llm: GradLLM):
        self.llm = llm

    @abstractmethod
    def estimate(self, task, fields, token_per_second=None):
        pass

    @abstractmethod
    def execute(self, in_time=0) -> Dict[str, Any]:
        pass