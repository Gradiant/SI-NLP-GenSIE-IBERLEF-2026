
from abc import abstractmethod
import os
from pathlib import Path
import time
from typing import Any, Dict
import tiktoken

from gensie.task import Task
from ..grad_llm import GradLLM
from ..utils import get_prompts
from abc import ABC
import json
from ..categorizer import  fit_schema_to_fields_dict

TIKTOKEN_PATH = Path(__file__).parent.parent.parent / "tiktoken_cache"
os.environ["TIKTOKEN_CACHE_DIR"] = str(TIKTOKEN_PATH)
import tiktoken
ESTIMATOR = tiktoken.get_encoding("cl100k_base")  # compatible con la mayoría de modelos



class StrategyV2(ABC):
    task: Task
    fields: list[str]

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

    def baseline(self):
        """
        Executes the extraction using OpenAI's response_format for strict schema compliance.
        """
        prompt = self.task.get_input_prompt()
        # Call OpenAI with the task's JSON schema
        response = self.llm.client.chat.completions.create(
            model=self.llm.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise data extraction agent.",
                },
                {"role": "user", "content": prompt},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "extraction",
                    "schema": self.task.target_schema,
                    "strict": True,
                },
            },
        )

        # Parse the structured JSON response
        try:
            content = response.choices[0].message.content
            return json.loads(content)
        except (json.JSONDecodeError, AttributeError, IndexError) as e:
            # Fallback for unexpected API errors
            return {"error": f"Failed to parse model response: {str(e)}"}
        except Exception as e:
            print(str(e))
            return {"error": str(e)}

    def baseline_fit_to_fields(self, reasoning_effort="low"):
        """
        Executes the extraction using OpenAI's response_format for strict schema compliance.
        """
        prompt = self.task.get_input_prompt()
        schema = fit_schema_to_fields_dict(self.fields, self.task.target_schema)

        init = time.time()
        # Call OpenAI with the task's JSON schema

        try:
            response = self.llm.client.chat.completions.create(
                model=self.llm.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise data extraction agent.",
                    },
                    {"role": "user", "content": prompt},
                ],
                # reasoning_effort=reasoning_effort,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "extraction",
                        "schema": schema,
                        "strict": True,
                    },
                },
            )
        except Exception as _:
            #try without reasoning effort
            response = self.llm.client.chat.completions.create(
                model=self.llm.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise data extraction agent.",
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "extraction",
                        "schema": schema,
                        "strict": True,
                    },
                },
            )

        exe_time = time.time() - init
        self.times.append(exe_time)
        total_tokens = response.usage.total_tokens if response.usage else None
        if total_tokens:
            self.tokens.append(total_tokens)

        # Parse the structured JSON response
        try:
            content = response.choices[0].message.content
            return json.loads(content)
        except (json.JSONDecodeError, AttributeError, IndexError) as e:
            # Fallback for unexpected API errors
            return {"error": f"Failed to parse model response: {str(e)}"}
        except Exception as e:
            print(str(e))
            return {"error": str(e)}