import time

from .Strategy import StrategyV2
from ..categorizer import fit_schema_to_fields
from ..corector import correct_in_text

from typing import Dict, Any




class FixedEntities(StrategyV2):

    def estimate(self, task, fields, token_per_second=None):
        ts_schema = fit_schema_to_fields(fields, task.target_schema)

        user_prompt = "{instruction}\n\nText:{input_text}:\\n\nReturn following schema:\n{ts_schema}".format(
            instruction=task.instruction,
            input_text=task.input_text,
            ts_schema=ts_schema
        )

        direct_system = self.prompts.get("direct_system", "You are a precise data extraction agent.")

        self.user_prompt = user_prompt
        self.task = task
        # Estimación para _direct_call
        tokens = len(self.encoder.encode(self.user_prompt + direct_system))
        self.estimated_time = tokens / token_per_second if token_per_second else 0.0


    def execute(self, in_time=0)-> Dict[str, Any]:
        init = time.time()

        if self.max_time > self.estimated_time:
            print(f"FixedEntities: using slow method with estimated_time={self.estimated_time} (max_time={self.max_time})")
            result = self.slow_method()
        else:
            print(f"FixedEntities: using fast method with estimated_time={self.estimated_time} (max_time={self.max_time})")
            result = self.fast_method()

        self.exe_time = time.time() - init
        return result

    def slow_method(self):
        result = self.llm.direct_call(self.task, use_schema=True,
                                      seed=43, temperature=0.5, user_prompt=self.user_prompt,
                                      system_name="direct_system",
                                      reasoning_effort="high")
        result = correct_in_text(result, self.task.input_text, schema=self.task.target_schema)
        return result

    def fast_method(self):
        result = self.llm.direct_call(self.task, use_schema=True,
                                      seed=43, temperature=0.5, user_prompt=self.user_prompt,
                                      system_name="direct_system")
        result = correct_in_text(result, self.task.input_text, schema=self.task.target_schema)
        return result


    def _direct_call(self):
        result = self.llm.direct_call(self.task, seed=43, temperature=0.5)
        result = correct_in_text(result, self.task.input_text, schema=self.task.target_schema)
        return result