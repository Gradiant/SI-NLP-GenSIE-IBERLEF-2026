import time
from typing import Any, Dict

from ..categorizer import fit_schema_to_fields
from ..corector import correct_in_text
from .Strategy import StrategyV2


class Direct(StrategyV2):

    def estimate(self, task, fields, token_per_second=None):

        self.task = task
        ts_schema = fit_schema_to_fields(fields, task.target_schema)

        self.direct_user = "Instrucción:{task.instruction}\n\nTexto de entrada:{task.input_text}\n\nReturn following schema:\n{ts_schema}".format(
            task=task,
            ts_schema=ts_schema
        )
        self.direct_system = self.prompts.get("direct_system", "")
        tokens = len(self.encoder.encode(self.direct_user + self.direct_system))

        if token_per_second:
            self.estimated_time = tokens / token_per_second
        else:
            self.estimated_time = 0.0

    def execute(self, in_time=0)-> Dict[str, Any]:
        init = time.time()

        if self.max_time > self.estimated_time:
            print(f"Direct: Executing slow method. Estimated time: {self.estimated_time:.2f}s, Max time: {self.max_time:.2f}s")
            result = self.slow_method()
        else:
            print(f"Direct: Executing fast method. Estimated time: {self.estimated_time:.2f}s, Max time: {self.max_time:.2f}s")
            result = self.fast_method()

        self.exe_time = time.time() - init
        return result

    def slow_method(self):

        result, tokens, llm_time = self.llm.call_llm(
            user_prompt=self.direct_user,
            system_prompt=self.direct_system,
            seed=43,
            use_schema=True,
            temperature=0.3,
            return_usage=True,
            reasoning_effort="high"
        )
        self.tokens.append(tokens)
        self.times.append(llm_time)

        if not result:
            result, tokens, llm_time = self.llm.direct_call(self.task, seed=43, temperature=0.3, return_usage=True)
            self.tokens.append(tokens)
            self.times.append(llm_time)

        result = correct_in_text(result, self.task.input_text, schema=self.task.target_schema)

        return result

    def fast_method(self):

        result, tokens, llm_time = self.llm.call_llm(
            user_prompt=self.direct_user,
            system_prompt=self.direct_system,
            seed=43,
            use_schema=True,
            temperature=0.3,
            return_usage=True,
            reasoning_effort="low"
        )
        self.tokens.append(tokens)
        self.times.append(llm_time)

        if not result:
            result, tokens, llm_time = self.llm.direct_call(self.task, seed=43, temperature=0.3, return_usage=True)
            self.tokens.append(tokens)
            self.times.append(llm_time)

        result = correct_in_text(result, self.task.input_text, schema=self.task.target_schema)

        return result