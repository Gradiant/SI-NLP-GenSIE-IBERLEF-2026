import os
import time
from typing import Any, Dict
from openai import OpenAI
from gensie.agent import GenSIEAgent
from gensie.task import Task

from gensie.agents.experimental_baseline.planner import generate_plan, update_times
from gensie.agents.experimental_baseline.grad_llm import GradLLM
import json
from deepmerge.merger import Merger

def unique_append(config, path, base, nxt):
    seen = set()

    result = []

    for item in base + nxt:
        key = json.dumps(item, sort_keys=True)

        if key not in seen:
            seen.add(key)
            result.append(item)

    return result


my_merger = Merger(
    [
        (list, [unique_append]),
        (dict, ["merge"]),
        (set, ["union"]),
    ],
    ["override"],
    ["override"]
)

class ExperimentalBaselineAgent(GenSIEAgent):
    # Hard wall-clock limit per task instance (seconds), matching competition rules
    MAX_TIME = 60

    def __init__(self):
        # OpenAI-compatible client — works with local LMStudio, vLLM, or OpenAI
        base_url = os.getenv("OPENAI_BASE_URL")
        api_key = os.getenv("OPENAI_API_KEY", "sk-dummy")
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        # Set at the start of each run() call to enforce MAX_TIME across all phases
        self.initial_time: float = 0.0
        self.current_task_id: str = ""  # Set at the start of run() for log tracing
        self.model: str = ""  # Set at the start of run() for log tracing
        self.llm = GradLLM(self.client, None)


    def run(self, task: Task, model: str) -> Dict[str, Any]:
        self.initial_time = time.time()
        self.current_task_id = task.id
        self.model = model

        self.llm.model = model

        # 1. generate plan
        self.plan = generate_plan(task, self.MAX_TIME, add_all_fild_strategy=True)
        #print("Time generating plan:", time.time() - self.initial_time)

        #print("Processing plan for: ", ",".join([f"{step.get('category')}({step.get('fields', [])})" for step in plan]))
        # TEMP: test direct strategy on all tasks
        # 2. execute strategies
        results = []

        for step in self.plan:
            #print("PROCESSING", step.get("category"), step.get("fields", []))
            strategy = step.get("strategy", None)
            strategy.use_model(self.llm)
            init = time.time()

            if strategy.estimated_time > self.MAX_TIME - (time.time() - self.initial_time):
                print(f"Skipping {step.get('category')} due to time constraints. Estimated: {strategy.estimated_time}, Remaining: {self.MAX_TIME - (time.time() - self.initial_time)}")
                break

            result = strategy.execute()
            if isinstance(result, dict):
                results.append(result)
            elif isinstance(result, list):
                if  len(step.get("fields", [])) == 1:
                    results.append({step.get("fields")[0]: result[0]})
            self.plan = update_times(self.plan, remaining_time=self.MAX_TIME - (time.time() - self.initial_time), task=task)
            print(f"Step {step.get('category')}, Execution Time:", init - self.initial_time)

            if time.time() - self.initial_time > self.MAX_TIME-10:
                print("Time limit exceeded during execution. Stopping further steps.")
                break

        # 3. join results
        final_result = {}
        for res in results:
            my_merger.merge(final_result,res)

        #print("Time joining results:", time.time() - init)
        return final_result
