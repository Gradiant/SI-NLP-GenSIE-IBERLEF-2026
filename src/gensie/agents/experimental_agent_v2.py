import time
from typing import Any, Dict

from gensie.agents.experimental_agent import ExperimentalAgent
from gensie.agents.experimental.planner import generate_plan
from gensie.task import Task

# Seconds remaining in the 60 s budget above which thinking is enabled.
# Kept high (50 s) so thinking only fires when almost the full budget is fresh;
# thinking calls can be slow and a lower threshold causes consistent timeouts.
_THINKING_THRESHOLD = 40

# Strategy categories where thinking is worth the token cost.
# "direct" and "categorical" are single cheap calls — reasoning overhead not justified.
_THINKING_CATEGORIES = {"fixed_entities", "soft_entities", "complex"}

# Substrings (case-insensitive) that identify models supporting thinking/reasoning.
# Add new model families here as needed.
_THINKING_CAPABLE_MODELS = {
    "qwen3",
    "qwq",
    "deepseek-r1",
    "o1",
    "o3",
    "o4",
}


def _supports_thinking(model: str) -> bool:
    m = model.lower()
    return any(substr in m for substr in _THINKING_CAPABLE_MODELS)


class ExperimentalAgentV2(ExperimentalAgent):
    """ExperimentalAgent with time-gated thinking.

    Enables reasoning_effort="low" at the start of the budget when there is
    enough wall-clock time left for the model to think.  Once the remaining
    time drops below _THINKING_THRESHOLD the effort is set to None so the
    remaining strategies finish within the 60-second limit.
    """
    # TODO: se puede hacer una estimación de tiempo para cada paso/estrategia
    def run(self, task: Task, model: str) -> Dict[str, Any]:
        self.initial_time = time.time()
        self.current_task_id = task.id
        self.model = model
        self.llm.model = model

        plan = generate_plan(task)

        results = []
        for step in plan:
            strategy = step.get("strategy", None)
            strategy.use_model(self.llm)

            in_time = self.MAX_TIME - (time.time() - self.initial_time)
            category = step.get("category")
            can_think = _supports_thinking(model) and category in _THINKING_CATEGORIES
            self.llm.reasoning_effort = "low" if (can_think and in_time > _THINKING_THRESHOLD) else None
            self.llm.timeout = max(in_time - 2, 5)  # leave 2 s margin; floor at 5 s

            if not _supports_thinking(model):
                reason = f"model '{model}' not in thinking-capable list"
            elif category not in _THINKING_CATEGORIES:
                reason = f"category '{category}' skipped"
            elif in_time <= _THINKING_THRESHOLD:
                reason = f"in_time {in_time:.1f}s <= threshold {_THINKING_THRESHOLD}s"
            else:
                reason = ""
            print(
                f"[v2] step={category} fields={step.get('fields')} "
                f"in_time={in_time:.1f}s timeout={self.llm.timeout:.1f}s "
                f"thinking={'ON' if self.llm.reasoning_effort else 'OFF'}"
                + (f" ({reason})" if reason else "")
            )

            result = strategy.execute(in_time=in_time)
            results.append(result)

        final_result = {}
        for res in results:
            final_result.update(res)
        return final_result
