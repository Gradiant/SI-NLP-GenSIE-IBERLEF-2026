
from gensie.agents.experimental.strategies.Categorical import Categorical
from gensie.agents.experimental.strategies.Complex import Complex
from gensie.agents.experimental.strategies.Direct import Direct
from gensie.agents.experimental.strategies.FixedEntities import FixedEntities
from gensie.agents.experimental.strategies.SoftEntities import SoftEntities

from .categorizer import classify_types, get_types
from statistics import mean

strategies ={
    "direct": Direct,
    "categorical": Categorical,
    "fixed_entities": FixedEntities,
    "soft_entities": SoftEntities,
    "complex": Complex,
}
def update_times(plan:list, remaining_time:float, task):
    token_per_second = mean([sum(step["strategy"].tokens)/sum(step["strategy"].times) for step in plan if step.get("strategy") and step["strategy"].tokens and step["strategy"].times])

    remaining_strategies = [step for step in plan if step.get("strategy") and step["strategy"].exe_time == 0]

    time_per_strategy = remaining_time / len(remaining_strategies) if remaining_strategies else 0

    for step in plan:
        if step in remaining_strategies:
            step["strategy"].max_time = time_per_strategy
            step["strategy"].estimate(task, step["fields"], token_per_second=token_per_second)
    return plan

def generate_plan(task, max_time = 60):
    schema = task.target_schema
    # #print("Required fields:", schema.get("required", []))
    fields_by_cat = {}
    for field in schema.get("properties", []):
        types = get_types(field, schema)
        cat = classify_types(types)
        fields_by_cat[cat] = fields_by_cat.get(cat, []) + [field]

    plan =[]

    for cat, fields in fields_by_cat.items():
        # #print(f"Category: {cat}, Fields: {fields}")
        fields_are_required = any(f in schema.get("required", []) for f in fields)
        strategy = strategies.get(cat, Direct)(llm=None)
        strategy.estimate(task, fields)
        strategy.max_time = max_time/len(fields_by_cat)  # Initial max_time allocation

        plan.append({
            "category": cat,
            "fields": fields,
            "priority": 1 if fields_are_required else 2,
            "estimated_time": strategy.estimated_time,
            "strategy": strategy
        })

    plan.sort(key=lambda x: x["priority"])

    return plan