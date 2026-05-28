
from gensie.agents.experimental_baseline.strategies.Categorical import Categorical
from gensie.agents.experimental_baseline.strategies.Complex import Complex
from gensie.agents.experimental_baseline.strategies.Direct import Direct
from gensie.agents.experimental_baseline.strategies.FixedEntities import FixedEntities
from gensie.agents.experimental_baseline.strategies.SoftEntities import SoftEntities

from .categorizer import classify_types, get_types
from statistics import mean

STRATEGIES ={
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

def generate_plan(task, max_time = 60, strategies = STRATEGIES, optimize= True, add_all_fild_strategy = False) -> list:
    schema = task.target_schema
    # #print("Required fields:", schema.get("required", []))
    fields_by_cat = {}
    for field in schema.get("properties", []):
        types = get_types(field, schema)
        cat = classify_types(types)
        fields_by_cat[cat] = fields_by_cat.get(cat, []) + [field]


    plan = []

    for cat, fields in fields_by_cat.items():
        # #print(f"Category: {cat}, Fields: {fields}")
        fields_are_required = any(f in schema.get("required", []) for f in fields)
        strategy = strategies.get(cat, Direct)(llm=None)
        strategy.estimate(task, fields)
        strategy.max_time = max_time/len(fields_by_cat)+1  # Initial max_time allocation

        plan.append({
            "category": cat,
            "fields": fields,
            "priority": 1 if fields_are_required else 2,
            "estimated_time": strategy.estimated_time,
            "strategy": strategy
        })


    if optimize:
        if len(plan) ==4:
            plan = generate_plan(task, max_time, strategies={**strategies, "categorical": Direct})
            if len(plan) ==4:
                plan = generate_plan(task, max_time, strategies={**strategies, "categorical": Direct, "soft_entities": FixedEntities})
        elif len(plan) >=5:
            plan = generate_plan(task, max_time, strategies={**strategies, "categorical": Direct, "soft_entities": FixedEntities})


    if add_all_fild_strategy:
        all_fields = list(schema.get("properties", {}).keys())
        general_strategy = Direct(llm=None)
        general_strategy.estimate(task, all_fields)
        general_strategy.max_time = max_time/len(fields_by_cat)+1

        plan.append({
            "category": "direct",
            "fields": all_fields,
            "priority": 3,
            "estimated_time": general_strategy.estimated_time,
            "strategy": general_strategy
        })

    plan.sort(key=lambda x: x["priority"])

    return plan