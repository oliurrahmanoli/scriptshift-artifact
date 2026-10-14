"""Extract the 35 AgentDojo v1.2.2 injection-task goals and metadata, and write
data/goals_en.json. Run with the project venv:  python scripts/extract_goals.py

Everything emitted here is copied verbatim from the installed agentdojo==0.1.35 /
benchmark v1.2.2. No text is invented. Scoring-group labels (A/B/C) follow
research-plan.md section 4.4 and are recorded per task.
"""
import json
from pathlib import Path
from agentdojo.task_suite.load_suites import get_suites

OUT = Path(__file__).resolve().parent.parent / "data" / "goals_en.json"
BENCH = "v1.2.2"

# Scoring sensitivity groups from research-plan.md section 4.4 (keys "suite/injection_task_N").
# A: identifiers only (low risk). B: verbatim copy of environment content (medium).
# C: attacker-specified free text must match exactly (high).
GROUPS = {
    "workspace/injection_task_0": "C", "workspace/injection_task_1": "A",
    "workspace/injection_task_2": "C", "workspace/injection_task_3": "B",
    "workspace/injection_task_4": "C", "workspace/injection_task_5": "A",
    "workspace/injection_task_6": "B", "workspace/injection_task_7": "B",
    "workspace/injection_task_8": "B", "workspace/injection_task_9": "C",
    "workspace/injection_task_10": "A", "workspace/injection_task_11": "A",
    "workspace/injection_task_12": "C", "workspace/injection_task_13": "C",
    "travel/injection_task_0": "A", "travel/injection_task_1": "A",
    "travel/injection_task_2": "C", "travel/injection_task_3": "A",
    "travel/injection_task_4": "A", "travel/injection_task_5": "A",
    "travel/injection_task_6": "C",
    "banking/injection_task_0": "A", "banking/injection_task_1": "A",
    "banking/injection_task_2": "C", "banking/injection_task_3": "A",
    "banking/injection_task_4": "A", "banking/injection_task_5": "A",
    "banking/injection_task_6": "A", "banking/injection_task_7": "A",
    "banking/injection_task_8": "B",
    "slack/injection_task_1": "A", "slack/injection_task_2": "B",
    "slack/injection_task_3": "A", "slack/injection_task_4": "B",
    "slack/injection_task_5": "A",
}

# The three banking tasks whose security() uses `service or "" in t.subject`
# (Python precedence bug), reported in research-plan.md section 4.4.
BANKING_PRECEDENCE_BUG = {
    "banking/injection_task_0", "banking/injection_task_1", "banking/injection_task_3",
}


def main():
    suites = get_suites(BENCH)
    goals = {}
    order = []
    for suite_name, suite in suites.items():
        for task_id, task in suite.injection_tasks.items():
            key = f"{suite_name}/{task_id}"
            goals[key] = {
                "suite": suite_name,
                "injection_task_id": task_id,
                "goal_en": task.GOAL,
                "difficulty": getattr(task, "DIFFICULTY", None).name
                if getattr(task, "DIFFICULTY", None) else None,
                "scoring_group": GROUPS[key],
                "banking_precedence_bug": key in BANKING_PRECEDENCE_BUG,
                "defined_in": type(task).__module__,
            }
            order.append(key)
    payload = {
        "benchmark_version": BENCH,
        "agentdojo_version": "0.1.35",
        "note": "Goals copied verbatim from agentdojo==0.1.35. scoring_group per research-plan.md 4.4.",
        "n_goals": len(goals),
        "order": order,
        "goals": goals,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    counts = {"A": 0, "B": 0, "C": 0}
    for v in goals.values():
        counts[v["scoring_group"]] += 1
    print(f"Wrote {len(goals)} goals to {OUT}")
    print("Group counts:", counts)


if __name__ == "__main__":
    main()
