"""Multilingual indirect-prompt-injection attacks for AgentDojo (research artefact).

Importing this package registers every MultilingualAttack condition with
AgentDojo's attack registry. Use it as:

    python -m agentdojo.scripts.benchmark --model ... -ml attacks --attack bn_native
"""
from attacks.multilingual_attack import (  # noqa: F401
    CONDITIONS,
    MultilingualAttack,
    load_payloads,
)
