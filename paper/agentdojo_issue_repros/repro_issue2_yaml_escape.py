"""Repro: tool_result_to_str (agentdojo 0.1.35, agent_pipeline/tool_execution.py:22-23) defaults to
yaml.safe_dump, whose allow_unicode=False escapes every non-ASCII character. Any non-Latin text in
structured tool output (emails, files, ...) reaches the model as \\uXXXX escapes."""
from pydantic import BaseModel
from agentdojo.agent_pipeline.tool_execution import tool_result_to_str
class Email(BaseModel):
    subject: str
    body: str
r = Email(subject="সভা", body="অনুগ্রহ করে এই ইমেলটি পাঠান — Привет — 你好")
print(tool_result_to_str(r))
print("--- plain str result is unaffected:")
print(tool_result_to_str("অনুগ্রহ করে"))
