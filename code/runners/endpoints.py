"""Pipeline builders for the two allowed routes (research-plan.md 11.3-11.5).

* local : an OpenAI-compatible server (llama.cpp llama-server / Ollama / vLLM) on
  localhost. Built as an agentdojo OpenAILLM, so native tool calling and the
  OpenAI-only tool_filter defence work (section 11.2).
* gemini : Google Gemini via google-genai, Vertex (trial credit, section 11.4) or
  AI Studio. Uses agentdojo's GoogleLLM.

NEVER builds an OpenRouter / OpenAI-hosted / Mistral-hosted client: those are
excluded on terms grounds (section 11, 11.5).
"""
from __future__ import annotations

import os

from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline, PipelineConfig

_EXCLUDED = ("openrouter.ai", "api.openai.com", "api.mistral.ai")


def _check_base_url(base_url: str):
    for bad in _EXCLUDED:
        if bad in (base_url or ""):
            raise RuntimeError(
                f"Refusing to use excluded endpoint '{base_url}'. Section 11 excludes "
                f"OpenRouter, the OpenAI hosted API and the Mistral hosted API.")


def build_local_pipeline(base_url=None, model_id=None, defense=None, api_key="EMPTY"):
    import openai
    from agentdojo.agent_pipeline.llms.openai_llm import OpenAILLM

    base_url = base_url or f"http://localhost:{os.environ.get('LOCAL_LLM_PORT', '8099')}/v1"
    _check_base_url(base_url)
    client = openai.OpenAI(api_key=api_key, base_url=base_url)
    if model_id is None:
        model_id = client.models.list().data[0].id
    llm = OpenAILLM(client, model_id)
    cfg = PipelineConfig(llm=llm, model_id=model_id, defense=defense,
                         system_message_name=None, system_message=None)
    pipe = AgentPipeline.from_config(cfg)
    pipe.name = f"local/{os.path.basename(model_id)}"
    if defense:
        pipe.name += f"-{defense}"
    return pipe


def build_gemini_pipeline(model="gemini-3.8-flash", defense=None, use_vertex=True):
    from google import genai
    from agentdojo.agent_pipeline.llms.google_llm import GoogleLLM

    if use_vertex:
        client = genai.Client(vertexai=True, project=os.environ.get("GCP_PROJECT"),
                              location=os.environ.get("GCP_LOCATION", "us-central1"))
    else:
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    llm = GoogleLLM(model, client)
    cfg = PipelineConfig(llm=llm, model_id=None, defense=defense,
                         system_message_name=None, system_message=None)
    pipe = AgentPipeline.from_config(cfg)
    pipe.name = f"gemini/{model}" + (f"-{defense}" if defense else "")
    return pipe
