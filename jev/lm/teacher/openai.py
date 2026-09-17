"""OpenAI API teacher = LLMBackend pointed at api.openai.com (logprobs supported, guided choice is not:
use mode='logprob' with `structured_outputs` removed by the server-side compatibility flag)."""
from jev.lm.backends.llm_backend import LLMBackend


def openai_teacher(model: str = "gpt-4.1-mini", api_key: str = "") -> LLMBackend:
    return LLMBackend("https://api.openai.com/v1", model, mode="logprob", api_key=api_key)
