"""Phase 2: an AI doctor repairs a damaged real LLM (Qwen3 0.6B). Run with `vf-run wsp-llm`."""

__all__ = ["LlmTaskset"]


def __getattr__(name: str):
    """Load the Verifiers taskset only when asked, so `wsp_llm.oneshot` works without Verifiers installed."""
    if name == "LlmTaskset":
        from wsp_llm.taskset import LlmTaskset
        return LlmTaskset
    raise AttributeError(f"module 'wsp_llm' has no attribute {name!r}")
