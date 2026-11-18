"""wsp-env: an RL environment where an AI doctor repairs a broken neural network by editing its weights."""
from wsp_env.taskset import WspTaskset

__all__ = ["WspTaskset"]


def main() -> None:
    print("wsp-env: run it with `uv run vf-run wsp-env`")
