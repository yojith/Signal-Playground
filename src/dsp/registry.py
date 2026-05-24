"""
Assigns all callable functions by the LLM to an accessible dictionary.
"""

import json
from typing import Callable

# This dictionary contains all DSP functions
DSP_REGISTRY: dict[str, Callable] = {}
with open("src/dsp/tools.json", "r") as f:
    TOOLS = json.load(f)  # Load tool definitions from JSON
    print(f"Loaded {len(TOOLS)} DSP tools from JSON.")


def register(name: str):
    def decorator(func: Callable):
        DSP_REGISTRY[name] = func
        return func

    return decorator
