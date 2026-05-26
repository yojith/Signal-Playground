"""
Assigns all callable functions by the LLM to an accessible dictionary.
"""

import json
from typing import Callable

# This dictionary contains all signal modification functions
MODIFY_REGISTRY: dict[str, Callable] = {}
# This dictionary contains all signal analysis functions
ANALYZE_REGISTRY: dict[str, Callable] = {}

with open("src/dsp/tools.json", "r") as f:
    TOOLS = json.load(f)  # Load tool definitions from JSON
    print(f"Loaded {len(TOOLS)} DSP tools from JSON.")


def register_modify(name: str):
    def decorator(func: Callable):
        MODIFY_REGISTRY[name] = func
        return func

    return decorator


def register_analyze(name: str):
    def decorator(func: Callable):
        ANALYZE_REGISTRY[name] = func
        return func

    return decorator
