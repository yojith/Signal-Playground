import os
import json
from typing import Any

from dataclasses import dataclass
from dotenv import load_dotenv

from openai import OpenAI

import dsp
from signal_module import Signal
from dsp.registry import MODIFY_REGISTRY, ANALYZE_REGISTRY, TOOLS

load_dotenv()  # Load .env file
api_key = os.getenv("OPENAI_API_KEY")  # Get key from environment
client = OpenAI(api_key=api_key)  # Create client


# Response Object
@dataclass
class LLMResponse:
    signal: Signal | None
    analysis: Any | None
    message: str


def _build_analysis_message(conversation: list[dict[str, str]], tool_name: str, analysis_result: Any) -> str:
    function_payload = json.dumps({"analysis": analysis_result}, default=str)
    messages = [
        {
            "role": "system",
            "content": (
                "You are an audio DSP assistant. A tool has returned raw analysis data. "
                "Use that data to generate a clear, concise analysis summary for the user. "
                "Do not invent extra information beyond the raw analysis result."
            ),
        },
        *conversation,
        {
            "role": "user",
            "content": (f"The analysis tool `{tool_name}` returned this raw data: {function_payload}. " "Summarize this analysis for the user in one clear response."),
        },
    ]
    response = client.chat.completions.create(
        model="gpt-5.4-mini",
        messages=messages,
    )

    if not response.choices or not response.choices[0].message or not response.choices[0].message.content:
        return "Analysis complete, but no summary available."
    return response.choices[0].message.content


def process_prompt(prompt: str, signal: Signal, conversation: list[dict[str, str]]) -> LLMResponse:
    response = client.chat.completions.create(
        model="gpt-5.4-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an audio DSP assistant. Choose the correct DSP tool to modify or analyze audio signals. "
                    "For analyze tools, return the requested analysis result as a structured value so the app can summarize it."
                ),
            },
            *conversation,
        ],
        tools=TOOLS,
        tool_choice="auto",
    )

    msg = response.choices[0].message
    if not msg.tool_calls:
        return LLMResponse(signal=signal, analysis=None, message="No DSP action taken.")

    # Execute Tool
    tool_call = msg.tool_calls[0]
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)

    if name not in MODIFY_REGISTRY and name not in ANALYZE_REGISTRY:
        return LLMResponse(signal=signal, analysis=None, message=f"Unknown DSP tool: {name}")

    if name in MODIFY_REGISTRY:
        fn = MODIFY_REGISTRY[name]
        out_signal = fn(signal, **args)
        return LLMResponse(signal=out_signal, analysis=None, message=f"Executed `{name}` with {args}")

    if name in ANALYZE_REGISTRY:
        fn = ANALYZE_REGISTRY[name]
        analysis_result = fn(signal, **args)
        message = _build_analysis_message(conversation, name, analysis_result)
        return LLMResponse(signal=signal, analysis=analysis_result, message=f"Executed `{name}` with {args}\n\n{message}")
