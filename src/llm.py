import os
import json

from dataclasses import dataclass
from dotenv import load_dotenv

from openai import OpenAI

from signal_module import Signal
from dsp.registry import DSP_REGISTRY, TOOLS

load_dotenv()  # Load .env file
api_key = os.getenv("OPENAI_API_KEY")  # Get key from environment
client = OpenAI(api_key=api_key)  # Create client


# Response Object
@dataclass
class LLMResponse:
    signal: Signal
    message: str


def process_prompt(prompt: str, signal: Signal) -> LLMResponse:
    response = client.chat.completions.create(
        model="gpt-5.4-mini",
        messages=[
            {"role": "system", "content": ("You are an audio DSP assistant. Choose the correct DSP tool for modifying audio signals.")},
            {"role": "user", "content": prompt},
        ],
        tools=TOOLS,
        tool_choice="auto",
    )

    msg = response.choices[0].message
    if not msg.tool_calls:
        return LLMResponse(signal=signal, message="No DSP action taken.")

    # Execute Tool
    tool_call = msg.tool_calls[0]
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)

    if name not in DSP_REGISTRY:
        return LLMResponse(signal=signal, message=f"Unknown DSP tool: {name}")

    fn = DSP_REGISTRY[name]
    out_signal = fn(signal, **args)

    return LLMResponse(signal=out_signal, message=f"Executed `{name}` with {args}")
