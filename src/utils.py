"""Utility functions for the collaborative discussion system."""

import json
import re
from typing import Dict, Iterable

from src import models

def format_thread_for_prompt(thread: Iterable[models.CommentTurn]) -> str:
    """Format a comment thread for display in a prompt."""
    if not thread:
        return "No comments yet."
    lines = []
    for turn in thread:
        lines.append(f"{turn.speaker.title()}: {turn.text}")
    return '\n'.join(lines)


def extract_json_object(text: str) -> Dict[str, str]:
    """Extract a JSON object from LLM response text."""
    text = text.strip()
    if not text:
        raise ValueError("LLM returned an empty string when JSON was expected.")

    # Try to find a JSON object wrapped in markdown code block
    match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        # If not in markdown, try to find a direct JSON object
        # This regex looks for the first '{' and the last '}'
        match = re.search(r"^\s*(\{.*\})\s*$", text, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            raise ValueError(f"Could not find a JSON object in:\n{text}")

    data = json.loads(json_str)
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object from the LLM response.")
    return data


def save_dataset_to_jsonl(entries: Iterable[models.ConflictContext], path: str) -> None:
    """Save dataset entries to a JSONL file."""
    with open(path, "w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(entry.to_json())
            handle.write('\n')
