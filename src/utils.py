"""Utility functions for the collaborative discussion system."""

import json
import re
from typing import Dict, Iterable, List

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

def load_dataset_from_jsonl(path: str) -> List[models.ConflictContext]:
    """Load dataset entries from a JSONL file."""
    dataset = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            data = json.loads(line)
            comment_thread = [models.CommentTurn(speaker=c['speaker'], text=c['text']) for c in data['comment_thread']]
            entry = models.ConflictContext(
                topic=data['topic'],
                document=data['document'],
                highlighted_sentence=data['highlighted_sentence'],
                comment_thread=comment_thread
            )
            dataset.append(entry)
    return dataset

def format_comment_thread(comment_thread: List[Dict], num_comments: int) -> str:
    """Format the comment thread for display, using only the first num_comments."""
    if not comment_thread or num_comments == 0:
        return "No comments yet."

    comments_to_show = comment_thread[:num_comments]
    lines = []
    for i, comment in enumerate(comments_to_show, 1):
        speaker = comment['speaker'].title()
        text = comment['text']
        lines.append(f"[Comment {i}] {speaker}: {text}")

    return '\n\n'.join(lines)

def extract_json_from_rater_judge_response(text: str) -> Dict:
    """Extract JSON object from LLM response (alternative implementation for judge)."""
    text = text.strip()
    if not text:
        raise ValueError("LLM returned empty response")

    # Try to find JSON in markdown code block first
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        # Try to find a JSON object anywhere in the text
        match = re.search(r"\{[^{}]*\"reasoning\"[^{}]*\"rating\"[^{}]*\}", text, re.DOTALL)
        if match:
            json_str = match.group(0)
        else:
            # Try to find any JSON object
            match = re.search(r"(\{.*\})", text, re.DOTALL)
            if match:
                json_str = match.group(1)
            else:
                raise ValueError(f"Could not find JSON in response: {text[:200]}")

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON: {e}\nJSON string: {json_str[:200]}")

def extract_json_from_selector_judge_response(text: str) -> Dict:
    """Extract JSON object from LLM response (alternative implementation for judge)."""
    text = text.strip()
    if not text:
        raise ValueError("LLM returned empty response")

    # Try to find JSON in markdown code block first
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        # Try to find a JSON object anywhere in the text
        match = re.search(r"\{[^{}]*\"reasoning\"[^{}]*\"selection\"[^{}]*\}", text, re.DOTALL)
        if match:
            json_str = match.group(0)
        else:
            # Try to find any JSON object
            match = re.search(r"(\{.*\})", text, re.DOTALL)
            if match:
                json_str = match.group(1)
            else:
                raise ValueError(f"Could not find JSON in response: {text[:200]}")

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON: {e}\nJSON string: {json_str[:200]}")
