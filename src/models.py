"""Shared data models for the collaborative discussion system."""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List
import json


class InterventionType(Enum):
    """Types of agent interventions in discussions."""
    COMPROMISE = "compromise"
    SOCRATIC_QUESTIONING = "socratic"
    NO_INTERVENTION = "no_intervention"
    ANY = ""


@dataclass
class CommentTurn:
    """Represents a single comment in a discussion thread."""
    speaker: str
    text: str

    def to_dict(self) -> Dict[str, str]:
        return {"speaker": self.speaker, "text": self.text}


@dataclass
class ConflictContext:
    """Represents a document with its discussion thread."""
    topic: str
    document: str
    highlighted_sentence: str
    comment_thread: List[CommentTurn]

    def to_dict(self) -> Dict[str, object]:
        return {
            "topic": self.topic,
            "document": self.document,
            "highlighted_sentence": self.highlighted_sentence,
            "comment_thread": [turn.to_dict() for turn in self.comment_thread],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)
