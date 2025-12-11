"""Core functionality for generating synthetic documents and comment threads."""

import textwrap
from typing import List, Optional
from src import llms, models, utils

_TOPICS: List[str] = [
    "a news report on a local incident",
    "a restaurant review for a food blog",
    "a quarterly update for the executive team",
    "an email to all the company employees communicating a new office policy",
    "an instructions manual for an electronics product",
    "an advertisement for a new product launch",
    "a statement from a political campaign",
    "an execution plan for a project at work",
    "a new research proposal for a grant",
    "an opinion piece for a newspaper",
    "a performance review for an employee who has great results but a bad attitude",
    "a rejection email to a job candidate who was interviewed by the whole team",
    "a post-mortem incident report explaining why a server crashed",
    "a product roadmap proposal that cuts a popular feature to meet a deadline",
    "a press release apologizing for a data breach",
    "a social media post responding to a viral rumor about the company",
    "a travel blog post that negatively describes a popular tourist destination",
    "a synopsis for a documentary about a polarizing historical figure"
]


_DOCUMENT_PROMPT_TEMPLATE = textwrap.dedent(
    """
You are writing a short document (3 paragraphs) about: {topic}.

Context: Write this document as if you are a professional who is slightly rushed or has a very strong, specific opinion on the matter.
"""
).strip()


_PEER_DISAGREEMENT_PROMPT_TEMPLATE = textwrap.dedent(
    """
You are a critical peer reviewer examining the document below. You are skeptical of the author's arguments and believe they have overlooked key risks or facts.

Document:
{document}

Your Task:
1. Identify the specific sentence in the text that you find most problematic. Do NOT look for grammatical errors or style issues. Look for a sentence that is factually questionable, logically weak, or presents a perspective you strongly disagree with.
2. Write a comment addressing the author.

Style Guidelines for the Comment:
- Be assertive and direct. State the problem as a fact, not an opinion.
- Do NOT use "hedging" language (e.g., avoid "Maybe we should," "I feel like," "Perhaps consider," "Small suggestion").
- Instead of "I think this is too aggressive," say "This is too aggressive."
- Should be brief, no more than 3 sentences.

    Respond with a JSON object that contains exactly the keys "highlighted_sentence" and "comment".
    """).strip()


_AUTHOR_RESPONSE_PROMPT_TEMPLATE = textwrap.dedent(
    """
    You are the author of the document below. One of your peers highlighted thae specific sentence and left comments that you must answer.

    Document:
    {document}

    Highlighted sentence:
    {highlighted_sentence}

    Comment thread so far:
    {thread}

    Your Context:
    You chose your words carefully in the original document. You believe the reviewer's comment is incorrect, missing the bigger picture, or overly cautious. You do NOT want to change the text.

    Your Task:
    Respond to the latest comment.

    Style Guidelines:
    - **Do NOT apologize.** Do not say "sorry," "my apologies," or "my bad."
    - **Do NOT validate.** Do not use phrases like "I see where you're coming from," "That's a fair point," or "I appreciate the feedback."
    - **Be terse and professional.** Treat this as a disagreement between equals. You don't need to smooth things over.
    - **Goal:** Your goal is to convince the reviewer to keep the text as it is.
    - Your comment must address the author.
    - Use 3 sentences max.
    
    What's your response to the comment?
    """
).strip()


_PEER_FOLLOWUP_PROMPT_TEMPLATE = textwrap.dedent(
    """
    You are the peer reviewer who highlighted the sentence in the document shown below. You still disagree with the author.

    Document:
    {document}

    Highlighted sentence:
    {highlighted_sentence}

    Comment thread so far:
    {thread}
    
    Your Context:
    You have read the author's defense and you find it unconvincing. You believe their refusal to change the text is a mistake that will confuse readers or cause legal/technical issues later. You are getting frustrated that they are not taking your feedback seriously.

    Your Task:
    Respond to the author's last comment. You are NOT backing down.

    Style Guidelines:
    - **Escalate.** Don't just repeat your first point—explain why the author's defense is flawed.
    - **Be cold and clinical.** Cut out all pleasantries.
    - **No Capitulation.** Do NOT say "Okay," "I understand," "If you are sure," or "Let's agree to disagree."
    - **Brevity:** 3 sentences max.
    
    This is a comment thread, so your response should be conversational. 
    """
).strip()


def _generate_document(llm: llms.BaseLLMClient, topic: str) -> str:
    """Generate a document on the given topic."""
    return llm.complete(_DOCUMENT_PROMPT_TEMPLATE.format(topic=topic))


def _generate_peer_disagreement(llm: llms.BaseLLMClient, document: str) -> dict:
    """Generate initial peer disagreement with a highlighted sentence."""
    prompt = _PEER_DISAGREEMENT_PROMPT_TEMPLATE.format(document=document)
    raw = llm.complete(prompt)
    data = utils.extract_json_object(raw)
    try:
        highlighted_sentence = data["highlighted_sentence"].strip()
        comment = data["comment"].strip()
    except KeyError as exc:
        raise KeyError(f"Missing key in peer disagreement response: {exc}") from exc
    return {"highlighted_sentence": highlighted_sentence, "comment": comment}


def _generate_author_response(
    llm: llms.BaseLLMClient,
    document: str,
    highlighted_sentence: str,
    thread: List[models.CommentTurn],
) -> str:
    """Generate author's response to peer comments."""
    prompt = _AUTHOR_RESPONSE_PROMPT_TEMPLATE.format(
        document=document,
        highlighted_sentence=highlighted_sentence,
        thread=utils.format_thread_for_prompt(thread),
    )
    return llm.complete(prompt)


def _generate_peer_followup(
    llm: llms.BaseLLMClient,
    document: str,
    highlighted_sentence: str,
    thread: List[models.CommentTurn],
) -> str:
    """Generate peer's follow-up response in the discussion."""
    prompt = _PEER_FOLLOWUP_PROMPT_TEMPLATE.format(
        document=document,
        highlighted_sentence=highlighted_sentence,
        thread=utils.format_thread_for_prompt(thread),
    )
    return llm.complete(prompt)


def generate_conflict_context(
    llm: llms.BaseLLMClient,
    selected_topic: Optional[str] = None,
) -> models.ConflictContext:
    """Generate a complete conflict context with document and comment thread.
    
    Args:
        llm: The LLM to be used for generation.
        selected_topic: The topic of which the conflict needs to be about.

    Returns:
        A ConflictContext object.
    """

    document = _generate_document(llm, selected_topic)
    disagreement = _generate_peer_disagreement(llm, document)
    highlighted_sentence = disagreement["highlighted_sentence"]
    comment_thread: List[models.CommentTurn] = [
        models.CommentTurn(speaker="peer", text=disagreement["comment"])
    ]

    peer_comments = 1

    while peer_comments < 3:
        author_reply = _generate_author_response(
            llm, document, highlighted_sentence, comment_thread
        )
        comment_thread.append(
            models.CommentTurn(speaker="author", text=author_reply.strip())
        )

        peer_reply = _generate_peer_followup(
            llm, document, highlighted_sentence, comment_thread
        )
        comment_thread.append(
            models.CommentTurn(speaker="peer", text=peer_reply.strip())
        )
        peer_comments += 1

    return models.ConflictContext(
        topic=selected_topic,
        document=document.strip(),
        highlighted_sentence=highlighted_sentence.strip(),
        comment_thread=comment_thread,
    )

def get_topic_for_index(index: int) -> str:
    """Returns the correspoinding topic for the index number. """
    return _TOPICS[index % len(_TOPICS)]