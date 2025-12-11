"""Core functionality for generating agent interventions in discussions."""

import textwrap
from typing import List
from src import llms, models, utils

_AGENT_INTERVENTION_PROMPT_TEMPLATE = textwrap.dedent(
    """
You are an AI assistant designed to mediate discussions between collaborators. You are observing a conversation between an author and a peer about a document. The peer has highlighted a sentence and left comments. Your goal is to help them resolve their disagreement.

Document:
{document}

Highlighted sentence:
{highlighted_sentence}

Comment thread so far:
{thread}

You can intervene in one of the following ways:

1. SOCRATIC QUESTIONING

Trigger when: The conflict arises from ambiguity. Users are using abstract concepts (e.g., "Make it more punchy" vs. "Keep it professional") without specific examples, or it is unclear why a user objects.

Goal: Reveal the underlying constraint or intent.

Output: A single, neutral, targeted and clarifying question to prompt the human collaborators to resolve the ambiguity themselves.

2. COMPROMISE SYNTHESIS

Trigger when: You can clearly see a path to combine the intent of User A with the structure/constraints of User B (or vice versa).

Goal: Provide a concrete solution.

Output: Propose a new version of the text that attempts to synthesize conflicting user feedback into a coherent whole. Clearly state that you're suggesting to replace the sentence (something like: "I suggest we change this sentence to '....'). You should also explain why your proposal addresses the different viewpoints.

3. NO_INTERVENTION

Trigger when: users problem-solving and expressing similar points of view.

Goal: Avoid interferring if you can't say anything that will help advance the discussion.

Output: exactly the string NO_INTERVENTION.

Instructions for Generation:

First, silently evaluate the conversation against the triggers above.

Use Socratic Questioning if you lack the information to suggest a perfect edit. Use Compromise Synthesis if you believe the synthesis solves both problems. No intervention should be used if the users are actively engaged in problem solving in a constructive manner.

{intervention_instruction}

{output_format}
"""
).strip()

_AGENT_OUTPUT_PLAIN_TEXT = """
Generate a response. If your choice is not to intervene, indicate 'NO_INTERVENTION'. For the other types of interventions, just respond with the comment to be added to the conversation, and don't include anything else.
""".strip()

_AGENT_OUTPUT_JSON = """
**Output Format:**
You must output a single valid JSON object. Do not include markdown formatting (like ```json). Use the following schema:

{
  "reasoning": "A brief analysis of how the discussion between the users is developing, the expressed point of views, and what would be the most appropriate intervention by a helpful assistance at this stage. (Max 4 sentences)",
  "intervention_type": "no_intervention" | "socratic" | "compromise",
  "comment": "The actual comment you want to post. If intervention_type is NO_INTERVENTION, this must be empty."
}
""".strip()

def build_prompt(
    document: str,
    highlighted_sentence: str,
    comment_thread: List[models.CommentTurn],
    intervention_instruction: str,
    use_chain_of_thought: bool
) -> str:
    return _AGENT_INTERVENTION_PROMPT_TEMPLATE.format(
        document=document,
        highlighted_sentence=highlighted_sentence,
        thread=utils.format_thread_for_prompt(comment_thread),
        intervention_instruction=intervention_instruction,
        output_format=_AGENT_OUTPUT_JSON if use_chain_of_thought else _AGENT_OUTPUT_PLAIN_TEXT
    )

def generate_single_intervention(
    llm: llms.BaseLLMClient,
    context: models.ConflictContext,
    comments_used: int,
    intervention_type: models.InterventionType,
    chain_of_thought: bool
) -> str:
    """
    Generate a single intervention for a given scenario.

    Args:
        llm: The LLM client to use for generation
        context: The conflict context to generate interventions for
        comments_used: Number of comments to include from thread
        intervention_type: Type of intervention (InterventionType enum)
        require_json_output: Uses JSON output for chain-of-thought

    Returns:
        The generated intervention text
    """

    output_format = _AGENT_OUTPUT_JSON if chain_of_thought else _AGENT_OUTPUT_PLAIN_TEXT

    partial_thread = context.comment_thread[:comments_used]

    # Determine the intervention instruction based on type
    if intervention_type == models.InterventionType.COMPROMISE:
        intervention_instruction = "In this case, you MUST use a compromise synthesis approach for this intervention."
    elif intervention_type == models.InterventionType.SOCRATIC_QUESTIONING:
        intervention_instruction = "In this case, you MUST use a Socratic questioning approach for this intervention."
    elif intervention_type == models.InterventionType.NO_INTERVENTION:
        intervention_instruction = "In this case, you MUST decide not to intervene in the discussion."
    else:
        intervention_instruction = ""

    prompt = build_prompt(
        document=context.document,
        highlighted_sentence=context.highlighted_sentence,
        thread=partial_thread,
        intervention_instruction=intervention_instruction,
        use_chain_of_thought=chain_of_thought)

    return llm.complete(prompt).strip()
