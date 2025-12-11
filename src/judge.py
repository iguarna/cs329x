"""Core functionality for evaluating agent interventions using an LLM judge."""

import json
import textwrap
from typing import Dict, List
from src import llms, utils

_JUDGE_RATER_INSTRUCTIONS = """
Please rate the intervention on a scale of 0-5 using the evaluation criteria.

Respond with a JSON object containing:
- "reasoning": a brief explanation (2-3 sentences) of your rating
- "rating": a number from 0 to 5

Example response:
```json
{{
  "reasoning": "The intervention effectively synthesizes both perspectives by proposing concrete text that addresses the peer's concern for evidence while maintaining the author's focus on emotional impact. The suggested revision provides a clear path forward that both parties could accept.",
  "rating": 4.5
}}
```
"""

_JUDGE_SELECTOR_INSTRUCTIONS = """
Compare interventions 1 and 2. You must choose a winner according to the evaluation criteria. If both are excellent, choose the one that is more concise or insightful. If both are bad, choose the one that is less harmful.

Respond with a JSON object containing:
- "reasoning": a brief explanation (2-3 sentences) for why the selected intervention is better. 
- "selection": the intervention number that was selected, which can be 1 or 2. Indicate only the number.

Example response:
```json
{{
  "reasoning": "Intervention 2 effectively synthesizes both perspectives by proposing concrete text that addresses the peer's concern for evidence while maintaining the author's focus on emotional impact. The suggested revision provides a clear path forward that both parties could accept.",
  "selection": 2
}}
```
"""


_JUDGE_RATER_INTERVENTIONS_TEMPLATE = textwrap.dedent(
    """

## Agent intervention:
{intervention}
    
    """
).strip()


_JUDGE_SELECTOR_INTERVENTIONS_TEMPLATE = textwrap.dedent(
    """

## Agent intervention 1:
{intervention_1}
    
## Agent intervention 2:
{intervention_2}

    """
).strip()

_JUDGE_PROMPT_TEMPLATE = textwrap.dedent(
    """
You are an expert evaluator assessing the quality of the comments made by an AI assistant in multi-user collaborative discussions.

You will be shown:
1. A document being discussed
2. A highlighted sentence from the document
3. A comment thread between an author and peer reviewer
4. An AI agent's intervention in the discussion. If this says NO_INTERVENTION, it means that the agent decided not to intervene at this point and let the user discussion continue for longer.

Your task is to evaluate how helpful the intervention is in driving the discussion towards a productive conclusion.

## Document:
{document}

## Highlighted Sentence:
{highlighted_sentence}

## Comment Thread:
{comment_thread}

{interventions}

---
Evaluation criteria:

1. **Format**:
Is the intervention formatted as a new comment to be made by the agent in the comment thread?
Examples of bad formatting that should result in a zero rating are:
- Responding as one of the users instead of as the AI assistant (for example, completing the chat with what the next user could say).
- Preceding the comment with a header or an indication of what type of intervention the assistant is using (for example, starting with "Socratic Questioning:").

1. **Relevance**:
Does the intervention address the core disagreement?

2. **Constructiveness**:
Does it propose actionable solutions or insights? A question that is trivial or a suggestion that merely repeats what the users are stating should result in a low rating.
If it's proposing an alternative phrasing, does it include a convincing explanation for why this would be an improvement over the existing text?

3. **Balance**:
Does it fairly consider both perspectives? A proposal that addresses both points of view, or a question that is directed to both users should result in high ratings.

4. **Timeliness**:
Is this a good time to intervene or to stay quiet and let the discussion continue before jumping in?
If the discussion has gotten to a deadlock and stopped progressing, then the assistant must intervene, so an empty intervention should result in a low rating.
An indication of the discussion getting to a deadlock is that comments become reptitive, and users are re-instating the same point of view without any progress. In those cases, not intervening must result in a low rating.
If the users are making significant progress towards a resolution, then a timely comment must add something new or help advance the discussion significantly, otherwise it's better not to intervene.

5. **Effectiveness**:
How likely is it to move the discussion toward conclusion?
Does the assistant add anything new to the discussion?

---

{instructions}
"""
).strip()

def judge_rate_intervention(
    llm: llms.BaseLLMClient,
    document: str,
    highlighted_sentence: str,
    comment_thread: List[Dict],
    intervention: str,
    comments_used: int,
    agent_uses_chain_of_thought: bool = True,
    retries: int = 5,
) -> Dict:
    """
    Use LLM to judge the quality of an intervention.

    Args:
        llm: llms.BaseLLMClient instance to use for evaluation
        document: The document being discussed
        highlighted_sentence: The sentence under discussion
        comment_thread: Full comment thread as list of dicts
        intervention: The intervention text to evaluate (raw agent output)
        comments_used: Number of comments to include in evaluation
        agent_uses_chain_of_thought: If True, extract intervention from JSON format
        retries: Retries left.
    Returns:
        Dictionary with 'rating' and 'reasoning' keys
    """
    # Extract intervention text from JSON if needed
    if agent_uses_chain_of_thought:
        intervention_text, is_valid = _extract_intervention_from_json(intervention)

        # If the agent output is invalid (missing keys), assign a rating of 0
        if not is_valid:
            return {
                "rating": 0.0,
                "reasoning": f"Agent output is invalid or missing required keys: {intervention_text}"
            }
    else:
        intervention_text = intervention

    # Format the comment thread to only include comments that were used
    formatted_thread = utils.format_comment_thread(comment_thread, comments_used)

    # Create the prompt
    prompt = _JUDGE_PROMPT_TEMPLATE.format(
        document=document,
        highlighted_sentence=highlighted_sentence,
        comment_thread=formatted_thread,
        interventions=_JUDGE_RATER_INTERVENTIONS_TEMPLATE.format(intervention=intervention_text),
        instructions=_JUDGE_RATER_INSTRUCTIONS
    )

    # Call the LLM with a system prompt
    system_prompt = "You are an expert evaluator of AI interventions in collaborative discussions."
    response_text = llm.complete(
        prompt=prompt,
        system_prompt=system_prompt,
        max_output_tokens=500
    )

    # Extract and parse the response, retry if needed
    try:
        result = utils.extract_json_from_rater_judge_response(response_text)
    except:
        if retries == 0:
            raise
        result = judge_rate_intervention(
            llm,
            document,
            highlighted_sentence,
            comment_thread,
            intervention,
            comments_used,
            agent_uses_chain_of_thought,
            retries-1
        )
    return result



def judge_select_intervention(
    llm: llms.BaseLLMClient,
    document: str,
    highlighted_sentence: str,
    comment_thread: List[Dict],
    intervention_1: str,
    intervention_2: str,
    comments_used: int,
    agent_uses_chain_of_thought: bool = True,
    retries: int = 5,
) -> Dict:
    """
    Use LLM to judge the quality of an intervention.

    Args:
        llm: llms.BaseLLMClient instance to use for evaluation
        document: The document being discussed
        highlighted_sentence: The sentence under discussion
        comment_thread: Full comment thread as list of dicts
        intervention_1: The first intervention text to evaluate (raw agent output)
        intervention_2: The second intervention text to evaluate (raw agent output)
        comments_used: Number of comments to include in evaluation
        agent_uses_chain_of_thought: If True, extract interventions from JSON format
        retries: Retries left.
    Returns:
        Dictionary with 'selection' and 'reasoning' keys, or 'tie' if both are invalid
    """
    # Extract intervention texts from JSON if needed
    if agent_uses_chain_of_thought:
        intervention_1_text, is_valid_1 = _extract_intervention_from_json(intervention_1)
        intervention_2_text, is_valid_2 = _extract_intervention_from_json(intervention_2)

        # Handle cases where one or both interventions are invalid
        if not is_valid_1 and not is_valid_2:
            # Both invalid - return a tie
            return {
                "selection": "0",
                "reasoning": f"Both interventions are invalid. Intervention 1: {intervention_1_text}. Intervention 2: {intervention_2_text}"
            }
        elif not is_valid_1:
            # Only intervention 1 is invalid - intervention 2 wins
            return {
                "selection": 2,
                "reasoning": f"Intervention 1 is invalid ({intervention_1_text}), so intervention 2 wins by default."
            }
        elif not is_valid_2:
            # Only intervention 2 is invalid - intervention 1 wins
            return {
                "selection": 1,
                "reasoning": f"Intervention 2 is invalid ({intervention_2_text}), so intervention 1 wins by default."
            }
    else:
        intervention_1_text = intervention_1
        intervention_2_text = intervention_2

    # Format the comment thread to only include comments that were used
    formatted_thread = utils.format_comment_thread(comment_thread, comments_used)

    # Create the prompt
    prompt = _JUDGE_PROMPT_TEMPLATE.format(
        document=document,
        highlighted_sentence=highlighted_sentence,
        comment_thread=formatted_thread,
        interventions=_JUDGE_SELECTOR_INTERVENTIONS_TEMPLATE.format(
            intervention_1=intervention_1_text,
            intervention_2=intervention_2_text
        ),
        instructions=_JUDGE_SELECTOR_INSTRUCTIONS
    )

    # Call the LLM with a system prompt
    system_prompt = "You are an expert evaluator of AI interventions in collaborative discussions."
    response_text = llm.complete(
        prompt=prompt,
        system_prompt=system_prompt,
        max_output_tokens=500
    )

    # Extract and parse the response, retry if needed
    try:
        result = utils.extract_json_from_selector_judge_response(response_text)
    except:
        if retries == 0:
            raise
        result = judge_select_intervention(
            llm,
            document,
            highlighted_sentence,
            comment_thread,
            intervention_1,
            intervention_2,
            comments_used,
            agent_uses_chain_of_thought,
            retries-1
        )
    return result


def _extract_intervention_from_json(agent_output: str) -> tuple[str, bool]:
    """
    Extract intervention text from agent's JSON output.

    Args:
        agent_output: The agent's output, either JSON string or plain text

    Returns:
        Tuple of (intervention_text, is_valid) where:
        - intervention_text: The extracted comment text or error message
        - is_valid: True if all expected keys are present, False otherwise
    """
    # Try to parse as JSON
    try:
        data = utils.extract_json_object(agent_output)

        # Check for required keys: reasoning and intervention_type are always required
        required_keys = {"reasoning", "intervention_type"}
        missing_keys = required_keys - set(data.keys())

        if missing_keys:
            return f"[INVALID: Missing keys: {', '.join(missing_keys)}]", False

        # Check if intervention_type is no_intervention
        intervention_type = data.get("intervention_type", "").strip().lower()
        if intervention_type == "no_intervention":
            # For no_intervention, comment is optional/should be empty
            return "NO_INTERVENTION", True

        # For other intervention types, comment is required
        if "comment" not in data:
            return "[INVALID: Missing key: comment]", False

        # Extract the comment field
        comment = data.get("comment", "").strip()

        # Return the comment
        return comment, True

    except (ValueError, json.JSONDecodeError):
        # If it's not JSON, return as-is and mark as invalid
        return agent_output, False


