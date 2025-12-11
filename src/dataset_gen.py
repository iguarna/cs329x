import random
from src import llms, models, task_conflict_generator, agent, judge
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

def generate_conflict_context_dataset(llm: llms.BaseLLMClient, size: int) -> List[models.ConflictContext]:
    """Generate multiple dataset entries in parallel.

    Args:
        llm: The LLM client to use for generation
        size: Number of entries to generate

    Returns:
        List of generated dataset entries
    """
    def generate_single_entry(index: int) -> models.ConflictContext:
        """Helper function to generate a single entry with its index."""
        return task_conflict_generator.generate_conflict_context(llm, selected_topic=task_conflict_generator.get_topic_for_index(index))

    samples: List[models.ConflictContext] = []

    with ThreadPoolExecutor() as executor:
        future_to_index = {executor.submit(generate_single_entry, i): i for i in range(size)}
        for future in as_completed(future_to_index):
            samples.append(future.result())

    return samples

def _generate_paired_interventions_for_scenario(
    llm_1: llms.BaseLLMClient,
    llm_2: llms.BaseLLMClient,
    type1: models.InterventionType,
    type2: models.InterventionType,
    doc_index: int,
    context: models.ConflictContext,
    comments_used: int,
    use_chain_of_thought: bool,
) -> models.InterventionCandidatePair:
    """
    Generate two different agent interventions for the same scenario (document + comments).

    Args:
        llm_1: The LLM client to use for generation 1
        llm_2: The LLM client to use for generation 2
        type1: the intervention type to use for generation 1
        type2: the intervention type to use for generation 2
        doc_index: Index of the document in the dataset
        context: The dataset entry to generate interventions for
        comments_used: Number of comments to include in the scenario
        use_chain_of_thought: Require JSON output with reasoning traces.
    Returns:
        Tuple of two InterventionCandidatePair objects with different intervention types
    """
    # Generate first intervention
    intervention1 = agent.generate_single_intervention(llm_1, context, comments_used, type1, use_chain_of_thought)

    # Generate second intervention
    intervention2 = agent.generate_single_intervention(llm_2, context, comments_used, type2, use_chain_of_thought)

    return models.InterventionCandidatePair(
        doc_index= doc_index,
        comments_used=comments_used,
        intervention_1=intervention1,
        intervention_1_type=type1,
        intervention_2=intervention2,
        intervention_2_type=type2
    )

def _generate_agent_interventions_for_doc(
    doc_idx: int,
    context: models.ConflictContext,
    agent_llm_1: llms.BaseLLMClient,
    agent_llm_2: llms.BaseLLMClient,
    use_hidden_prompt_for_interventions: bool,
    conversation_points: List[int],
    require_json_output: bool
) -> List[models.InterventionCandidatePair]:
    
    results = []

    # Randomly choose two different intervention types
    if use_hidden_prompt_for_interventions:
        intervention_types_1 = [models.InterventionType.COMPROMISE, models.InterventionType.SOCRATIC_QUESTIONING, models.InterventionType.NO_INTERVENTION]
        intervention_types_2 = [models.InterventionType.COMPROMISE, models.InterventionType.SOCRATIC_QUESTIONING]

    else:
        intervention_types_1 = [models.InterventionType.ANY]
        intervention_types_2 = [models.InterventionType.ANY]

    for n_comments in conversation_points:
        for intervention_type_1 in intervention_types_1:
            for intervention_type_2 in intervention_types_2:

                # Generate two interventions for this scenario
                results.append(_generate_paired_interventions_for_scenario(
                    agent_llm_1, 
                    agent_llm_2,
                    intervention_type_1,
                    intervention_type_2,
                    doc_idx, 
                    context, 
                    n_comments,
                    require_json_output
                ))

    return results

def generate_intervention_candidates_dataset(
    documents: List[models.ConflictContext],
    agent_llm_1: llms.BaseLLMClient,
    agent_llm_2: llms.BaseLLMClient,
    use_hidden_prompt_for_interventions: bool,
    conversation_points: List[int],
    use_chain_of_thought: bool
) -> List[models.InterventionCandidatePair]:
    # Generate paired interventions for each document
    intervention_pairs = []

    # Use ThreadPoolExecutor to parallelize all tasks
    with ThreadPoolExecutor() as executor:
        # Submit all tasks
        future_to_idx = {
            executor.submit(
                _generate_agent_interventions_for_doc,
                idx,
                entry,
                agent_llm_1,
                agent_llm_2,
                use_hidden_prompt_for_interventions,
                conversation_points,
                use_chain_of_thought
            ): idx
            for idx, entry in enumerate(documents)
        }

        # Collect results as they complete
        for future in as_completed(future_to_idx):
            doc_idx = future_to_idx[future]
            try:
                results = future.result()
                if results:  # results is a list, extend training_data with all items
                    intervention_pairs.extend(results)
            except Exception as e:
                print(f"Error processing document {doc_idx}: {str(e)}")

    return intervention_pairs

def _judge_who_got_highest_rating(
    judge_llm: llms.BaseLLMClient,
    context: models.ConflictContext,
    intervention1: str,
    intervention2: str,
    n_comments: int,
    agent_output_is_json: bool
) -> int:
    
    document_data = {
        'document': context.document,
        'highlighted_sentence': context.highlighted_sentence,
        'comment_thread': [c.to_dict() for c in context.comment_thread]
    }

    judge1 = judge.judge_rate_intervention(
        judge_llm,
        document_data['document'],
        document_data['highlighted_sentence'],
        document_data['comment_thread'],
        intervention1,
        n_comments,
        agent_output_is_json
    )

    judge2 = judge.judge_rate_intervention(
        judge_llm,
        document_data['document'],
        document_data['highlighted_sentence'],
        document_data['comment_thread'],
        intervention2,
        n_comments,
        agent_output_is_json
    )

    rating1 = judge1['rating']
    rating2 = judge2['rating']

    if rating1 > rating2:
        return 1
    if rating2 > rating1:
        return 2
    return 0

def _judge_select_intervention(
    judge_llm: llms.BaseLLMClient,
    context: models.ConflictContext,
    intervention1: str,
    intervention2: str,
    n_comments: int,
    agent_output_is_json: bool
) -> int:
    
    document_data = {
        'document': context.document,
        'highlighted_sentence': context.highlighted_sentence,
        'comment_thread': [c.to_dict() for c in context.comment_thread]
    }

    judge_result = judge.judge_select_intervention(
        judge_llm,
        document_data['document'],
        document_data['highlighted_sentence'],
        document_data['comment_thread'],
        intervention1,
        intervention2,
        n_comments,
        agent_output_is_json
    )

    selection = judge_result['selection']

    if isinstance(selection, str):
        selection = int(selection)
    
    if selection not in [0, 1, 2]:
        raise ValueError('Wrong selection value!')

    return selection

def _judge_intervention_pair(
        judge_llm: llms.BaseLLMClient,
        entry: models.ConflictContext,
        intervention_pair: Dict,
        judge_method: str,
        agent_output_is_json: bool
) -> Dict:
    
    intervention1 = intervention_pair['intervention_1']
    intervention1_type = intervention_pair['intervention_1_type']
    intervention2 = intervention_pair['intervention_2']
    intervention2_type = intervention_pair['intervention_2_type']
    n_comments = intervention_pair['comments_used']
    doc_idx = intervention_pair['doc_index']

    # Judge both interventions
    if judge_method == 'rating':
        who_won = _judge_who_got_highest_rating(
            judge_llm=judge_llm,
            context=entry,
            intervention1=intervention1,
            intervention2=intervention2,
            n_comments=n_comments,
            agent_output_is_json=agent_output_is_json
        )
    elif judge_method == 'selection':
        who_won = _judge_select_intervention(
            judge_llm=judge_llm,
            context=entry,
            intervention1=intervention1,
            intervention2=intervention2,
            n_comments=n_comments,
            agent_output_is_json=agent_output_is_json
        )
    else:
        raise ValueError('Invalid judge_method value. Must be selection or rating.')

    # Determine accepted and rejected based on ratings
    if who_won == 1:
        accepted = intervention1
        accepted_type = intervention1_type
        rejected = intervention2
        rejected_type = intervention2_type
    elif who_won == 2:
        accepted = intervention2
        accepted_type = intervention2_type
        rejected = intervention1
        rejected_type = intervention1_type
    elif who_won == 0:
        return {}
    else:
        raise ValueError('Invalid who_won value.')
    
    # Create the prompt (same as what the agent sees)
    partial_thread = entry.comment_thread[:n_comments]

    prompt = agent.build_prompt(
        document=entry.document,
        highlighted_sentence=entry.highlighted_sentence,
        comment_thread=partial_thread,
        intervention_instruction="",
        use_chain_of_thought=agent_output_is_json)

    print(f"Using {n_comments} comments\n * Intervention 1: {intervention1}\n * Intervention 2: {intervention2}\n - Judge decision: {who_won}")

    return {
        'doc_index': doc_idx,
        'comments_used': n_comments,
        'prompt': prompt,
        'accepted': accepted,
        'rejected': rejected,
        'accepted_type': accepted_type,
        'rejected_type': rejected_type,
        'accepted_agent': who_won
    }

def generate_judge_dataset(
        intervention_pairs: List[Dict],
        context: List[models.ConflictContext],
        judge_llm: llms.BaseLLMClient,
        judge_method: str = 'selection',
        use_chain_of_thought: bool = True
    ) -> List[Dict]:

    # Generate paired interventions for each document
    judge_results = []

    # Use ThreadPoolExecutor to parallelize all tasks
    with ThreadPoolExecutor() as executor:
        # Submit all tasks
        future_to_idx = {
            executor.submit(
                _judge_intervention_pair,
                judge_llm,
                context[intervention_pair['doc_index']],
                intervention_pair,
                judge_method,
                use_chain_of_thought
            ): idx
            for idx, intervention_pair in enumerate(intervention_pairs)
        }

        # Collect results as they complete
        for future in as_completed(future_to_idx):
            doc_idx = future_to_idx[future]
            try:
                judge_result = future.result()
                if not judge_result:
                    continue
                judge_results.append(judge_result)
            except Exception as e:
                print(f"Error processing document {doc_idx}: {str(e)}")

    return judge_results