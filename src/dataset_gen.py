from src import llms, models, task_conflict_generator, agent
from typing import List, Tuple, Dict
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
