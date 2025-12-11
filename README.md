# AI collaboration facilitation

This is a final project for Stanford CS329X in Fall 2025.

See [the final writeup](AI_collaboration_facilitators.pdf) for a complete description of the project.

## Abstract

AI assistants have demonstrated impressive capabilities in supporting individuals on a wide range of text-based tasks, yet there has been relatively limited progress in integrating these assistants into multi-human collaboration settings. From a human-centered design perspective, agents must enhance and foster human collaboration rather than promote isolated work. In this work, we propose a framework to train collaboration facilitator assistants, capable of intervening in contentious comment threads to help drive the discussion to more efficient and constructive resolution. We create a synthetic dataset of simulated collaborative writing stalemates, and utilize preference labels from an LLM-as-a-judge to align a 4B-parameter open-weights model via Direct Preference Optimization (DPO). Our results demonstrate that while DPO significantly improves the agent's win-rate against its base instruction-tuned counterpart, the model struggles to implicitly learn an intervention selection policy solely from preference labels. Furthermore, we provide a quantitative analysis of LLM-as-a-judge limitations and propose new research directions to advance collaboration facilitator assistants.

## Source code

To reproduce the results in this paper, the following scripts must be run in order:

* training_data_gen.ipynb (can be run locally, doesn't require a GPU)
* training_dpo.ipynb (requires a GPU, and is ready to run on Google Colab)
* eval_inference.ipynb (requires a GPU, and is ready to run on Google Colab)
* eval_analysis.ipynb (can be run locally, doesn't require a GPU)

Before running the scripts, you will need to set up your Open AI API key as an environment variable (OPENAI_API_KEY).

