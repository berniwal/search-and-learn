# ---
# jupyter:
#   jupytext:
#     formats: ipynb,robot
#     text_representation:
#       extension: .robot
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
from datasets import load_dataset

dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
# -

system_prompt = """
Solve the following math problem efficiently and clearly:

- For simple problems (2 steps or fewer):
Provide a concise solution with minimal explanation.

- For complex problems (3 steps or more):
Use this step-by-step format:

{
    "steps": [
        {
            "explanation": "[Concise description]",
            "output": "[Brief explanation and calculations]"
        },
        {
            "explanation": "[Concise description]",
            "output": "[Brief explanation and calculations]"
        },
        ...
    ],
    "final_answer": "$\boxed{answer}$"
}

Where [answer] is just the final number or expression that solves the problem." 
Make sure to put your final answer within \boxed{}.
"""

for data in dataset.shuffle().select(range(5)):
    print(data)
    # print(data.problem)
    # print(data.solution)
    break

# +
from pydantic import BaseModel
from openai import OpenAI
import os

os.environ["OPENAI_API_KEY"] = "sk-proj-_" # Replace with API Key

client = OpenAI()
MODEL = "gpt-4o-2024-08-06" # gpt-4o-2024-08-06 , gpt-4o-mini-2024-07-18

class Step(BaseModel):
    explanation: str
    output: str

class MathReasoning(BaseModel):
    steps: list[Step]
    final_answer: str

# +
import logging
logging.basicConfig(level=logging.WARNING)

def generate_response(problem):
    logger = logging.getLogger(__name__)
    client = OpenAI()
    
    # Init
    response = None
    math_result = None
    
    # Generation
    try:
        response = client.beta.chat.completions.parse(
          model=MODEL,
          messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": problem}
          ],
          response_format=MathReasoning,
          max_tokens=1000,
          temperature=0.8
        )
        math_result = response.choices[0].message.parsed
    except Exception as e:
        logger.warning(f"Error: {e}")
    return response, math_result

# +
import numpy as np

class PRMRating(BaseModel):
    justification: str
    rating: float
    
    
prm_system_prompt = """
You are an expert evaluator for a Preference Ranking Model (PRM) that assesses the quality of step-by-step answers. Your task is to rate the progress on a scale from 0 to 1, where:

0 indicates the steps are incorrect, irrelevant, or unclear.
1 indicates the steps are fully correct, relevant, and clear.

Please provide a rating along with a brief justification for the score in the following format:
    
    {
        "justification": "[Brief explanation for the rating]",
        "rating": [0.0-1.0]
    }
"""
    
def score_response(problem, math_result):
    client = OpenAI()
    responses = []
    scores = []
    for idx in range(1, len(math_result.steps) + 2):
        step_prompt = f"""
        Problem: {problem}
        Current Steps: {math_result.steps[:idx]}
        {'Final Answer:' + str(math_result.final_answer) if idx > len(math_result.steps) else f''}
        """
        
        try:
            prm_completion = client.beta.chat.completions.parse(
              model=MODEL,
              messages=[
                {"role": "system", "content": prm_system_prompt},
                {"role": "user", "content": step_prompt}
              ],
              response_format=PRMRating,
              max_tokens=1000
            )
            prm_rating = prm_completion.choices[0].message.parsed
        except Exception as e:
            prm_rating = PRMRating(justification="Error", rating=0.0)
            prm_completion = {'usage': {'prompt_tokens': 0, 'completion_tokens': 0}}
        responses.append(prm_completion)
        scores.append(prm_rating.rating)
    return responses, scores


def aggregate_scores(scores, agg_method='last'):
    if agg_method == 'last':
        return scores[-1]
    elif agg_method == 'prod':
        return np.prod(scores)
    elif agg_method == 'min':
        return min(scores)
    else:
        raise ValueError(f"Invalid aggregation method: {agg_method}")

# +
def get_cost_request(response_dict):
    # Token Counts
    num_tokens_prompt = response_dict['usage']['prompt_tokens']
    num_tokens_completion = response_dict['usage']['completion_tokens']
    # Token Costs
    cost_per_million_i = 0.15
    cost_per_million_o = 0.6
    # Cost Calculation
    cost = (num_tokens_prompt * cost_per_million_i + num_tokens_completion * cost_per_million_o) / 1e6
    return cost, num_tokens_prompt, num_tokens_completion


def get_cost_problem(responses, scoring_responses):
    # Costs for Responses
    costs = [get_cost_request(response_dict) for response_dict in responses]
    costs_scoring = [sum([get_cost_request(response_dict)[0] for response_dict in scoring_response]) for scoring_response in scoring_responses]
    
    # Total Costs
    total_cost = sum([cost[0] for cost in costs])
    total_cost_scoring = sum(costs_scoring)
    
    # Total Costs
    total_cost = total_cost + total_cost_scoring
    
    return total_cost

# get_cost_request(response)
# sum([get_cost_request(response) for response in responses])

# +
from tqdm import tqdm
def generate_responses(problem, n):
    responses = []
    math_results = []
    for i in range(n):
        response, math_result = generate_response(problem)
        responses.append(response)
        math_results.append(math_result)
    return responses, math_results


def score_responses(problem, math_results):
    scoring_responses = []
    scores = []
    for math_result in math_results:
        if math_result is None:
            # Filter Error Responses
            continue
        response, score = score_response(problem, math_result)
        scoring_responses.append(response)
        scores.append(score)
    return scoring_responses, scores
# -

from sal.utils.math import (
    extract_completion_answers,
    compute_weighted_pred,
    compute_maj_pred,
    compute_naive_pred,
    extract_answer
)

# +
import logging
import time
logging.basicConfig(level=logging.WARNING)

def generation_single_problem(problem_dict, n):
    logger = logging.getLogger(__name__)
    logger.warning(f'Start Problem: {problem_dict["unique_id"]}')
    responses, math_results = generate_responses(problem_dict['problem'], n)
    logger.warning(f'Generated Responses: {problem_dict["unique_id"]}')
    completions = [math_result.final_answer for math_result in math_results if math_result is not None]
    scoring_responses, scores = score_responses(problem_dict['problem'], math_results)
    logger.warning(f'Scored Responses: {problem_dict["unique_id"]}')
    # Sleep for 1 Minute to avoid API Rate Limit
    logger.warning(f'Start Sleep: {problem_dict["unique_id"]}')
    time.sleep(60)
    logger.warning(f'End Sleep: {problem_dict["unique_id"]}')
    return {
        'completions': completions,
        'scores': scores,
        'responses': [response.to_dict() for response in responses if response is not None],
        'scoring_responses': [[response.to_dict() if not isinstance(response, dict) else response for response in scoring_response] for scoring_response in scoring_responses if scoring_response is not None],
    }
# -

dataset_pandas = dataset.to_pandas()
dataset_sample = dataset_pandas.sample(10)

# +
from joblib import Parallel, delayed
n = 32

# Sequential Apply
# generation_responses = dataset_sample.apply(lambda x: generation_single_problem(x, n), axis=1)

# Parallize Apply
# 6 * 32 ~ 200k
generation_responses = Parallel(n_jobs=10, backend='threading')(delayed(generation_single_problem)(x, n) for x in dataset_sample.to_dict(orient='records'))

# +
import pandas as pd

# Extend all Response and Scoring Responses Lists
generation_responses_df = pd.DataFrame(list(generation_responses))
generation_responses_df['ProblemCost'] = generation_responses_df.apply(lambda x: get_cost_problem(x['responses'], x['scoring_responses']), axis=1)
print(f'Total Cost: {generation_responses_df["ProblemCost"].sum()} - Num Problems: {len(generation_responses_df)}')
# -

generation_responses_df

# Add to DataFrame
generation_responses_df = pd.concat([dataset_sample.reset_index(drop=True), generation_responses_df.reset_index(drop=True)], axis=1)

def evaluate_single_problem(generation_row, n):
    aggregate_scores_list = [aggregate_scores(score) for score in generation_row['scores']]
    completion_dict = {
        'completions': generation_row['completions'],
        'agg_scores': aggregate_scores_list,
        'solution': extract_answer(generation_row['solution'], "math")
    }
    preds = extract_completion_answers(x=completion_dict)
    completion_dict.update(preds)
    
    # Filter on Top N
    completion_dict[f'preds@{n}'] = completion_dict['preds'][:n]
    completion_dict[f'agg_scores@{n}'] = completion_dict['agg_scores'][:n]
    
    # Extract Weighted
    pred_weighted = compute_weighted_pred(x=completion_dict, n=n)
    pred_weighted[f'pred_weighted@{n}'] = extract_answer(pred_weighted[f'pred_weighted@{n}'], "math")
    completion_dict.update(pred_weighted)
    
    # Extract Majority
    pred_majority = compute_maj_pred(x=completion_dict, n=n)
    pred_majority[f'pred_maj@{n}'] = extract_answer(pred_majority[f'pred_maj@{n}'], "math")
    completion_dict.update(pred_majority)
    
    # Extract Highest Score
    pred_highest_score = compute_naive_pred(x=completion_dict, n=n)
    pred_highest_score[f'pred_naive@{n}'] = extract_answer(pred_highest_score[f'pred_naive@{n}'], "math")
    completion_dict.update(pred_highest_score)
    
    # Evaluate Correctness
    completion_dict['weighted_correct'] = (completion_dict[f'pred_weighted@{n}'] == completion_dict['solution'])
    completion_dict['majority_correct'] = (completion_dict[f'pred_maj@{n}'] == completion_dict['solution'])
    completion_dict['highest_score_correct'] = (completion_dict[f'pred_naive@{n}'] == completion_dict['solution'])
    
    return completion_dict

def compute_metrics(generation_responses_df, n):
    # Evaluate Results
    evaluation_results = generation_responses_df.apply(lambda x: evaluate_single_problem(x, n), axis=1)
    evaluation_results_df = pd.DataFrame(list(evaluation_results))
    
    # Combine Evaluation Results with DataFrame
    full_df = pd.concat([generation_responses_df.reset_index(drop=True), evaluation_results_df.reset_index(drop=True)], axis=1)
    
    # Metrics for Evaluation
    metrics_df = pd.DataFrame([{'N': n}])
    
    # Overall Metrics
    metrics_overall_df = full_df[['weighted_correct', 'majority_correct', 'highest_score_correct']].mean().to_frame().T
    
    # Metrics for Evaluation Across Difficulty
    metrics_per_level_df = full_df.groupby('level')[['weighted_correct', 'majority_correct', 'highest_score_correct']].mean()
    # Pivot to single row with single index
    metrics_per_level_df['Index'] = 0
    metrics_per_level_df = metrics_per_level_df.reset_index().pivot(columns='level', values=['weighted_correct', 'majority_correct', 'highest_score_correct'], index='Index')
    
    # Combine Metrics
    metrics_df = pd.concat([metrics_df, metrics_overall_df, metrics_per_level_df], axis=1)
    
    return evaluation_results_df, metrics_df

metrics = []
evaluations = []
for n in [2**i for i in range(0, 6)]:
    print(n)
    evaluation_results_df, metrics_df = compute_metrics(generation_responses_df, n)
    evaluations.append(evaluation_results_df)
    metrics.append(metrics_df)

generation_responses_df

evaluations[0]

pd.concat(metrics)
