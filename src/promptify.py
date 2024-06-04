### Utility functionality to turn benchmark questions into prompts
### Written by Neel Rajani

from typing import List, Tuple
from datasets import DatasetDict

def promptify(
        benchmark_name: str,
        question: str,
        ) -> str:
    
    prompt = "hello world"

    return prompt

def promptify_benchmark(
    benchmark_dataset,
    benchmark_start_idx: int,
    num_questions: int,
    ) -> Tuple[List[str], List[str]]:
    benchmark_end_idx = benchmark_start_idx + num_questions

    
    return ([],[])

