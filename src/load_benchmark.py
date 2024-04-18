### Utility functions to load benchmarks from disk and prepare question and answer lists
### Written by Neel Rajani

from typing import Tuple, List
from time import time
from datasets import load_dataset, DatasetDict

def promptify_bioasq(dataset, benchmark_start_idx, num_questions):
        benchmark_prompts, benchmark_answers = [], []
        benchmark_end_idx = min(benchmark_start_idx + num_questions, len(dataset["train"]))
        for i in range(benchmark_start_idx, benchmark_end_idx):
                question = dataset["train"][i]
                if question["type"] == "factoid":
                        snippets = question["snippets"]
                        body = question["body"]
                        # ideal_answer = question["ideal_answer"]
                        exact_answer = question["exact_answer"]
                        if type(exact_answer) == list:exact_answer = exact_answer[0]
        return (None, None)

def promptify_pubmedqa(dataset, benchmark_start_idx, num_questions):
        return (None, None)

def promptify_medqa_4(dataset, benchmark_start_idx, num_questions):
        return (None, None)

def promptify_medqa_5(dataset, benchmark_start_idx, num_questions):
        return (None, None)

def promptify_medmcqa(dataset, benchmark_start_idx, num_questions):
        return (None, None)


def load_benchmark(
        benchmark_name: str,
        benchmark_start_idx: int,
        num_questions: int
        ):
        time_before = time()
        match benchmark_name:
                case "BioASQ6b":
                        dataset = load_dataset("HiTZ/Multilingual-BioASQ-6B", 'en')
                        benchmark_prompts, benchmark_answers = promptify_bioasq(dataset, benchmark_start_idx, num_questions)
                case "PubMedQA":
                        dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled") # "pqa_artifical"
                        benchmark_prompts, benchmark_answers = promptify_pubmedqa(dataset, benchmark_start_idx, num_questions)
                case "MedQA_4":
                        dataset = load_dataset("GBaker/MedQA-USMLE-4-options")
                        benchmark_prompts, benchmark_answers = promptify_medqa_4(dataset, benchmark_start_idx, num_questions)
                case "MedQA_5":
                        dataset = load_dataset("bigbio/med_qa")
                        benchmark_prompts, benchmark_answers = promptify_medqa_5(dataset, benchmark_start_idx, num_questions)
                case "MedMCQA":
                        dataset = load_dataset("openlifescienceai/medmcqa")
                        benchmark_prompts, benchmark_answers = promptify_medmcqa(dataset, benchmark_start_idx, num_questions)
        time_after = time()

        print(dataset)
        print(f"Time taken to load the dataset: {time_after - time_before} seconds")
        return benchmark_prompts, benchmark_answers 

load_benchmark("BioASQ6b", 10, 1000)
load_benchmark("PubMedQA", 10, 1000)
load_benchmark("MedMCQA", 10, 1000)
load_benchmark("MedQA_4", 10, 1000)
load_benchmark("MedQA_5", 10, 1000)
