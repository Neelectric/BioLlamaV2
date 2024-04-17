### Utility functions to load benchmarks from disk and prepare question and answer lists
### Written by Neel Rajani

from typing import Tuple, List
from time import time
from datasets import load_dataset

def load_benchmark(
        benchmark_name: str,
        benchmark_start_idx: int,
        num_questions: int
        ) -> Tuple[List[str], List[str]]:

        benchmark_questions = []
        benchmark_answers = []
        time_before = time()
        match benchmark_name:
                case "BioASQ6b":
                        dataset = load_dataset("HiTZ/Multilingual-BioASQ-6B", 'en')
                case "PubMedQA":
                        dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled") # "pqa_artifical"
                case "MedQA_4":
                        dataset = load_dataset("GBaker/MedQA-USMLE-4-options")
                case "MedQA_5":
                        dataset = load_dataset("bigbio/med_qa")
                case "MedMCQA":
                        dataset = load_dataset("openlifescienceai/medmcqa")
        time_after = time()

        print(dataset)
        print(f"Time taken to load the dataset: {time_after - time_before} seconds")
        return benchmark_questions, benchmark_answers


load_benchmark("BioASQ6b", 10, 1000)
load_benchmark("PubMedQA", 10, 1000)
load_benchmark("MedMCQA", 10, 1000)
load_benchmark("MedQA_4", 10, 1000)
load_benchmark("MedQA_5", 10, 1000)
