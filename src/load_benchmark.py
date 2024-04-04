### Utility functions to load benchmarks from disk and prepare question and answer lists
### Written by Neel Rajani

from typing import Tuple, List


def load_benchmark(
        benchmark_name: str,
        benchmark_stard_idx: int,
        num_questions: int
        ) -> Tuple[List[str], List[str]]:

        benchmark_questions = []
        benchmark_answers = []

        return benchmark_questions, benchmark_answers