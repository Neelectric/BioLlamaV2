### Main inference file to run an experiment
### Written by Neel Rajani

from typing import List
import torch
from src.load_benchmark import load_benchmark
from src.promptify import promptify
from src.model_kitchen import model


def run_benchmark(
        model_type: str,
        model_size: str,
        model_state: str,
        benchmark_name: str,
        benchmark_start_idx: int,
        num_questions: int,
        db_name: str,
        retriever_name: str,
        neighbour_length: int,
        RETRO_layer_ids: List[int]
        ):
    
    # Load the benchmark
    benchmark_questions, benchmark_answers = load_benchmark(benchmark_name = benchmark_name, 
                                                            benchmark_stard_idx = benchmark_start_idx, 
                                                            num_questions = num_questions)
    
    # Turn the questions into prompts
    benchmark_end_idx = benchmark_start_idx + num_questions
    prompts = []
    for question in benchmark_questions[benchmark_start_idx : benchmark_end_idx]:
        promptified_question = promptify(benchmark_name = benchmark_name, question = question)
        prompts.append(promptified_question)
    
    # Create a model object
    torch_dtype = torch.float32 # temporarily hardcoding this
    llm = model(
        model_type = model_type,
        model_size = model_size,
        model_state = model_state,
        torch_dtype = torch_dtype,
        RETRO_layer_ids = RETRO_layer_ids,
        training = False, # If benchmarking, training should never be true
        retriever_name = retriever_name,
        db_name = db_name,
        neighbour_length = neighbour_length)
    
    # 
    return