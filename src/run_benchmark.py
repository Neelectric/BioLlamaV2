### Main inference file to run an experiment
### Written by Neel Rajani

from load_benchmark import load_benchmark
from promptify import promptify

def prepare_model_path(
        model_type: str,
        model_size: str,
        model_state: str,
) -> str:
    model_path = "meta-llama/Llama-2-13b-chat-hf"
    return model_path


def run_benchmark(
        model_type: str,
        model_size: str,
        model_state: str,
        benchmark_name: str,
        benchmark_start_idx: int,
        num_questions: int,
        retrieval_corpus: str,
        retriever: str,
        neighbour_length: int,
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
    
    # Identify the correct model path
    model_path = prepare_model_path(
        model_type = model_type,
        model_size = model_size,
        model_state = model_state)
    
    # Create a model object
    llm = llm(
        model_path = model_path, 
        model_size = model_size, 
        model_state = model_state)
    
    # 
    return