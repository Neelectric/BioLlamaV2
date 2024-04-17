from time import time
from src.biollama import BioLlama
import torch


max_new_tokens = 5
prompt = "In the era of generative AI, "
medmcqa2 = """
Question.
Low insulin to glucagon ratio is not seen in:
(A) Glycogen synthesis
(B) Glycogen breakdown
(C) Gluconeogenesis
(D) Ketogenesis
Answer. 
"""
llama_path = "meta-llama/Llama-2-7b-chat-hf"
torch_dtype = torch.float32
retriever_name = "medcpt"
db_name = "pma"
neighbour_length = 32
print("about to instantiate biollama")
biollama = BioLlama(model_path = llama_path,
                    torch_dtype = torch_dtype,
                    RETRO_layer_ids = [19],
                    training = True,
                    retriever_name = retriever_name,
                    db_name = db_name,
                    neighbour_length = neighbour_length)

print("created biollama")

output, num_new_tokens, time_taken = biollama.generate(medmcqa2, max_new_tokens=25)
print(output)
print(torch_dtype)
print(f"newly generated {num_new_tokens}")
print(f"{num_new_tokens / time_taken} t/s")