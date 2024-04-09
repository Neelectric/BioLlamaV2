from time import time
from src.biollama import BioLlama



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
Let's think step by step. To this end, let's first reflect on the question and which concepts will be relevant in answering it. Then, let's brainstorm on how concepts in the answer options relate back to it. Finally, let's evaluate each option in relation to the question, choosing one.
"""

biollama = BioLlama()

time_before = time()
output, num_new_tokens = biollama.generate(tokenized_prompt,
                            # max_new_tokens = max_new_tokens,
                            temperature = 0.01)
time_after = time()
time_taken = time_after - time_before
print(output)
print(f"newly generated {num_new_tokens}")
print(f"{num_generated / time_taken} t/s")