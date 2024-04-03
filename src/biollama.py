print("started script")
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
print("imported...")

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", device_map = "auto")

print("model loaded!")

prompt = "In the era of generative AI, "
tokenized_prompt = tokenizer.encode(prompt, return_tensors = "pt")
model_output = model.generate(tokenized_prompt,
                            max_new_tokens = 25,
                            )